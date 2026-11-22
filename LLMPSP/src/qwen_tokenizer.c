#include "falcon_h1.h"

#include <stdlib.h>
#include <string.h>

// 引入正则表达式库，使用处理 UTF-8 字节串的接口。
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// 一些底层实现
// 将连续字节还原成整数，供后面的词表查询使用。
static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// 检查当前位置是否为特殊标记，返回它的编号，并通过 length 返回字节数量。
static int special_at(const FalconTokenizer *tokenizer, const uint8_t *text,
                      int remaining, int *length) {
    int i;
    for (i = 0; i < tokenizer->special_count; ++i) {
        int id = (int)rd32(tokenizer->special_ids + i * 4);
        size_t size;
        const uint8_t *piece = falcon_token_piece(tokenizer, id, &size);
        if (size <= (size_t)remaining && memcmp(text, piece, size) == 0) {
            *length = (int)size;
            return id;
        }
    }
    return -1;
}

// 比较两个字节片段，供二分查找决定下一次查询哪一半词表。
static int piece_compare(const uint8_t *left, size_t left_len,
                         const uint8_t *right, size_t right_len) {
    size_t common = left_len < right_len ? left_len : right_len;
    int result = memcmp(left, right, common);
    if (result) return result;
    return left_len < right_len ? -1 : left_len > right_len;
}

// 预先声明后面定义的函数，供分词过程调用。
static int find_token(const FalconTokenizer *tokenizer,
                      const uint8_t *piece, size_t length);
static int bpe_chunk(const FalconTokenizer *tokenizer,
                     const uint8_t *text, int length,
                     int *output, int capacity);

// Tokenizer 的基础概念
// token 是词表中的文本或字节片段，input_id 是它在词表中的整数编号。
// 多个 token 的编号按顺序组成 input_ids，本文件用 tokens 数组保存它们。

// 1. 分词
// 将普通文本按 Qwen 的规则划分片段，每个片段交给 BPE 生成 token 编号。
// 特殊标记有预先规定的编号，识别后直接写入结果。
int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text,
                    int *tokens, int max_tokens) {
    // 规则处理英文缩写、文字、数字、标点和空白；空格也可能属于片段。
    static const char pattern[] =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    pcre2_code *code;
    pcre2_match_data *match;
    PCRE2_SIZE error_offset;
    int error_code, at = 0, count = 0, length;
    if (!tokenizer || !text || !tokens || max_tokens <= 0) return -1;
    length = (int)strlen(text);
    code = pcre2_compile((PCRE2_SPTR)pattern, PCRE2_ZERO_TERMINATED,
                         PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, NULL);
    if (!code) return -1;
    match = pcre2_match_data_create_from_pattern(code, NULL);
    if (!match) {
        pcre2_code_free(code);
        return -1;
    }
    while (at < length) {
        int end = at, special = -1, special_length = 0, cursor = 0;
        // 找到下一个特殊标记，对它之前的普通文本进行预分词。
        while (end < length) {
            special = special_at(tokenizer, (const uint8_t *)text + end,
                                  length - end, &special_length);
            if (special >= 0) break;
            ++end;
        }
        while (cursor < end - at) {
            PCRE2_SIZE *span;
            int made;
            if (pcre2_match(code, (PCRE2_SPTR)text + at, (PCRE2_SIZE)(end - at),
                             (PCRE2_SIZE)cursor, PCRE2_ANCHORED, match, NULL) < 0)
                goto failure;
            span = pcre2_get_ovector_pointer(match);
            if (span[1] <= span[0]) goto failure;
            // span 给出片段的起止字节位置，BPE 将编号追加到 tokens 数组。
            made = bpe_chunk(tokenizer, (const uint8_t *)text + at + span[0],
                              (int)(span[1] - span[0]), tokens + count,
                              max_tokens - count);
            if (made < 0) goto failure;
            count += made;
            cursor = (int)span[1];
        }
        at = end;
        if (special >= 0) {
            if (count >= max_tokens) goto failure;
            tokens[count++] = special;
            at += special_length;
        }
    }
    pcre2_match_data_free(match);
    pcre2_code_free(code);
    return count;
failure:
    pcre2_match_data_free(match);
    pcre2_code_free(code);
    return -1;
}

// 2. Token -> input_id
// 将一个普通片段编码为编号数组，返回写入数量，失败返回 -1。
// 先取得每个字节的编号，再按词表中的合并优先级生成更大的 token。
static int bpe_chunk(const FalconTokenizer *tokenizer,
                     const uint8_t *text, int length,
                     int *output, int capacity) {
    int *work;
    uint8_t combined[512];
    int count = length, i;
    if (length <= 0) return 0;
    if (length > capacity) return -1;
    work = (int *)malloc((size_t)length * sizeof(int));
    if (!work) return -1;
    // 每个字节先查到一个初始编号；UTF-8 字符可以包含多个字节。
    for (i = 0; i < length; ++i)
        work[i] = (int)rd32(tokenizer->byte_ids + text[i] * 4);
    while (count > 1) {
        uint32_t best_rank = 0xffffffffu;
        int best_index = -1, best_id = -1;
        // 检查所有相邻 token，选择能够合并且 rank 最小的一对。
        for (i = 0; i < count - 1; ++i) {
            size_t left_len, right_len;
            const uint8_t *left = falcon_token_piece(tokenizer, work[i], &left_len);
            const uint8_t *right = falcon_token_piece(tokenizer, work[i + 1], &right_len);
            int merged;
            uint32_t rank;
            if (left_len + right_len > (size_t)tokenizer->max_piece_bytes ||
                left_len + right_len > sizeof(combined))
                continue;
            memcpy(combined, left, left_len);
            memcpy(combined + left_len, right, right_len);
            merged = find_token(tokenizer, combined, left_len + right_len);
            if (merged < 0) continue;
            rank = rd32(tokenizer->ranks + merged * 4);
            if (rank < best_rank) {
                best_rank = rank;
                best_index = i;
                best_id = merged;
            }
        }
        if (best_index < 0) break;
        // 用合并后的编号替换这一对，后面的编号向前移动一个位置。
        work[best_index] = best_id;
        memmove(work + best_index + 1, work + best_index + 2,
                (size_t)(count - best_index - 2) * sizeof(int));
        --count;
    }
    if (count > capacity) {
        free(work);
        return -1;
    }
    memcpy(output, work, (size_t)count * sizeof(int));
    free(work);
    return count;
}

// 3. 查表
// 支持字节片段查询编号，以及编号查询字节片段；分词和 BPE 都会使用。
// 模型的 embedding 查询在推理代码中完成。

// 输入 token 编号，返回对应字节片段的地址，并通过 piece_len 返回长度。
const uint8_t *falcon_token_piece(const FalconTokenizer *tokenizer,
                                  int token, size_t *piece_len) {
    uint32_t begin, end;
    if (piece_len) *piece_len = 0;
    if (!tokenizer || token < 0 || token >= tokenizer->vocab_size) return NULL;
    begin = rd32(tokenizer->offsets + token * 4);
    end = rd32(tokenizer->offsets + (token + 1) * 4);
    if (piece_len) *piece_len = end - begin;
    return tokenizer->pieces + begin;
}

// 输入字节片段，在按内容排序的词表索引中二分查找，返回编号或 -1。
static int find_token(const FalconTokenizer *tokenizer,
                      const uint8_t *piece, size_t length) {
    int low = 0, high = tokenizer->vocab_size - 1;
    while (low <= high) {
        int middle = low + (high - low) / 2;
        int id = (int)rd32(tokenizer->sorted_ids + middle * 4);
        size_t candidate_len;
        const uint8_t *candidate = falcon_token_piece(tokenizer, id, &candidate_len);
        int comparison = piece_compare(piece, length, candidate, candidate_len);
        if (comparison == 0) return id;
        if (comparison < 0) high = middle - 1;
        else low = middle + 1;
    }
    return -1;
}

// 对话输入的组织
// 为用户文本添加角色和消息边界标记，组成模型需要的 input_ids。
static int append_text(const FalconTokenizer *tokenizer, const char *text,
                       int *tokens, int count, int capacity) {
    int made = falcon_tokenize(tokenizer, text, tokens + count, capacity - count);
    return made < 0 ? -1 : count + made;
}

int falcon_build_chat_prompt(const FalconModel *model, const char *prompt,
                             int *tokens, int max_tokens) {
    int count = 0;
    if (!model || !prompt || !tokens || max_tokens < 4) return -1;
    tokens[count++] = model->config.im_start_id;
    count = append_text(&model->tokenizer, "user\n", tokens, count, max_tokens);
    if (count < 0) return -1;
    count = append_text(&model->tokenizer, prompt, tokens, count, max_tokens);
    if (count < 0 || count >= max_tokens) return -1;
    tokens[count++] = model->config.im_end_id;
    count = append_text(&model->tokenizer, "\n", tokens, count, max_tokens);
    if (count < 0 || count >= max_tokens) return -1;
    tokens[count++] = model->config.im_start_id;
    return append_text(&model->tokenizer, FALCON_ASSISTANT_PREFIX,
                       tokens, count, max_tokens);
}

