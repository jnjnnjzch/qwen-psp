#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "tokenizer_internal.h"
#include <string.h>

/* 官方 Qwen 预分词规则：将普通文本切成适合独立执行 BPE 的连续片段。
 * 片段边界会限制哪些相邻字节可以合并，因此这一步也影响最后的 token 编号。 */
static const char pretoken_pattern[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)"       /* 英文缩写，如 's、're。 */
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"  /* 字母序列，可包含前面的一个标点或空格。 */
    "|\\p{N}"                         /* 每次一个数字。 */
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" /* 标点序列及可能跟随的换行。 */
    "|\\s*[\\r\\n]+"                  /* 换行及前面的空白。 */
    "|\\s+(?!\\S)|\\s+";              /* 其他空白。 */

static int encode_normal_text(const Tokenizer *tokenizer, const uint8_t *text, int length,
                               int *output, int capacity, pcre2_code *pattern,
                               pcre2_match_data *match, TokenizerObserver observer, void *user) {
    int cursor = 0, count = 0;
    while (cursor < length) {
        PCRE2_SIZE *span;
        TokenizerEvent event = {0};
        int produced;
        if (pcre2_match(pattern, text, (PCRE2_SIZE)length, (PCRE2_SIZE)cursor,
                         PCRE2_ANCHORED, match, NULL) < 0) return -1;
        span = pcre2_get_ovector_pointer(match);
        if (span[1] <= span[0]) return -1;
        if (observer) {
            event.kind = TOKENIZER_SEGMENT;
            event.bytes = text + span[0]; event.byte_count = span[1] - span[0];
            observer(&event, user);
        }
        produced = bpe_encode_segment(tokenizer, text + span[0], (int)(span[1] - span[0]),
                                       output + count, capacity - count, observer, user);
        if (produced < 0) return -1;
        count += produced;
        cursor = (int)span[1];
    }
    return count;
}

int tokenizer_encode_with_trace(const Tokenizer *tokenizer, const char *text,
                                int *output, int capacity, TokenizerObserver observer, void *user) {
    pcre2_code *pattern;
    pcre2_match_data *match;
    PCRE2_SIZE error_offset;
    int error_code, at = 0, count = 0, length;
    if (!tokenizer || !text || !output || capacity <= 0) return -1;
    length = (int)strlen(text);
    pattern = pcre2_compile((PCRE2_SPTR)pretoken_pattern, PCRE2_ZERO_TERMINATED,
                            PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, NULL);
    if (!pattern) return -1;
    match = pcre2_match_data_create_from_pattern(pattern, NULL);
    if (!match) { pcre2_code_free(pattern); return -1; }

    while (at < length) {
        int end = at, special = -1, produced;
        size_t special_length = 0;
        /* 特殊标记直接使用模型给定的编号，例如 <|im_start|> 表示一条消息开始。 */
        while (end < length) {
            special = vocabulary_special_at(tokenizer, (const uint8_t *)text + end,
                                              (size_t)(length - end), &special_length);
            if (special >= 0) break;
            ++end;
        }
        produced = encode_normal_text(tokenizer, (const uint8_t *)text + at, end - at,
                                       output + count, capacity - count, pattern, match, observer, user);
        if (produced < 0) goto failure;
        count += produced;
        at = end;
        if (special >= 0) {
            TokenizerEvent event = {0};
            if (count >= capacity) goto failure;
            output[count++] = special;
            if (observer) {
                event.kind = TOKENIZER_SPECIAL;
                event.bytes = (const uint8_t *)text + at; event.byte_count = special_length;
                event.merged_id = special;
                observer(&event, user);
            }
            at += (int)special_length;
        }
    }
    if (observer) {
        TokenizerEvent event = {0};
        event.kind = TOKENIZER_RESULT; event.token_ids = output; event.token_count = count;
        observer(&event, user);
    }
    pcre2_match_data_free(match);
    pcre2_code_free(pattern);
    return count;
failure:
    pcre2_match_data_free(match);
    pcre2_code_free(pattern);
    return -1;
}

int tokenizer_encode(const Tokenizer *tokenizer, const char *text, int *output, int capacity) {
    return tokenizer_encode_with_trace(tokenizer, text, output, capacity, NULL, NULL);
}
