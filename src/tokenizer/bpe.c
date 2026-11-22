#include "tokenizer_internal.h"

#include <stdlib.h>
#include <string.h>

/* BPE = Byte Pair Encoding，这里从 UTF-8 字节对应的词表条目开始合并。
 * text 是预分词得到的一段连续文本，例如 "Hello" 或 " world"。
 * 这个函数沿用原项目 bpe_chunk 的合并算法，名称明确表示它处理一个文本片段。 */
int bpe_encode_segment(const Tokenizer *tokenizer, const uint8_t *text, int length,
                        int *output, int capacity, TokenizerObserver observer, void *user) {
    int *token_ids;
    uint8_t merged_bytes[512];
    int count = length, index;
    TokenizerEvent event = {0};
    if (length <= 0) return 0;
    token_ids = malloc((size_t)length * sizeof(*token_ids));
    if (!token_ids) return -1;

    /* 起点：每个字节对应一个基础 token。一个汉字通常占多个 UTF-8 字节。 */
    for (index = 0; index < length; ++index)
        token_ids[index] = vocabulary_byte_token(tokenizer, text[index]);
    if (observer) {
        event.kind = TOKENIZER_INITIAL_BYTES;
        event.bytes = text; event.byte_count = (size_t)length;
        event.token_ids = token_ids; event.token_count = count;
        observer(&event, user);
    }

    while (count > 1) {
        uint32_t best_rank = 0xffffffffu;
        int best_pair = -1, best_token = -1;
        /* 检查每一对相邻条目，寻找本轮最优先的可用合并。
         * 合并规则来自模型的词表；本程序使用已经训练好的规则。 */
        for (index = 0; index < count - 1; ++index) {
            size_t left_length, right_length;
            const uint8_t *left = tokenizer_token_bytes(tokenizer, token_ids[index], &left_length);
            const uint8_t *right = tokenizer_token_bytes(tokenizer, token_ids[index + 1], &right_length);
            int merged_token;
            uint32_t rank;
            if (left_length + right_length > (size_t)tokenizer->max_piece_bytes ||
                left_length + right_length > sizeof(merged_bytes)) continue;
            memcpy(merged_bytes, left, left_length);
            memcpy(merged_bytes + left_length, right, right_length);
            merged_token = vocabulary_find(tokenizer, merged_bytes, left_length + right_length);
            if (merged_token < 0) continue;
            rank = vocabulary_merge_rank(tokenizer, merged_token);
            if (rank < best_rank) {
                best_rank = rank;
                best_pair = index;
                best_token = merged_token;
            }
        }
        /* 相邻条目全部无法合并时，这个片段的编码完成。 */
        if (best_pair < 0) break;
        event.left_id = token_ids[best_pair];
        event.right_id = token_ids[best_pair + 1];
        token_ids[best_pair] = best_token;
        /* 两个条目变成一个条目，将后面的编号向前移动一个位置。 */
        memmove(token_ids + best_pair + 1, token_ids + best_pair + 2,
                  (size_t)(count - best_pair - 2) * sizeof(*token_ids));
        --count;
        if (observer) {
            event.kind = TOKENIZER_MERGE;
            event.merged_id = best_token; event.merge_rank = best_rank;
            event.token_ids = token_ids; event.token_count = count;
            observer(&event, user);
        }
    }
    if (count > capacity) { free(token_ids); return -1; }
    memcpy(output, token_ids, (size_t)count * sizeof(*token_ids));
    free(token_ids);
    return count;
}
