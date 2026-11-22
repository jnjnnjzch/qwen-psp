#include "tokenizer_internal.h"
#include "storage/binary.h"

#include <string.h>

const uint8_t *tokenizer_token_bytes(const Tokenizer *tokenizer, int token, size_t *length) {
    uint32_t begin, end;
    if (length) *length = 0;
    if (!tokenizer || token < 0 || token >= tokenizer->vocab_size) return NULL;
    begin = read_u32_le(tokenizer->offsets + token * 4);
    end = read_u32_le(tokenizer->offsets + (token + 1) * 4);
    if (length) *length = end - begin;
    return tokenizer->pieces + begin;
}

static int compare_bytes(const uint8_t *left, size_t left_length,
                         const uint8_t *right, size_t right_length) {
    size_t common = left_length < right_length ? left_length : right_length;
    int comparison = memcmp(left, right, common);
    if (comparison) return comparison;
    if (left_length < right_length) return -1;
    if (left_length > right_length) return 1;
    return 0;
}

/* 按字节内容查找词表编号。sorted_ids 保存排好顺序的编号，
 * 二分查找每次排除一半候选，直到找到内容完全相同的条目。 */
int vocabulary_find(const Tokenizer *tokenizer, const uint8_t *bytes, size_t length) {
    int low = 0, high = tokenizer->vocab_size - 1;
    while (low <= high) {
        int middle = low + (high - low) / 2;
        int token = (int)read_u32_le(tokenizer->sorted_ids + middle * 4);
        size_t candidate_length;
        const uint8_t *candidate = tokenizer_token_bytes(tokenizer, token, &candidate_length);
        int comparison = compare_bytes(bytes, length, candidate, candidate_length);
        if (comparison == 0) return token;
        if (comparison < 0) high = middle - 1;
        else low = middle + 1;
    }
    return -1;
}

int vocabulary_byte_token(const Tokenizer *tokenizer, uint8_t byte) {
    return (int)read_u32_le(tokenizer->byte_ids + byte * 4);
}

/* rank 数值越小，这个合并操作的优先级越高。0xffffffff 表示没有该合并。 */
uint32_t vocabulary_merge_rank(const Tokenizer *tokenizer, int token) {
    return read_u32_le(tokenizer->ranks + token * 4);
}

int vocabulary_special_at(const Tokenizer *tokenizer, const uint8_t *text,
                           size_t remaining, size_t *length) {
    int index;
    for (index = 0; index < tokenizer->special_count; ++index) {
        int token = (int)read_u32_le(tokenizer->special_ids + index * 4);
        size_t token_length;
        const uint8_t *bytes = tokenizer_token_bytes(tokenizer, token, &token_length);
        if (token_length <= remaining && memcmp(text, bytes, token_length) == 0) {
            *length = token_length;
            return token;
        }
    }
    return -1;
}
