#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

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

int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text,
                    int *tokens, int max_tokens) {
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
