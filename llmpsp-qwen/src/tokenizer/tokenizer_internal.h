#ifndef QWEN_TOKENIZER_INTERNAL_H
#define QWEN_TOKENIZER_INTERNAL_H

#include "qwen_tokenizer.h"

int vocabulary_find(const Tokenizer *tokenizer, const uint8_t *bytes, size_t length);
int vocabulary_byte_token(const Tokenizer *tokenizer, uint8_t byte);
uint32_t vocabulary_merge_rank(const Tokenizer *tokenizer, int token);
int vocabulary_special_at(const Tokenizer *tokenizer, const uint8_t *text,
                           size_t remaining, size_t *length);
int bpe_encode_segment(const Tokenizer *tokenizer, const uint8_t *text, int length,
                        int *output, int capacity, TokenizerObserver observer, void *user);

#endif
