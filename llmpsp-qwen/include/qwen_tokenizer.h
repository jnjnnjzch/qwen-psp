#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include "qwen_types.h"

/* 词表中的一个编号对应一段原始字节；有些条目恰好是完整汉字或单词，
 * 有些条目只有一个字节。offsets 保存每个条目在 pieces 中的起止位置。 */
typedef struct {
    uint8_t *section;
    size_t section_bytes;
    int vocab_size, max_piece_bytes;
    const uint8_t *offsets, *pieces, *sorted_ids, *ranks, *byte_ids;
    int special_count;
    const uint8_t *special_ids;
} Tokenizer;

/* 学习演示使用同一套真实分词过程。这些事件只观察数据，不改变合并选择。
 * observer 在编码过程中同步执行；事件中的指针只在本次回调期间有效。 */
typedef enum {
    TOKENIZER_SEGMENT, TOKENIZER_INITIAL_BYTES, TOKENIZER_MERGE,
    TOKENIZER_SPECIAL, TOKENIZER_RESULT
} TokenizerEventKind;

typedef struct {
    TokenizerEventKind kind;
    const uint8_t *bytes;
    size_t byte_count;
    const int *token_ids;
    int token_count;
    int left_id, right_id, merged_id;
    uint32_t merge_rank;
} TokenizerEvent;
typedef void (*TokenizerObserver)(const TokenizerEvent *event, void *user);

/* encode：UTF-8 文本 -> token 编号数组。成功返回编号数量，失败返回 -1。
 * capacity 是 output 数组可以容纳的编号数量。 */
int tokenizer_encode(const Tokenizer *tokenizer, const char *text, int *output, int capacity);
int tokenizer_encode_with_trace(const Tokenizer *tokenizer, const char *text,
                                int *output, int capacity,
                                TokenizerObserver observer, void *user);

/* decode 的基本操作：返回一个 token 对应的原始字节及长度。
 * 多个 token 的字节依次拼接，就得到模型生成的文本。 */
const uint8_t *tokenizer_token_bytes(const Tokenizer *tokenizer, int token, size_t *length);
int tokenizer_build_chat_prompt(const QwenModel *model, const char *text,
                                int *output, int capacity);

#endif
