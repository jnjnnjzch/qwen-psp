#include "qwen.h"

#include <stdio.h>

/* 单字节条目可能只包含汉字的部分 UTF-8 字节，用十六进制显示这些字节。 */
static void print_bytes(const uint8_t *bytes, size_t length) {
    size_t index;
    putchar('"');
    for (index = 0; index < length; ++index) {
        unsigned byte = bytes[index];
        if (byte == '\n') fputs("\\n", stdout);
        else if (byte == '\r') fputs("\\r", stdout);
        else if (byte == '\t') fputs("\\t", stdout);
        else if (byte == '"' || byte == '\\') printf("\\%c", (int)byte);
        else if (byte >= 32 && byte < 127) putchar((int)byte);
        else printf("\\x%02X", byte);
    }
    putchar('"');
}

static void print_ids(const Tokenizer *tokenizer, const int *tokens, int count) {
    int index;
    putchar('[');
    for (index = 0; index < count; ++index) {
        size_t length;
        const uint8_t *bytes = tokenizer_token_bytes(tokenizer, tokens[index], &length);
        if (index) fputs(", ", stdout);
        printf("%d:", tokens[index]); print_bytes(bytes, length);
    }
    puts("]");
}

static void observe(const TokenizerEvent *event, void *user) {
    const Tokenizer *tokenizer = user;
    if (event->kind == TOKENIZER_SEGMENT) {
        fputs("\n预分词片段：", stdout);
        print_bytes(event->bytes, event->byte_count);
        printf("，%u 个字节\n", (unsigned)event->byte_count);
    } else if (event->kind == TOKENIZER_INITIAL_BYTES) {
        fputs("初始字节对应的条目：", stdout);
        print_ids(tokenizer, event->token_ids, event->token_count);
    } else if (event->kind == TOKENIZER_MERGE) {
        printf("合并 %d + %d -> %d，词表合并优先级 %u\n", event->left_id,
                 event->right_id, event->merged_id, (unsigned)event->merge_rank);
        fputs("本轮结果：", stdout); print_ids(tokenizer, event->token_ids, event->token_count);
    } else if (event->kind == TOKENIZER_SPECIAL) {
        fputs("特殊标记：", stdout); print_bytes(event->bytes, event->byte_count);
        printf(" -> %d\n", event->merged_id);
    } else if (event->kind == TOKENIZER_RESULT) {
        fputs("\n全部输出：", stdout); print_ids(tokenizer, event->token_ids, event->token_count);
    }
}

int main(int argc, char **argv) {
    QwenModel model;
    char error[256];
    int tokens[512], count, index;
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL \"text\"\n", argv[0]); return 2; }
    if (!qwen_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    printf("输入文本：%s\n", argv[2]);
    count = tokenizer_encode_with_trace(&model.tokenizer, argv[2], tokens, 512,
                                         observe, &model.tokenizer);
    if (count < 0) { fprintf(stderr, "tokenization failed\n"); return 1; }
    fputs("重新拼接 token 的字节：", stdout);
    for (index = 0; index < count; ++index) {
        size_t length;
        const uint8_t *bytes = tokenizer_token_bytes(&model.tokenizer, tokens[index], &length);
        fwrite(bytes, 1, length, stdout);
    }
    putchar('\n');
    qwen_model_close(&model);
    return 0;
}
