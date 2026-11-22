#include "qwen.h"

#include <stdio.h>
#include <stdlib.h>

static int print_token(int token, const uint8_t *bytes, size_t length, void *user) {
    (void)token; (void)user;
    if (bytes && length) fwrite(bytes, 1, length, stdout);
    fflush(stdout);
    return 1;
}

/* 应用程序入口：加载 -> 编码提示 -> 生成 -> 释放。
 * 学习 tokenizer 时可运行 tokenizer_demo；学习网络结构时进入 qwen3.c。 */
int main(int argc, char **argv) {
    QwenModel model;
    QwenRuntime runtime;
    Sampler sampler;
    char error[256];
    int tokens[128], count, generated = 0;
    int max_new = 16, cache_mib = 6;
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s MODEL \"prompt\" [max-new] [cache-MiB]\n", argv[0]);
        return 2;
    }
    if (argc > 3) max_new = atoi(argv[3]);
    if (argc > 4) cache_mib = atoi(argv[4]);
    if (max_new < 1 || max_new > 128 || cache_mib < 0 || cache_mib > 6) return 2;
    if (!qwen_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "model: %s\n", error); return 1;
    }
    if (!qwen_runtime_init(&runtime, &model, error, sizeof(error))) {
        fprintf(stderr, "runtime: %s\n", error); qwen_model_close(&model); return 1;
    }
    qwen_model_cache_weights(&model, (size_t)cache_mib * 1048576u, error, sizeof(error));
    if (error[0]) { fprintf(stderr, "cache: %s\n", error); return 1; }
    count = tokenizer_build_chat_prompt(&model, argv[2], tokens, 128);
    if (count <= 0 || count + max_new > model.config.context) {
        fprintf(stderr, "prompt exceeds context\n"); return 1;
    }
    fprintf(stderr, "prompt tokens: %d; resident: %.2f MiB; cached: %.2f MiB\n", count,
              (double)(qwen_model_resident_bytes(&model) + qwen_runtime_resident_bytes(&runtime)) / 1048576.0,
              (double)model.weight_cache_bytes / 1048576.0);
    sampler_init(&sampler, 0, 1, 32, 1, 1);
    if (!qwen_generate(&model, &runtime, &sampler, tokens, count, max_new,
                        print_token, NULL, &generated)) {
        fprintf(stderr, "generation failed: io_error=%d\n", runtime.io_error); return 1;
    }
    putchar('\n');
    fprintf(stderr, "generated: %d\n", generated);
    qwen_runtime_free(&runtime);
    qwen_model_close(&model);
    return 0;
}
