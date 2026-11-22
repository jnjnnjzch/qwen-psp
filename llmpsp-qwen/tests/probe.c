#include "qwen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { FILE *file; QwenRuntime *runtime; int features; } Trace;

static void trace_layer(int position, int layer, int total, void *user) {
    Trace *trace = user;
    (void)position; (void)layer; (void)total;
    if (fwrite(trace->runtime->hidden, sizeof(float), (size_t)trace->features, trace->file) !=
        (size_t)trace->features) exit(1);
}

int main(int argc, char **argv) {
    QwenModel model;
    QwenRuntime runtime;
    char error[256];
    int tokens[512], count, index;
    if (argc < 4) return 2;
    if (!qwen_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    if (!strcmp(argv[2], "chat")) count = tokenizer_build_chat_prompt(&model, argv[3], tokens, 512);
    else count = tokenizer_encode(&model.tokenizer, argv[3], tokens, 512);
    if (count <= 0) return 1;
    for (index = 0; index < count; ++index) printf("%s%d", index ? " " : "", tokens[index]);
    putchar('\n');
    if (!strcmp(argv[2], "logits")) {
        FILE *output;
        Trace trace;
        char trace_path[1024];
        if (argc != 5 || !qwen_runtime_init(&runtime, &model, error, sizeof(error))) return 1;
        if (snprintf(trace_path, sizeof(trace_path), "%s.layers", argv[4]) >= (int)sizeof(trace_path)) return 1;
        trace.file = fopen(trace_path, "wb");
        if (!trace.file) return 1;
        trace.runtime = &runtime; trace.features = model.config.hidden_size;
        qwen_runtime_set_progress(&runtime, trace_layer, &trace);
        for (index = 0; index < count; ++index)
            if (!qwen_forward(&model, &runtime, tokens[index], index)) return 1;
        if (fclose(trace.file) != 0) return 1;
        output = fopen(argv[4], "wb");
        if (!output || fwrite(runtime.logits, sizeof(float), (size_t)model.config.vocab_size, output) !=
            (size_t)model.config.vocab_size || fclose(output) != 0) return 1;
        qwen_runtime_free(&runtime);
    } else if (strcmp(argv[2], "tokenize") && strcmp(argv[2], "chat")) return 2;
    qwen_model_close(&model);
    return 0;
}
