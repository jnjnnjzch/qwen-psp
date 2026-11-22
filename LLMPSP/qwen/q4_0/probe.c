#include "falcon_h1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *file;
    FILE *kv_file;
    FalconRuntime *runtime;
    int dim;
} Trace;

static void trace_layer(int position, int layer, int total, void *user) {
    Trace *trace = user;
    (void)position; (void)layer; (void)total;
    if (fwrite(trace->runtime->x, sizeof(float), (size_t)trace->dim, trace->file) !=
        (size_t)trace->dim) exit(1);
    if (layer > 0) {
        if (fwrite(trace->runtime->k, sizeof(float), 1024, trace->kv_file) != 1024 ||
            fwrite(trace->runtime->v, sizeof(float), 1024, trace->kv_file) != 1024)
            exit(1);
    }
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    char error[256];
    int tokens[512], count, i;
    if (argc < 4) {
        fprintf(stderr, "usage: probe MODEL tokenize|chat|logits TEXT [OUTPUT]\n");
        return 2;
    }
    if (!falcon_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (!strcmp(argv[2], "chat"))
        count = falcon_build_chat_prompt(&model, argv[3], tokens, 512);
    else
        count = falcon_tokenize(&model.tokenizer, argv[3], tokens, 512);
    if (count <= 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    for (i = 0; i < count; ++i) printf("%s%d", i ? " " : "", tokens[i]);
    putchar('\n');
    if (!strcmp(argv[2], "logits")) {
        FILE *output;
        Trace trace;
        char trace_path[1024];
        if (argc != 5 || !falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
            fprintf(stderr, "runtime or output argument missing\n");
            return 1;
        }
        if (snprintf(trace_path, sizeof(trace_path), "%s.layers", argv[4]) >=
            (int)sizeof(trace_path)) return 1;
        trace.file = fopen(trace_path, "wb");
        if (!trace.file) return 1;
        if (snprintf(trace_path, sizeof(trace_path), "%s.kv", argv[4]) >=
            (int)sizeof(trace_path)) return 1;
        trace.kv_file = fopen(trace_path, "wb");
        if (!trace.kv_file) return 1;
        trace.runtime = &runtime;
        trace.dim = model.config.dim;
        falcon_runtime_set_progress(&runtime, trace_layer, &trace);
        for (i = 0; i < count; ++i)
            if (!falcon_forward(&model, &runtime, tokens[i], i)) {
                fprintf(stderr, "forward failed\n");
                return 1;
            }
        if (fclose(trace.file) != 0) return 1;
        if (fclose(trace.kv_file) != 0) return 1;
        output = fopen(argv[4], "wb");
        if (!output || fwrite(runtime.logits, sizeof(float),
                               (size_t)model.config.vocab_size, output) !=
                        (size_t)model.config.vocab_size || fclose(output) != 0) {
            fprintf(stderr, "cannot write logits\n");
            return 1;
        }
        falcon_runtime_free(&runtime);
    } else if (strcmp(argv[2], "tokenize") && strcmp(argv[2], "chat")) {
        fprintf(stderr, "unknown mode\n");
        return 2;
    }
    falcon_model_close(&model);
    return 0;
}
