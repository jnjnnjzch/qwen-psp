#include "qwen.h"

#include <stdio.h>
#include <string.h>

typedef struct { int ids[32]; int count; } Output;
static int capture(int token, const uint8_t *bytes, size_t length, void *user) {
    Output *output = user;
    (void)bytes; (void)length;
    output->ids[output->count++] = token;
    return 1;
}

int main(int argc, char **argv) {
    QwenModel model;
    QwenRuntime runtime;
    Sampler sampler;
    Output first = {{0}, 0}, continued = {{0}, 0}, replayed = {{0}, 0};
    char error[256];
    int prompt[128], next[128], full[128];
    int count, next_count = 0, full_count, generated, position, pending;
    if (argc != 2) return 2;
    if (!qwen_model_open(&model, argv[1], error, sizeof(error)) ||
        !qwen_runtime_init(&runtime, &model, error, sizeof(error))) return 1;
    count = tokenizer_build_chat_prompt(&model, "What is 2+2?", prompt, 128);
    if (count <= 0) return 1;
    sampler_init(&sampler, 0, 1, 32, 1, 1);
    if (!qwen_generate_turn(&model, &runtime, &sampler, prompt, count, 0, 8,
                             capture, &first, &generated, &position, &pending)) return 1;
    if (pending >= 0) next[next_count++] = pending;
    next[next_count++] = model.config.im_end_id;
    {
        int made = tokenizer_encode(&model.tokenizer, "\n", next + next_count, 128 - next_count);
        if (made < 0) return 1;
        next_count += made;
        made = tokenizer_build_chat_prompt(&model, "And 3+3?", next + next_count, 128 - next_count);
        if (made < 0) return 1;
        next_count += made;
    }
    if (!qwen_generate_turn(&model, &runtime, &sampler, next, next_count, position, 8,
                             capture, &continued, &generated, NULL, NULL)) return 1;
    /* 将同样的真实对话完整重放，核验继续会话时的 KV 状态。 */
    memcpy(full, prompt, (size_t)count * sizeof(int));
    full_count = count;
    memcpy(full + full_count, first.ids, (size_t)first.count * sizeof(int));
    full_count += first.count;
    {
        int skip_pending = pending >= 0 ? 1 : 0;
        memcpy(full + full_count, next + skip_pending,
                 (size_t)(next_count - skip_pending) * sizeof(int));
        full_count += next_count - skip_pending;
    }
    if (!qwen_generate(&model, &runtime, &sampler, full, full_count, 8,
                        capture, &replayed, &generated)) return 1;
    if (continued.count != replayed.count ||
        memcmp(continued.ids, replayed.ids, (size_t)continued.count * sizeof(int))) return 1;
    printf("multiturn replay: PASS (%d tokens)\n", continued.count);
    qwen_runtime_free(&runtime); qwen_model_close(&model);
    return 0;
}
