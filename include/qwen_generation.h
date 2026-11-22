#ifndef QWEN_GENERATION_H
#define QWEN_GENERATION_H

#include "qwen_types.h"

typedef struct {
    float temperature, top_p, repetition_penalty;
    int top_k;
    uint64_t rng;
} Sampler;

void sampler_init(Sampler *sampler, float temperature, float top_p, int top_k,
                  float repetition_penalty, uint64_t seed);
int sampler_sample(Sampler *sampler, const float *logits, int vocab_size,
                    const int *history, int history_count);

/* generate 包含多次 forward：读入提示，再不断选择并处理新 token。
 * callback 每次收到一个真实生成的 token，返回 0 可以停止生成。 */
int qwen_generate(QwenModel *model, QwenRuntime *runtime, Sampler *sampler,
                   const int *prompt, int prompt_count, int max_new_tokens,
                   QwenTokenCallback callback, void *user, int *generated_count);
int qwen_generate_turn(QwenModel *model, QwenRuntime *runtime, Sampler *sampler,
                        const int *prompt, int prompt_count, int start_position,
                        int max_new_tokens, QwenTokenCallback callback, void *user,
                        int *generated_count, int *next_position, int *pending_token);

#endif
