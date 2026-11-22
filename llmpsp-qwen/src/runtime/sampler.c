#include "qwen_generation.h"

#include <math.h>

/* 沿用原项目的采样算法。temperature=0 时直接选择分数最大的 token。 */
typedef struct { float logit; int id; } Candidate;

static uint32_t random_u32(Sampler *sampler) {
    uint64_t value = sampler->rng;
    value ^= value >> 12; value ^= value << 25; value ^= value >> 27;
    sampler->rng = value;
    return (uint32_t)((value * 0x2545f4914f6cdd1dULL) >> 32);
}

static float random_f32(Sampler *sampler) {
    return (float)(random_u32(sampler) >> 8) / 16777216.0f;
}

void sampler_init(Sampler *sampler, float temperature, float top_p, int top_k,
                  float repetition_penalty, uint64_t seed) {
    sampler->temperature = temperature < 0.0f ? 0.0f : temperature;
    sampler->top_p = top_p <= 0.0f || top_p > 1.0f ? 1.0f : top_p;
    sampler->top_k = top_k < 1 ? 1 : (top_k > QWEN_TOP_K_MAX ? QWEN_TOP_K_MAX : top_k);
    sampler->repetition_penalty = repetition_penalty < 1.0f ? 1.0f : repetition_penalty;
    sampler->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static int in_recent_history(int token, const int *history, int count) {
    int begin = count > 64 ? count - 64 : 0, index;
    for (index = begin; index < count; ++index) if (history[index] == token) return 1;
    return 0;
}

static float penalized(const Sampler *sampler, float value, int token,
                        const int *history, int count) {
    if (in_recent_history(token, history, count))
        return value < 0.0f ? value * sampler->repetition_penalty : value / sampler->repetition_penalty;
    return value;
}

int sampler_sample(Sampler *sampler, const float *logits, int vocab_size,
                    const int *history, int history_count) {
    Candidate top[QWEN_TOP_K_MAX];
    int count = 0, token;
    if (sampler->temperature <= 0.0f) {
        int best = 0;
        float best_value = penalized(sampler, logits[0], 0, history, history_count);
        for (token = 1; token < vocab_size; ++token) {
            float value = penalized(sampler, logits[token], token, history, history_count);
            if (value > best_value) { best_value = value; best = token; }
        }
        return best;
    }
    for (token = 0; token < vocab_size; ++token) {
        float value = penalized(sampler, logits[token], token, history, history_count) / sampler->temperature;
        int position;
        if (count == sampler->top_k && value <= top[count - 1].logit) continue;
        position = count < sampler->top_k ? count++ : count - 1;
        while (position > 0 && top[position - 1].logit < value) {
            top[position] = top[position - 1]; --position;
        }
        top[position].logit = value; top[position].id = token;
    }
    {
        float probabilities[QWEN_TOP_K_MAX];
        float maximum = top[0].logit, total = 0.0f, cumulative = 0.0f;
        float coin;
        int index, keep = count;
        for (index = 0; index < count; ++index) {
            probabilities[index] = expf(top[index].logit - maximum);
            total += probabilities[index];
        }
        if (sampler->top_p < 1.0f) {
            for (index = 0; index < count; ++index) {
                cumulative += probabilities[index] / total;
                if (cumulative >= sampler->top_p) { keep = index + 1; break; }
            }
        }
        total = 0.0f;
        for (index = 0; index < keep; ++index) total += probabilities[index];
        coin = random_f32(sampler) * total;
        cumulative = 0.0f;
        for (index = 0; index < keep; ++index) {
            cumulative += probabilities[index];
            if (coin < cumulative) return top[index].id;
        }
        return top[keep - 1].id;
    }
}
