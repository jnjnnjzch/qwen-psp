#include "ops.h"

#include <math.h>
#include <string.h>

/* RMSNorm：求向量平方的平均值，再用它缩放每个元素，最后乘以学习得到的 weight。
 * epsilon 是一个很小的正数，参与分母计算。运算顺序沿用原项目。 */
void rms_norm(float *output, const float *input, const float *weight, int count, float epsilon) {
    float mean = 0.0f;
    int index;
    for (index = 0; index < count; ++index) mean += input[index] * input[index];
    mean = 1.0f / sqrtf(mean / (float)count + epsilon);
    for (index = 0; index < count; ++index) output[index] = input[index] * mean * weight[index];
}

/* 沿用原项目在 PSP 上使用的指数近似。调用位置保证 value <= 0。 */
float exp_negative(float value) {
    float t, fraction, polynomial, scale;
    int exponent;
    uint32_t bits;
    if (value < -60.0f) return 0.0f;
    t = value * 1.4426950408889634f;
    exponent = (int)(t - 0.5f);
    fraction = t - (float)exponent;
    polynomial = 1.0f + fraction * (0.69314718056f + fraction * (0.24022650696f +
        fraction * (0.05550410866f + fraction * (0.00961812911f + fraction * 0.00133335581f))));
    bits = (uint32_t)(127 + exponent) << 23;
    memcpy(&scale, &bits, sizeof(scale));
    return polynomial * scale;
}

float silu(float value) {
    if (value >= 0.0f) return value / (1.0f + exp_negative(-value));
    {
        float exponential = exp_negative(value);
        return value * exponential / (1.0f + exponential);
    }
}

void residual_add(float *hidden, const float *branch, int count) {
    int index;
    for (index = 0; index < count; ++index) hidden[index] += branch[index];
}

/* Softmax 将历史位置的分数变为权重，所有权重之和约等于 1。 */
void softmax_inplace(float *scores, int count) {
    float maximum = -1.0e30f, sum = 0.0f;
    int index;
    for (index = 0; index < count; ++index)
        if (scores[index] > maximum) maximum = scores[index];
    for (index = 0; index < count; ++index) {
        scores[index] = exp_negative(scores[index] - maximum);
        sum += scores[index];
    }
    for (index = 0; index < count; ++index) scores[index] = scores[index] / sum;
}

void rope_initialize(QwenRuntime *runtime, const QwenConfig *config) {
    int index;
    for (index = 0; index < config->head_dim / 2; ++index)
        runtime->rope_freq[index] = powf(config->rope_theta,
                                         -(2.0f * (float)index) / (float)config->head_dim);
    runtime->rope_position = -1;
}

void rope_set_position(QwenRuntime *runtime, int half_dim, int position) {
    int index;
    if (runtime->rope_position == position) return;
    for (index = 0; index < half_dim; ++index) {
        float angle = (float)position * runtime->rope_freq[index];
        runtime->rope_cos[index] = cosf(angle);
        runtime->rope_sin[index] = sinf(angle);
    }
    runtime->rope_position = position;
}

/* RoPE 在每个头内部成对旋转元素；位置越靠后，旋转角度相应改变。 */
void rope_apply(const QwenRuntime *runtime, float *values, int heads, int head_dim) {
    int head, index, half = head_dim / 2;
    for (head = 0; head < heads; ++head) {
        float *vector = values + head * head_dim;
        for (index = 0; index < half; ++index) {
            float cosine = runtime->rope_cos[index], sine = runtime->rope_sin[index];
            float first = vector[index], second = vector[index + half];
            vector[index] = first * cosine - second * sine;
            vector[index + half] = second * cosine + first * sine;
        }
    }
}
