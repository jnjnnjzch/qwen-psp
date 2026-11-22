#include "ops.h"

#include <math.h>
#include <string.h>

static void quantize_vector(int8_t *output, float *scale_out, const float *input, int count) {
    float maximum = 0.0f, scale;
    int index;
    for (index = 0; index < count; ++index) {
        float absolute = fabsf(input[index]);
        if (absolute > maximum) maximum = absolute;
    }
    scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    *scale_out = scale;
    for (index = 0; index < count; ++index) {
        int value = (int)lrintf(input[index] / scale);
        if (value < -127) value = -127;
        if (value > 127) value = 127;
        output[index] = (int8_t)value;
    }
}

void kv_cache_store(KVCache *cache, int layer, int position, const float *key, const float *value) {
    int head, kv_dim = cache->num_heads * cache->head_dim;
    size_t vector_offset = ((size_t)layer * cache->context + position) * kv_dim;
    size_t scale_offset = ((size_t)layer * cache->context + position) * cache->num_heads;
    for (head = 0; head < cache->num_heads; ++head) {
        quantize_vector(cache->keys + vector_offset + head * cache->head_dim,
                         cache->key_scales + scale_offset + head,
                         key + head * cache->head_dim, cache->head_dim);
        quantize_vector(cache->values + vector_offset + head * cache->head_dim,
                         cache->value_scales + scale_offset + head,
                         value + head * cache->head_dim, cache->head_dim);
    }
}

/* 当前 Q 与每一个历史 K 做点积，得到 position+1 个分数。 */
void attention_scores(const KVCache *cache, int layer, int kv_head, int position,
                       const float *query, float *scores) {
    int time, kv_dim = cache->num_heads * cache->head_dim;
    size_t layer_offset = (size_t)layer * cache->context * kv_dim;
    size_t scale_offset = (size_t)layer * cache->context * cache->num_heads;
    float inverse_sqrt = 1.0f / sqrtf((float)cache->head_dim);
#if defined(_PSP_FW_VERSION)
    /* 模型加载器限定 head_dim 为 128；一次 VFPU 调用得到完整的原始点积。 */
    attention_scores_vfpu_128(query, cache->keys + layer_offset + kv_head * cache->head_dim,
                              kv_dim, position + 1, scores);
    for (time = 0; time <= position; ++time) {
        float scale = cache->key_scales[scale_offset + (size_t)time * cache->num_heads + kv_head];
        scores[time] = scores[time] * scale * inverse_sqrt;
    }
#else
    int index;
    for (time = 0; time <= position; ++time) {
        const int8_t *key = cache->keys + layer_offset + (size_t)time * kv_dim + kv_head * cache->head_dim;
        float scale = cache->key_scales[scale_offset + (size_t)time * cache->num_heads + kv_head];
        float even = 0.0f, odd = 0.0f;
        for (index = 0; index < cache->head_dim; index += 2) {
            even += query[index] * (float)key[index];
            odd += query[index + 1] * (float)key[index + 1];
        }
        scores[time] = (even + odd) * scale * inverse_sqrt;
    }
#endif
}

/* 对历史 V 做加权求和。Softmax 权重大，那个位置对本次输出的影响相应更大。 */
void attention_weighted_sum(const KVCache *cache, int layer, int kv_head, int position,
                             const float *scores, float *output) {
    int time, kv_dim = cache->num_heads * cache->head_dim;
    size_t layer_offset = (size_t)layer * cache->context * kv_dim;
    size_t scale_offset = (size_t)layer * cache->context * cache->num_heads;
#if defined(_PSP_FW_VERSION)
    /* 每个历史位置使用一个 scale。准备好这些系数后，一次计算 128 个输出元素。 */
    float weights[position + 1];
    for (time = 0; time <= position; ++time)
        weights[time] = scores[time] * cache->value_scales[
            scale_offset + (size_t)time * cache->num_heads + kv_head];
    attention_accum_vfpu_128(output, cache->values + layer_offset + kv_head * cache->head_dim,
                             kv_dim, position + 1, weights);
#else
    int index;
    memset(output, 0, (size_t)cache->head_dim * sizeof(float));
    for (time = 0; time <= position; ++time) {
        const int8_t *value = cache->values + layer_offset + (size_t)time * kv_dim + kv_head * cache->head_dim;
        float weight = scores[time] * cache->value_scales[scale_offset + (size_t)time * cache->num_heads + kv_head];
        for (index = 0; index < cache->head_dim; ++index) output[index] += weight * (float)value[index];
    }
#endif
}
