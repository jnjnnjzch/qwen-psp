#ifndef QWEN_OPS_H
#define QWEN_OPS_H

#include "qwen_runtime.h"

/* 这些算子只计算数组；模型由哪些层组成，写在 model/ 及 layers/ 中。 */
void q4_initialize_lookup(void);
size_t q4_row_bytes(int columns);
void q4_decode_row(float *output, const uint8_t *row, int columns);
float q4_dot(const uint8_t *row, const float *input, int columns, const float *block_sums);
void q4_input_block_sums(const float *input, int columns, float *block_sums);
void rms_norm(float *output, const float *input, const float *weight, int count, float epsilon);
float exp_negative(float value);
float silu(float value);
void residual_add(float *hidden, const float *branch, int count);
void rope_initialize(QwenRuntime *runtime, const QwenConfig *config);
void rope_set_position(QwenRuntime *runtime, int half_dim, int position);
void rope_apply(const QwenRuntime *runtime, float *values, int heads, int head_dim);
void softmax_inplace(float *scores, int count);
void kv_cache_store(KVCache *cache, int layer, int position, const float *key, const float *value);
void attention_scores(const KVCache *cache, int layer, int kv_head, int position,
                       const float *query, float *scores);
void attention_weighted_sum(const KVCache *cache, int layer, int kv_head, int position,
                             const float *scores, float *output);

#if defined(_PSP_FW_VERSION)
/* PSP 的底层接口一次处理完整的 128 维注意力头；stride 是相邻历史位置的字节距离。 */
void attention_scores_vfpu_128(const float *query, const int8_t *keys, int stride,
                                int count, float *scores);
void attention_accum_vfpu_128(float *output, const int8_t *values, int stride,
                               int count, const float *weights);
#endif

#endif
