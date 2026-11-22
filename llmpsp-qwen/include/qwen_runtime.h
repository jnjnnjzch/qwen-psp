#ifndef QWEN_RUNTIME_H
#define QWEN_RUNTIME_H

#include "qwen_types.h"

/* 每一层、每个历史位置都保存 K/V，后续 token 会读取这些历史信息。
 * 每个 KV 头使用一个 FP32 scale，将 int8 数值还原到原来的数值范围。 */
typedef struct {
    int8_t *keys, *values;
    float *key_scales, *value_scales;
    int num_layers, context, num_heads, head_dim;
} KVCache;

/* Runtime 保存这一次推理用到的数组。数组在所有层之间重复使用。
 * 模型权重保存在 QwenModel 中；这里的数组保存计算的中间结果。 */
struct QwenRuntime {
    float *hidden;          /* 当前 token 的 1024 维表示。 */
    float *normalized;      /* RMSNorm 的输出。 */
    float *query, *key, *value;
    float *attention_scores, *attention_output;
    float *branch_output;   /* Attention 或 MLP 的输出，随后加回 hidden。 */
    float *mlp_gate, *mlp_up;
    float *logits;          /* 151936 个候选 token 的分数。 */
    KVCache kv_cache;
    uint8_t *weight_buffer;
    size_t weight_buffer_bytes, bytes_allocated;
    float rope_freq[QWEN_ROPE_HALF], rope_cos[QWEN_ROPE_HALF], rope_sin[QWEN_ROPE_HALF];
    int rope_position, io_error;
    QwenProgressCallback progress;
    void *progress_user;
    QwenClockCallback clock_us;
    unsigned int forward_us, matvec_us;
    uint64_t streamed_bytes;
};

int qwen_runtime_init(QwenRuntime *runtime, const QwenModel *model, char *error, size_t size);
void qwen_runtime_reset(QwenRuntime *runtime, const QwenModel *model);
void qwen_runtime_free(QwenRuntime *runtime);
void qwen_runtime_set_progress(QwenRuntime *runtime, QwenProgressCallback callback, void *user);
size_t qwen_runtime_resident_bytes(const QwenRuntime *runtime);

#endif
