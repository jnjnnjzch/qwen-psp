#ifndef QWEN_LAYERS_H
#define QWEN_LAYERS_H

#include "qwen_types.h"

/* 对应 nn.Linear：输入向量长 in_features，输出向量长 out_features。
 * 权重保存在模型文件的 weight_offset 位置，运行时逐段读取。 */
typedef struct {
    uint32_t weight_offset;
    int in_features, out_features;
} Linear;

/* 对应 nn.Embedding：用 token 编号查找一行向量。 */
typedef struct {
    uint32_t weight_offset;
    int num_embeddings, embedding_dim;
} Embedding;

/* 对应 RMSNorm：权重较小，model_loader.c 会将它们读进内存。 */
typedef struct {
    uint32_t weight_offset;
    const float *weight;
    int features;
    float epsilon;
} RMSNorm;

typedef struct {
    Linear q_proj, k_proj, v_proj, o_proj;
    RMSNorm q_norm, k_norm;
} SelfAttention;

typedef struct {
    Linear gate_proj, up_proj, down_proj;
} MLP;

/* 对应一个 Qwen3DecoderLayer。整个 Qwen3-0.6B 一共有 28 个这样的组件。 */
typedef struct {
    RMSNorm input_layernorm;
    SelfAttention self_attn;
    RMSNorm post_attention_layernorm;
    MLP mlp;
} DecoderLayer;

int embedding_forward(const Embedding *layer, QwenModel *model, QwenRuntime *runtime,
                      int token, float *output);
int linear_forward(const Linear *layer, QwenModel *model, QwenRuntime *runtime,
                   const float *input, float *output);
void rmsnorm_forward(const RMSNorm *layer, const float *input, float *output);
int self_attention_forward(const SelfAttention *layer, QwenModel *model,
                           QwenRuntime *runtime, int layer_index, int position);
int mlp_forward(const MLP *layer, QwenModel *model, QwenRuntime *runtime);
int decoder_layer_forward(const DecoderLayer *layer, QwenModel *model,
                          QwenRuntime *runtime, int layer_index, int position);

#endif
