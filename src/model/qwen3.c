#include "qwen_model.h"
#include "ops/ops.h"

/* 从这里阅读一次模型计算：Embedding -> 28 个 DecoderLayer -> RMSNorm -> LM head。
 * hidden 是当前 token 的向量；每个 DecoderLayer 都接收并更新这个向量。 */
float *qwen_forward_token(QwenModel *model, QwenRuntime *runtime,
                          int token, int position, int need_logits) {
    const QwenConfig *config = &model->config;
    unsigned int started = runtime->clock_us ? runtime->clock_us() : 0;
    int layer_index;
    if (token < 0 || token >= config->vocab_size || position < 0 || position >= config->context)
        return NULL;
    runtime->io_error = 0;
    rope_set_position(runtime, config->head_dim / 2, position);

    /* 对应 self.embed_tokens(token)：从词嵌入矩阵中取出一个 1024 维向量。 */
    if (!embedding_forward(&model->embed_tokens, model, runtime, token, runtime->hidden))
        return NULL;

    /* 对应 for layer in self.layers: hidden = layer(hidden, ...)。
     * 不同层拥有不同权重；中间结果数组由 runtime 重复使用。 */
    for (layer_index = 0; layer_index < config->num_layers; ++layer_index) {
        if (runtime->progress)
            runtime->progress(position, layer_index, config->num_layers, runtime->progress_user);
        if (!decoder_layer_forward(&model->layers[layer_index], model, runtime, layer_index, position))
            return NULL;
    }
    if (runtime->progress)
        runtime->progress(position, config->num_layers, config->num_layers, runtime->progress_user);

    /* 提示中的中间 token 需要更新 KV。最后一个提示 token 以及生成阶段的 token
     * 还需要 logits：词表中每一个候选 token 的分数。 */
    if (need_logits) {
        rmsnorm_forward(&model->norm, runtime->hidden, runtime->normalized);
        if (!linear_forward(&model->lm_head, model, runtime, runtime->normalized, runtime->logits))
            return NULL;
    }
    if (runtime->clock_us) runtime->forward_us += runtime->clock_us() - started;
    return runtime->logits;
}

float *qwen_forward(QwenModel *model, QwenRuntime *runtime, int token, int position) {
    return qwen_forward_token(model, runtime, token, position, 1);
}
