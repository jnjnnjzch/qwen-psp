#include "qwen_model.h"
#include "ops/ops.h"

/* 对应 Qwen3Attention。先准备 Q/K/V，再计算当前 token 与历史 token 的关联。
 * 16 个 Q 头，每个 128 维；8 个 KV 头，每个 128 维。
 * 每两个 Q 头共享一个 KV 头，这就是本模型使用的 grouped-query attention。 */
int self_attention_forward(const SelfAttention *layer, QwenModel *model,
                           QwenRuntime *runtime, int layer_index, int position) {
    const QwenConfig *config = &model->config;
    int head;
    int queries_per_kv_head = config->num_attention_heads / config->num_key_value_heads;
    if (!linear_forward(&layer->q_proj, model, runtime, runtime->normalized, runtime->query) ||
        !linear_forward(&layer->k_proj, model, runtime, runtime->normalized, runtime->key) ||
        !linear_forward(&layer->v_proj, model, runtime, runtime->normalized, runtime->value))
        return 0;

    /* Qwen3 在每个 Q/K 头上做 RMSNorm，然后用 RoPE 加入位置信息。 */
    for (head = 0; head < config->num_attention_heads; ++head)
        rmsnorm_forward(&layer->q_norm, runtime->query + head * config->head_dim,
                         runtime->query + head * config->head_dim);
    for (head = 0; head < config->num_key_value_heads; ++head)
        rmsnorm_forward(&layer->k_norm, runtime->key + head * config->head_dim,
                         runtime->key + head * config->head_dim);
    rope_apply(runtime, runtime->query, config->num_attention_heads, config->head_dim);
    rope_apply(runtime, runtime->key, config->num_key_value_heads, config->head_dim);

    /* 将当前 token 的 K/V 加入这一层的历史缓存。 */
    kv_cache_store(&runtime->kv_cache, layer_index, position, runtime->key, runtime->value);
    for (head = 0; head < config->num_attention_heads; ++head) {
        int kv_head = head / queries_per_kv_head;
        attention_scores(&runtime->kv_cache, layer_index, kv_head, position,
                           runtime->query + head * config->head_dim, runtime->attention_scores);
        softmax_inplace(runtime->attention_scores, position + 1);
        attention_weighted_sum(&runtime->kv_cache, layer_index, kv_head, position,
                                runtime->attention_scores,
                                runtime->attention_output + head * config->head_dim);
    }
    /* 16 个头拼接后有 2048 个元素，o_proj 将它投影回 1024 维。 */
    return linear_forward(&layer->o_proj, model, runtime,
                           runtime->attention_output, runtime->branch_output);
}
