#include "qwen_model.h"
#include "ops/ops.h"

/* 一个 DecoderLayer 包含两条残差连接。可以直接按这些函数调用阅读结构：
 * hidden += SelfAttention(RMSNorm(hidden))
 * hidden += MLP(RMSNorm(hidden))
 * branch_output 保存当前分支的结果，residual_add 将它逐元素加回 hidden。 */
int decoder_layer_forward(const DecoderLayer *layer, QwenModel *model,
                          QwenRuntime *runtime, int layer_index, int position) {
    rmsnorm_forward(&layer->input_layernorm, runtime->hidden, runtime->normalized);
    if (!self_attention_forward(&layer->self_attn, model, runtime, layer_index, position))
        return 0;
    residual_add(runtime->hidden, runtime->branch_output, model->config.hidden_size);

    rmsnorm_forward(&layer->post_attention_layernorm, runtime->hidden, runtime->normalized);
    if (!mlp_forward(&layer->mlp, model, runtime)) return 0;
    residual_add(runtime->hidden, runtime->branch_output, model->config.hidden_size);
    return 1;
}
