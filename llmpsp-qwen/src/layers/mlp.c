#include "qwen_model.h"
#include "ops/ops.h"

/* 对应 Qwen3MLP：down_proj(SiLU(gate_proj(x)) * up_proj(x))。
 * 两个输入投影将 1024 维展开到 3072 维，down_proj 将结果变回 1024 维。 */
int mlp_forward(const MLP *layer, QwenModel *model, QwenRuntime *runtime) {
    int feature;
    if (!linear_forward(&layer->gate_proj, model, runtime, runtime->normalized, runtime->mlp_gate) ||
        !linear_forward(&layer->up_proj, model, runtime, runtime->normalized, runtime->mlp_up))
        return 0;
    /* 这里的乘法是同一位置的两个元素相乘。 */
    for (feature = 0; feature < model->config.intermediate_size; ++feature)
        runtime->mlp_gate[feature] = silu(runtime->mlp_gate[feature]) * runtime->mlp_up[feature];
    return linear_forward(&layer->down_proj, model, runtime,
                           runtime->mlp_gate, runtime->branch_output);
}
