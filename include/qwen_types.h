#ifndef QWEN_TYPES_H
#define QWEN_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define QWEN_MAX_LAYERS 28
#define QWEN_ROPE_HALF 64
#define QWEN_Q4_BLOCK_SIZE 32
#define QWEN_TOP_K_MAX 64
#define QWEN_ASSISTANT_PREFIX "assistant\n<think>\n\n</think>\n\n"

/* 配置描述数组的形状。hidden_size 是每个 token 的向量长度；
 * intermediate_size 是 MLP 内部展开后的长度。 */
typedef struct {
    int hidden_size, intermediate_size, num_layers;
    int num_attention_heads, num_key_value_heads, head_dim;
    int vocab_size, model_context, context;
    int pad_id, eos_id, bos_id, im_start_id, im_end_id, eom_id, eot_id;
    float rope_theta, rms_norm_eps;
} QwenConfig;

/* 这里提前声明类型名，让不同组件可以传递指针；完整成员分别写在对应头文件中。 */
typedef struct QwenModel QwenModel;
typedef struct QwenRuntime QwenRuntime;
typedef void (*QwenProgressCallback)(int position, int layer, int total_layers, void *user);
typedef unsigned int (*QwenClockCallback)(void);
typedef int (*QwenTokenCallback)(int token, const uint8_t *bytes, size_t length, void *user);

#endif
