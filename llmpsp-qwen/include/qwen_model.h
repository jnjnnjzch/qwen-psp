#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include <stdio.h>
#include "qwen_layers.h"
#include "qwen_tokenizer.h"
#include "qwen_runtime.h"

struct QwenModel {
    /* 阅读模型结构时，关注这些成员：它们对应 PyTorch 中的子模块。 */
    QwenConfig config;
    Embedding embed_tokens;
    DecoderLayer layers[QWEN_MAX_LAYERS];
    RMSNorm norm;
    Linear lm_head;
    Tokenizer tokenizer;

    /* 下方由 storage/weights.c 管理：文件句柄、读取缓冲区和权重缓存。
     * 单独学习模型 forward 时，可以通过 Linear/Embedding 接口理解权重访问。 */
    FILE *file;
    uint8_t *io_buffer;
    size_t io_buffer_bytes;
    uint8_t **weight_cache;
    size_t weight_cache_bytes, weight_cache_block_bytes;
    int weight_cache_blocks;
    uint8_t *cache_extra_base;
    size_t cache_extra_bytes;
    uint32_t weights_offset, file_size;
    float *norm_weights;
    size_t norm_weight_bytes;
};

int qwen_model_open(QwenModel *model, const char *path, char *error, size_t size);
void qwen_model_close(QwenModel *model);
size_t qwen_model_cache_weights(QwenModel *model, size_t bytes, char *error, size_t size);
size_t qwen_model_cache_weights_extra(QwenModel *model, size_t bytes,
                                     uint8_t *extra, size_t extra_bytes, int stripe,
                                     char *error, size_t size);
size_t qwen_model_resident_bytes(const QwenModel *model);

/* 一次 forward 处理一个 token。position 是它在会话中的位置，从 0 开始。
 * need_logits 为 0 时更新各层和 KV，省去本次用不到的最终词表投影。 */
float *qwen_forward_token(QwenModel *model, QwenRuntime *runtime,
                          int token, int position, int need_logits);
float *qwen_forward(QwenModel *model, QwenRuntime *runtime, int token, int position);

#endif
