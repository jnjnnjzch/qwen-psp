#ifndef QWEN_LEGACY_FRONTEND_H
#define QWEN_LEGACY_FRONTEND_H

#include "qwen.h"

/* 原 PSP 界面使用这些类型名和函数名。这里给出接口声明；
 * falcon_frontend.c 把调用转交给 Qwen。模型实现使用 include/qwen_*.h 中的名称。 */
typedef QwenModel FalconModel;
typedef QwenRuntime FalconRuntime;
typedef Sampler FalconSampler;
typedef Tokenizer FalconTokenizer;
typedef QwenProgressCallback FalconProgressCallback;
typedef QwenTokenCallback FalconTokenCallback;
#define FALCON_MODEL_NAME "Qwen3 0.6B Q4_0"
#define FALCON_ASSISTANT_PREFIX QWEN_ASSISTANT_PREFIX

int falcon_model_open(FalconModel *model, const char *path, char *error, size_t size);
void falcon_model_close(FalconModel *model);
size_t falcon_model_cache_weights_extra(FalconModel *model, size_t bytes,
                                         uint8_t *extra, size_t extra_bytes, int stripe,
                                         char *error, size_t size);
size_t falcon_model_resident_bytes(const FalconModel *model);
int falcon_runtime_init(FalconRuntime *runtime, const FalconModel *model, char *error, size_t size);
void falcon_runtime_free(FalconRuntime *runtime);
void falcon_runtime_set_progress(FalconRuntime *runtime, FalconProgressCallback callback, void *user);
size_t falcon_runtime_resident_bytes(const FalconRuntime *runtime);
int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text, int *output, int capacity);
int falcon_build_chat_prompt(const FalconModel *model, const char *text, int *output, int capacity);
void falcon_sampler_init(FalconSampler *sampler, float temperature, float top_p, int top_k,
                          float repetition_penalty, uint64_t seed);
int falcon_generate_turn(FalconModel *model, FalconRuntime *runtime, FalconSampler *sampler,
                          const int *prompt, int count, int position, int max_new,
                          FalconTokenCallback callback, void *user,
                          int *generated, int *next_position, int *pending);

int falcon_prefetch_open(const char *path);
void falcon_prefetch_configure(const FalconModel *model);
void falcon_prefetch_stats(unsigned int *served_kib, unsigned int *wait_ms, int *alive);
void falcon_prefetch_stats_reset(void);
void falcon_prefetch_close(void);

#endif
