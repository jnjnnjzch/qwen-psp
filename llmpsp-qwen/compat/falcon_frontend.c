#include "falcon_h1.h"

/* 这些函数只转换界面使用的调用名称，参数直接传给对应的 Qwen 接口。 */
int falcon_model_open(FalconModel *model, const char *path, char *error, size_t size) {
    return qwen_model_open(model, path, error, size);
}
void falcon_model_close(FalconModel *model) { qwen_model_close(model); }
size_t falcon_model_cache_weights_extra(FalconModel *model, size_t bytes,
                                         uint8_t *extra, size_t extra_bytes, int stripe,
                                         char *error, size_t size) {
    return qwen_model_cache_weights_extra(model, bytes, extra, extra_bytes, stripe, error, size);
}
size_t falcon_model_resident_bytes(const FalconModel *model) { return qwen_model_resident_bytes(model); }
int falcon_runtime_init(FalconRuntime *runtime, const FalconModel *model, char *error, size_t size) {
    return qwen_runtime_init(runtime, model, error, size);
}
void falcon_runtime_free(FalconRuntime *runtime) { qwen_runtime_free(runtime); }
void falcon_runtime_set_progress(FalconRuntime *runtime, FalconProgressCallback callback, void *user) {
    qwen_runtime_set_progress(runtime, callback, user);
}
size_t falcon_runtime_resident_bytes(const FalconRuntime *runtime) { return qwen_runtime_resident_bytes(runtime); }
int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text, int *output, int capacity) {
    return tokenizer_encode(tokenizer, text, output, capacity);
}
int falcon_build_chat_prompt(const FalconModel *model, const char *text, int *output, int capacity) {
    return tokenizer_build_chat_prompt(model, text, output, capacity);
}
void falcon_sampler_init(FalconSampler *sampler, float temperature, float top_p, int top_k,
                          float repetition_penalty, uint64_t seed) {
    sampler_init(sampler, temperature, top_p, top_k, repetition_penalty, seed);
}
int falcon_generate_turn(FalconModel *model, FalconRuntime *runtime, FalconSampler *sampler,
                          const int *prompt, int count, int position, int max_new,
                          FalconTokenCallback callback, void *user,
                          int *generated, int *next_position, int *pending) {
    return qwen_generate_turn(model, runtime, sampler, prompt, count, position, max_new,
                               callback, user, generated, next_position, pending);
}
