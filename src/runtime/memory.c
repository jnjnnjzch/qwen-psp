#include "qwen_model.h"
#include "ops/ops.h"
#include "storage/binary.h"

#include <stdlib.h>
#include <string.h>

/* VFPU 每次读取 16 字节。将数组起始地址安排在 16 字节边界，
 * 并在前面保存 malloc 返回的原始地址，释放时可以找到它。 */
static void *allocate_array(QwenRuntime *runtime, size_t count, size_t item_size) {
    size_t bytes = count * item_size;
    size_t total = bytes + 16 + sizeof(void *);
    uint8_t *raw = malloc(total), *aligned;
    if (!raw) return NULL;
    aligned = (uint8_t *)(((uintptr_t)raw + sizeof(void *) + 15u) & ~(uintptr_t)15u);
    ((void **)aligned)[-1] = raw;
    memset(aligned, 0, bytes);
    runtime->bytes_allocated += total;
    return aligned;
}

static void free_array(void *memory) {
    if (memory) free(((void **)memory)[-1]);
}

int qwen_runtime_init(QwenRuntime *runtime, const QwenModel *model, char *error, size_t size) {
    const QwenConfig *config = &model->config;
    KVCache *cache = &runtime->kv_cache;
    size_t kv_values = (size_t)config->num_layers * config->context *
                        config->num_key_value_heads * config->head_dim;
    size_t kv_scales = (size_t)config->num_layers * config->context * config->num_key_value_heads;
    memset(runtime, 0, sizeof(*runtime));
#define FLOAT_ARRAY(NAME, COUNT) runtime->NAME = allocate_array(runtime, COUNT, sizeof(float))
    FLOAT_ARRAY(hidden, config->hidden_size);
    FLOAT_ARRAY(normalized, config->hidden_size);
    FLOAT_ARRAY(query, config->num_attention_heads * config->head_dim);
    FLOAT_ARRAY(key, config->num_key_value_heads * config->head_dim);
    FLOAT_ARRAY(value, config->num_key_value_heads * config->head_dim);
    FLOAT_ARRAY(attention_scores, config->context);
    FLOAT_ARRAY(attention_output, config->num_attention_heads * config->head_dim);
    FLOAT_ARRAY(branch_output, config->hidden_size);
    FLOAT_ARRAY(mlp_gate, config->intermediate_size);
    FLOAT_ARRAY(mlp_up, config->intermediate_size);
    FLOAT_ARRAY(logits, config->vocab_size);
#undef FLOAT_ARRAY
    cache->keys = allocate_array(runtime, kv_values, sizeof(int8_t));
    cache->values = allocate_array(runtime, kv_values, sizeof(int8_t));
    cache->key_scales = allocate_array(runtime, kv_scales, sizeof(float));
    cache->value_scales = allocate_array(runtime, kv_scales, sizeof(float));
    cache->num_layers = config->num_layers;
    cache->context = config->context;
    cache->num_heads = config->num_key_value_heads;
    cache->head_dim = config->head_dim;
    runtime->weight_buffer_bytes = 32u * 1024u;
    runtime->weight_buffer = allocate_array(runtime, runtime->weight_buffer_bytes, 1);
    if (!runtime->hidden || !runtime->normalized || !runtime->query || !runtime->key ||
        !runtime->value || !runtime->attention_scores || !runtime->attention_output ||
        !runtime->branch_output || !runtime->mlp_gate || !runtime->mlp_up || !runtime->logits ||
        !cache->keys || !cache->values || !cache->key_scales || !cache->value_scales ||
        !runtime->weight_buffer) {
        error_format(error, size, "cannot allocate Qwen runtime");
        qwen_runtime_free(runtime);
        return 0;
    }
    rope_initialize(runtime, config);
    return 1;
}

void qwen_runtime_reset(QwenRuntime *runtime, const QwenModel *model) {
    size_t values = (size_t)model->config.num_layers * model->config.context *
                     model->config.num_key_value_heads * model->config.head_dim;
    memset(runtime->kv_cache.keys, 0, values);
    memset(runtime->kv_cache.values, 0, values);
    runtime->io_error = 0;
}

void qwen_runtime_free(QwenRuntime *runtime) {
    free_array(runtime->hidden); free_array(runtime->normalized);
    free_array(runtime->query); free_array(runtime->key); free_array(runtime->value);
    free_array(runtime->attention_scores); free_array(runtime->attention_output);
    free_array(runtime->branch_output); free_array(runtime->mlp_gate); free_array(runtime->mlp_up);
    free_array(runtime->logits); free_array(runtime->weight_buffer);
    free_array(runtime->kv_cache.keys); free_array(runtime->kv_cache.values);
    free_array(runtime->kv_cache.key_scales); free_array(runtime->kv_cache.value_scales);
    memset(runtime, 0, sizeof(*runtime));
}

void qwen_runtime_set_progress(QwenRuntime *runtime, QwenProgressCallback callback, void *user) {
    runtime->progress = callback;
    runtime->progress_user = user;
}

size_t qwen_runtime_resident_bytes(const QwenRuntime *runtime) {
    return runtime->bytes_allocated;
}
