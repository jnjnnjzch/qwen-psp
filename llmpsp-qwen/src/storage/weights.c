#include "storage.h"

#include <stdlib.h>
#include <string.h>

/* 一张指针表描述哪些 256 KiB 权重块在内存中。NULL 表示该块从模型文件读取。 */
const uint8_t *weights_cached_pointer(const QwenModel *model, uint32_t offset, size_t bytes) {
    size_t relative, region, block, inside;
    const uint8_t *base;
    if (!model->weight_cache || offset < model->weights_offset) return NULL;
    relative = (size_t)(offset - model->weights_offset);
    region = (size_t)(model->file_size - model->weights_offset);
    if (relative > region || bytes > region - relative) return NULL;
    block = relative / model->weight_cache_block_bytes;
    inside = relative % model->weight_cache_block_bytes;
    if (inside + bytes > model->weight_cache_block_bytes) return NULL;
    base = model->weight_cache[block];
    return base ? base + inside : NULL;
}

int weights_copy_cached(const QwenModel *model, uint32_t offset, void *destination, size_t bytes) {
    size_t relative, region, copied = 0;
    uint8_t *output = destination;
    if (!model->weight_cache || offset < model->weights_offset) return 0;
    relative = (size_t)(offset - model->weights_offset);
    region = (size_t)(model->file_size - model->weights_offset);
    if (relative > region || bytes > region - relative) return 0;
    while (copied < bytes) {
        size_t position = relative + copied;
        size_t block = position / model->weight_cache_block_bytes;
        size_t inside = position % model->weight_cache_block_bytes;
        size_t available = model->weight_cache_block_bytes - inside;
        size_t amount = bytes - copied < available ? bytes - copied : available;
        const uint8_t *base = model->weight_cache[block];
        if (!base) return 0;
        memcpy(output + copied, base + inside, amount);
        copied += amount;
    }
    return 1;
}

int weights_read(QwenModel *model, uint32_t offset, void *destination, size_t bytes) {
    if (weights_copy_cached(model, offset, destination, bytes)) return 1;
    /* 缓存未覆盖整个请求时，从文件读取完整请求。文件读取失败立即返回失败。 */
    return weights_read_file(model->file, offset, destination, bytes);
}

static int cache_is_extra(const QwenModel *model, const uint8_t *pointer) {
    uintptr_t address = (uintptr_t)pointer, start = (uintptr_t)model->cache_extra_base;
    return model->cache_extra_base && address >= start && address < start + model->cache_extra_bytes;
}

size_t qwen_model_cache_weights_extra(QwenModel *model, size_t target_bytes,
                                     uint8_t *extra, size_t extra_bytes, int stripe,
                                     char *error, size_t size) {
    const size_t block_bytes = 256u * 1024u;
    size_t region, total_blocks, extra_blocks, heap_blocks, pool_count = 0, used = 0, loaded = 0;
    size_t embedding_blocks, layer_blocks = 0, layer_holes = 0, index;
    uint8_t **pool;
    if (error && size) error[0] = '\0';
    if (model->weight_cache) return model->weight_cache_bytes;
    if (!target_bytes && !extra_bytes) return 0;
    region = model->file_size - model->weights_offset;
    total_blocks = (region + block_bytes - 1) / block_bytes;
    extra_blocks = extra ? extra_bytes / block_bytes : 0;
    if (extra_blocks > total_blocks) extra_blocks = total_blocks;
    heap_blocks = (target_bytes + block_bytes - 1) / block_bytes;
    pool = calloc(total_blocks, sizeof(*pool));
    model->weight_cache = calloc(total_blocks, sizeof(*model->weight_cache));
    if (!pool || !model->weight_cache) {
        error_format(error, size, "cannot allocate weight cache index");
        free(pool); free(model->weight_cache); model->weight_cache = NULL;
        return 0;
    }
    model->weight_cache_block_bytes = block_bytes;
    model->weight_cache_blocks = (int)total_blocks;
    model->cache_extra_base = extra;
    model->cache_extra_bytes = extra_blocks * block_bytes;
    for (index = 0; index < extra_blocks; ++index) pool[pool_count++] = extra + index * block_bytes;
    for (index = 0; index < heap_blocks && pool_count < total_blocks; ++index) {
        uint8_t *memory = malloc(block_bytes);
        if (!memory) {
            error_format(error, size, "cannot allocate requested weight cache");
            goto failure;
        }
        pool[pool_count++] = memory;
    }
    embedding_blocks = ((size_t)(model->layers[0].input_layernorm.weight_offset - model->weights_offset) +
                         block_bytes - 1) / block_bytes;
    if (stripe && pool_count > embedding_blocks && pool_count < total_blocks) {
        layer_blocks = total_blocks - embedding_blocks;
        layer_holes = layer_blocks - (pool_count - embedding_blocks);
    } else stripe = 0;
    for (index = 0; index < total_blocks && used < pool_count; ++index) {
        size_t amount;
        int cached;
        if (!stripe) cached = used < pool_count;
        else if (index < embedding_blocks) cached = 1;
        else {
            size_t layer_block = index - embedding_blocks;
            cached = (layer_block + 1) * layer_holes / layer_blocks == layer_block * layer_holes / layer_blocks;
        }
        if (!cached) continue;
        amount = region - index * block_bytes;
        if (amount > block_bytes) amount = block_bytes;
        if (!weights_read_file(model->file, model->weights_offset + (uint32_t)(index * block_bytes),
                               pool[used], amount)) {
            error_format(error, size, "short read while filling weight cache");
            goto failure;
        }
        model->weight_cache[index] = pool[used++];
        loaded += amount;
    }
    model->weight_cache_bytes = loaded;
    free(pool);
    return loaded;
failure:
    for (index = 0; index < pool_count; ++index)
        if (!cache_is_extra(model, pool[index])) free(pool[index]);
    free(pool); free(model->weight_cache);
    model->weight_cache = NULL;
    model->weight_cache_blocks = 0;
    return 0;
}

size_t qwen_model_cache_weights(QwenModel *model, size_t bytes, char *error, size_t size) {
    return qwen_model_cache_weights_extra(model, bytes, NULL, 0, 0, error, size);
}

void qwen_model_close(QwenModel *model) {
    int block;
    if (model->file) fclose(model->file);
    for (block = 0; block < model->weight_cache_blocks; ++block)
        if (!cache_is_extra(model, model->weight_cache[block])) free(model->weight_cache[block]);
    free(model->weight_cache);
    free(model->io_buffer);
    free(model->norm_weights);
    free(model->tokenizer.section);
    memset(model, 0, sizeof(*model));
}

size_t qwen_model_resident_bytes(const QwenModel *model) {
    return model->io_buffer_bytes + model->tokenizer.section_bytes +
           model->norm_weight_bytes + model->weight_cache_bytes;
}
