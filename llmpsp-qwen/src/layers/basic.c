#include "qwen_model.h"
#include "ops/ops.h"
#include "storage/storage.h"

void rmsnorm_forward(const RMSNorm *layer, const float *input, float *output) {
    rms_norm(output, input, layer->weight, layer->features, layer->epsilon);
}

int embedding_forward(const Embedding *layer, QwenModel *model, QwenRuntime *runtime,
                      int token, float *output) {
    size_t bytes = q4_row_bytes(layer->embedding_dim);
    uint32_t offset = layer->weight_offset + (uint32_t)((size_t)token * bytes);
    if (bytes > runtime->weight_buffer_bytes ||
        !weights_read(model, offset, runtime->weight_buffer, bytes)) {
        runtime->io_error = 1;
        return 0;
    }
    q4_decode_row(output, runtime->weight_buffer, layer->embedding_dim);
    return 1;
}

/* Linear 的一次调用：output = weight @ input。
 * 每一行权重与 input 做点积，得到 output 中的一个元素。
 * 下方的 row_batch 专指一次从文件读取多少行权重，与 tokenizer 的文本片段无关。 */
int linear_forward(const Linear *layer, QwenModel *model, QwenRuntime *runtime,
                   const float *input, float *output) {
    size_t row_bytes = q4_row_bytes(layer->in_features);
    int row, file_active = 0;
    uint32_t file_position = 0;
    unsigned int started = runtime->clock_us ? runtime->clock_us() : 0;
    const float *block_sums = NULL;
#if defined(_PSP_FW_VERSION)
    static float sums[96] __attribute__((aligned(16)));
    q4_input_block_sums(input, layer->in_features, sums);
    block_sums = sums;
#endif
    if (row_bytes > runtime->weight_buffer_bytes) {
        runtime->io_error = 1;
        return 0;
    }
    for (row = 0; row < layer->out_features; ++row) {
        uint32_t offset = layer->weight_offset + (uint32_t)((size_t)row * row_bytes);
        const uint8_t *cached = weights_cached_pointer(model, offset, row_bytes);
        if (cached) {
            output[row] = q4_dot(cached, input, layer->in_features, block_sums);
            continue;
        }
        if (weights_copy_cached(model, offset, runtime->weight_buffer, row_bytes)) {
            output[row] = q4_dot(runtime->weight_buffer, input, layer->in_features, block_sums);
            continue;
        }
        {
            int row_batch = (int)(runtime->weight_buffer_bytes / row_bytes);
            int remaining_rows = layer->out_features - row, index;
            size_t batch_bytes;
            if (row_batch > remaining_rows) row_batch = remaining_rows;
            batch_bytes = (size_t)row_batch * row_bytes;
            if (!file_active || file_position != offset) {
                if (fseek(model->file, (long)offset, SEEK_SET) != 0) {
                    runtime->io_error = 1;
                    return 0;
                }
                file_active = 1;
            }
            if (fread(runtime->weight_buffer, 1, batch_bytes, model->file) != batch_bytes) {
                runtime->io_error = 1;
                return 0;
            }
            file_position = offset + (uint32_t)batch_bytes;
            runtime->streamed_bytes += batch_bytes;
            for (index = 0; index < row_batch; ++index)
                output[row + index] = q4_dot(runtime->weight_buffer + (size_t)index * row_bytes,
                                             input, layer->in_features, block_sums);
            row += row_batch - 1;
        }
    }
    if (runtime->clock_us) runtime->matvec_us += runtime->clock_us() - started;
    return 1;
}
