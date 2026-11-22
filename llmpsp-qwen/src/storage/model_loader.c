#include "storage.h"
#include "ops/ops.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* 加载器根据 FHQ4 v2 的排列顺序，为各个组件填写权重位置。
 * 这里创建层的描述信息，实际的计算写在 layers/ 中。 */
static Linear load_linear(uint32_t *offset, int input_features, int output_features) {
    Linear layer = {*offset, input_features, output_features};
    *offset += (uint32_t)((size_t)output_features * q4_row_bytes(input_features));
    return layer;
}

static RMSNorm load_norm(uint32_t *offset, int features, float epsilon) {
    RMSNorm layer = {*offset, NULL, features, epsilon};
    *offset += (uint32_t)((size_t)features * sizeof(float));
    return layer;
}

static int read_norm(QwenModel *model, RMSNorm *layer, float **destination) {
    size_t bytes = (size_t)layer->features * sizeof(float);
    if (!weights_read_file(model->file, layer->weight_offset, *destination, bytes)) return 0;
    layer->weight = *destination;
    *destination += layer->features;
    return 1;
}

int qwen_model_open(QwenModel *model, const char *path, char *error, size_t size) {
    uint8_t header[256], tokenizer_header[52];
    QwenConfig *config = &model->config;
    uint32_t offset;
    uint64_t file_size;
    float *norm_destination;
    int index;
    memset(model, 0, sizeof(*model));
    model->file = fopen(path, "rb");
    if (!model->file) {
        error_format(error, size, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
    model->io_buffer_bytes = 64u * 1024u;
    model->io_buffer = malloc(model->io_buffer_bytes);
    if (!model->io_buffer || setvbuf(model->file, (char *)model->io_buffer, _IOFBF,
                                    model->io_buffer_bytes) != 0) {
        error_format(error, size, "cannot allocate model file buffer");
        goto failure;
    }
    if (fread(header, 1, sizeof(header), model->file) != sizeof(header) ||
        read_u32_le(header) != 0x34514846u || read_u32_le(header + 4) != 2 ||
        read_u32_le(header + 8) != sizeof(header)) {
        error_format(error, size, "expected Qwen3 FHQ4 version 2");
        goto failure;
    }
    file_size = read_u64_le(header + 16);
    if (file_size > 0x7fffffffu || read_u64_le(header + 24) != sizeof(header) ||
        read_u64_le(header + 32) >= file_size || fseek(model->file, 0, SEEK_END) != 0 ||
        ftell(model->file) != (long)file_size) {
        error_format(error, size, "invalid model file size or offsets");
        goto failure;
    }
#define CONFIG_VALUE(I) ((int)read_u32_le(header + 40 + (I) * 4))
    config->hidden_size = CONFIG_VALUE(0); config->intermediate_size = CONFIG_VALUE(1);
    config->num_layers = CONFIG_VALUE(2); config->num_attention_heads = CONFIG_VALUE(3);
    config->num_key_value_heads = CONFIG_VALUE(4); config->head_dim = CONFIG_VALUE(5);
    config->vocab_size = CONFIG_VALUE(6); config->model_context = CONFIG_VALUE(7);
    config->context = CONFIG_VALUE(8); config->pad_id = CONFIG_VALUE(9);
    config->eos_id = CONFIG_VALUE(10); config->bos_id = CONFIG_VALUE(11);
    config->im_start_id = CONFIG_VALUE(12); config->im_end_id = CONFIG_VALUE(13);
    config->eom_id = CONFIG_VALUE(14); config->eot_id = CONFIG_VALUE(15);
    if (config->hidden_size != 1024 || config->intermediate_size != 3072 ||
        config->num_layers != 28 || config->num_attention_heads != 16 ||
        config->num_key_value_heads != 8 || config->head_dim != 128 ||
        config->vocab_size != 151936 || config->context < 32 || config->context > 128 ||
        CONFIG_VALUE(23) != QWEN_Q4_BLOCK_SIZE) {
        error_format(error, size, "unsupported Qwen3-0.6B architecture");
        goto failure;
    }
#undef CONFIG_VALUE
    config->rope_theta = read_f32_le(header + 140);
    config->rms_norm_eps = read_f32_le(header + 144);
    model->file_size = (uint32_t)file_size;
    model->weights_offset = (uint32_t)read_u64_le(header + 32);
    if (!weights_read_file(model->file, 256, tokenizer_header, sizeof(tokenizer_header)) ||
        read_u32_le(tokenizer_header) != 0x334b4f54u || read_u32_le(tokenizer_header + 4) != 1 ||
        read_u32_le(tokenizer_header + 8) != (uint32_t)config->vocab_size ||
        read_u32_le(tokenizer_header + 36) > model->weights_offset - 256) {
        error_format(error, size, "invalid TOK3 tokenizer");
        goto failure;
    }
    model->tokenizer.section_bytes = read_u32_le(tokenizer_header + 36);
    model->tokenizer.section = malloc(model->tokenizer.section_bytes);
    if (!model->tokenizer.section || !weights_read_file(model->file, 256,
         model->tokenizer.section, model->tokenizer.section_bytes)) {
        error_format(error, size, "cannot allocate or read tokenizer");
        goto failure;
    }
    model->tokenizer.vocab_size = config->vocab_size;
    model->tokenizer.max_piece_bytes = (int)read_u32_le(tokenizer_header + 40);
    model->tokenizer.special_count = (int)read_u32_le(tokenizer_header + 44);
#define TOKENIZER_POINTER(NAME, BYTE) \
    model->tokenizer.NAME = model->tokenizer.section + read_u32_le(tokenizer_header + BYTE)
    TOKENIZER_POINTER(offsets, 16); TOKENIZER_POINTER(pieces, 20);
    TOKENIZER_POINTER(sorted_ids, 24); TOKENIZER_POINTER(ranks, 28);
    TOKENIZER_POINTER(byte_ids, 32); TOKENIZER_POINTER(special_ids, 48);
#undef TOKENIZER_POINTER

    offset = model->weights_offset;
    model->embed_tokens.weight_offset = offset;
    model->embed_tokens.num_embeddings = config->vocab_size;
    model->embed_tokens.embedding_dim = config->hidden_size;
    /* 本模型共享 Embedding 与 LM head 的权重。 */
    model->lm_head = load_linear(&offset, config->hidden_size, config->vocab_size);
    for (index = 0; index < config->num_layers; ++index) {
        DecoderLayer *layer = &model->layers[index];
        SelfAttention *attention = &layer->self_attn;
        MLP *mlp = &layer->mlp;
        layer->input_layernorm = load_norm(&offset, config->hidden_size, config->rms_norm_eps);
        attention->q_proj = load_linear(&offset, config->hidden_size,
                                         config->num_attention_heads * config->head_dim);
        attention->k_proj = load_linear(&offset, config->hidden_size,
                                         config->num_key_value_heads * config->head_dim);
        attention->v_proj = load_linear(&offset, config->hidden_size,
                                         config->num_key_value_heads * config->head_dim);
        attention->o_proj = load_linear(&offset, config->num_attention_heads * config->head_dim,
                                         config->hidden_size);
        attention->q_norm = load_norm(&offset, config->head_dim, config->rms_norm_eps);
        attention->k_norm = load_norm(&offset, config->head_dim, config->rms_norm_eps);
        layer->post_attention_layernorm = load_norm(&offset, config->hidden_size, config->rms_norm_eps);
        mlp->gate_proj = load_linear(&offset, config->hidden_size, config->intermediate_size);
        mlp->up_proj = load_linear(&offset, config->hidden_size, config->intermediate_size);
        mlp->down_proj = load_linear(&offset, config->intermediate_size, config->hidden_size);
    }
    model->norm = load_norm(&offset, config->hidden_size, config->rms_norm_eps);
    if (offset != model->file_size) {
        error_format(error, size, "model weight layout mismatch");
        goto failure;
    }
    model->norm_weight_bytes = ((size_t)config->num_layers *
        (2 * config->hidden_size + 2 * config->head_dim) + config->hidden_size) * sizeof(float);
    model->norm_weights = malloc(model->norm_weight_bytes);
    if (!model->norm_weights) {
        error_format(error, size, "cannot allocate normalization weights");
        goto failure;
    }
    norm_destination = model->norm_weights;
    for (index = 0; index < config->num_layers; ++index) {
        DecoderLayer *layer = &model->layers[index];
        if (!read_norm(model, &layer->input_layernorm, &norm_destination) ||
            !read_norm(model, &layer->self_attn.q_norm, &norm_destination) ||
            !read_norm(model, &layer->self_attn.k_norm, &norm_destination) ||
            !read_norm(model, &layer->post_attention_layernorm, &norm_destination)) {
            error_format(error, size, "short read for normalization weights");
            goto failure;
        }
    }
    if (!read_norm(model, &model->norm, &norm_destination)) {
        error_format(error, size, "short read for final normalization");
        goto failure;
    }
    q4_initialize_lookup();
    return 1;
failure:
    qwen_model_close(model);
    return 0;
}
