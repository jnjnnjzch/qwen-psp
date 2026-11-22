/* 此文件复用 falcon_h1.c 的文件读取、Q4_0 布局和资源释放函数。 */
int falcon_model_open(FalconModel *model, const char *path,
                      char *error, size_t error_size) {
    uint8_t header[FALCON_HEADER_BYTES], tok[52];
    FalconConfig *c = &model->config;
    uint32_t off, tok_offset;
    uint64_t size;
    int i;
    float *p;
    memset(model, 0, sizeof(*model));
    model->file = fopen(path, "rb");
    if (!model->file) {
        set_error(error, error_size, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
    model->io_buffer_bytes = 64u * 1024u;
    model->io_buffer = malloc(model->io_buffer_bytes);
    if (!model->io_buffer ||
        setvbuf(model->file, (char *)model->io_buffer, _IOFBF,
                model->io_buffer_bytes) != 0) {
        set_error(error, error_size, "cannot allocate Qwen3 file buffer");
        falcon_model_close(model);
        return 0;
    }
    if (fread(header, 1, sizeof(header), model->file) != sizeof(header) ||
        rd32(header) != FALCON_MAGIC || rd32(header + 4) != 2 ||
        rd32(header + 8) != FALCON_HEADER_BYTES) {
        set_error(error, error_size, "expected Qwen3 FHQ4 version 2");
        falcon_model_close(model);
        return 0;
    }
    size = rd64(header + 16);
    if (size > 0x7fffffffu || rd64(header + 24) != FALCON_HEADER_BYTES ||
        rd64(header + 32) >= size || fseek(model->file, 0, SEEK_END) != 0 ||
        ftell(model->file) != (long)size) {
        set_error(error, error_size, "invalid Qwen3 file size or offsets");
        falcon_model_close(model);
        return 0;
    }
#define CFG(I) ((int)rd32(header + 40 + (I) * 4))
    c->dim = CFG(0); c->hidden_dim = CFG(1); c->n_layers = CFG(2);
    c->n_heads = CFG(3); c->n_kv_heads = CFG(4); c->head_dim = CFG(5);
    c->vocab_size = CFG(6); c->model_context = CFG(7); c->context = CFG(8);
    c->pad_id = CFG(9); c->eos_id = CFG(10); c->bos_id = CFG(11);
    c->im_start_id = CFG(12); c->im_end_id = CFG(13);
    c->eom_id = CFG(14); c->eot_id = CFG(15);
    if (c->dim != 1024 || c->hidden_dim != 3072 || c->n_layers != 28 ||
        c->n_heads != 16 || c->n_kv_heads != 8 || c->head_dim != 128 ||
        c->vocab_size != 151936 || c->context < 32 || c->context > 128 ||
        CFG(23) != FALCON_QK) {
        set_error(error, error_size, "unsupported Qwen3-0.6B architecture");
        falcon_model_close(model);
        return 0;
    }
#undef CFG
    c->rope_theta = rdf32(header + 140); c->rms_eps = rdf32(header + 144);
    c->embedding_multiplier = rdf32(header + 148);
    c->lm_head_multiplier = rdf32(header + 152);
    model->file_size = (uint32_t)size;
    model->weights_offset = (uint32_t)rd64(header + 32);
    tok_offset = FALCON_HEADER_BYTES;
    if (!seek_read(model->file, tok_offset, tok, sizeof(tok)) ||
        rd32(tok) != 0x334b4f54u || rd32(tok + 4) != 1 ||
        rd32(tok + 8) != (uint32_t)c->vocab_size ||
        rd32(tok + 36) > model->weights_offset - tok_offset) {
        set_error(error, error_size, "invalid Qwen3 TOK3 tokenizer");
        falcon_model_close(model);
        return 0;
    }
    model->tokenizer.section_bytes = rd32(tok + 36);
    model->tokenizer.section = malloc(model->tokenizer.section_bytes);
    if (!model->tokenizer.section ||
        !seek_read(model->file, tok_offset, model->tokenizer.section,
                   model->tokenizer.section_bytes)) {
        set_error(error, error_size, "cannot allocate or read Qwen3 tokenizer");
        falcon_model_close(model);
        return 0;
    }
    model->tokenizer.vocab_size = c->vocab_size;
    model->tokenizer.max_piece_bytes = (int)rd32(tok + 40);
    model->tokenizer.id_bytes = 4;
    model->tokenizer.special_count = (int)rd32(tok + 44);
#define TOK_PTR(FIELD, OFFSET) \
    model->tokenizer.FIELD = model->tokenizer.section + rd32(tok + OFFSET)
    TOK_PTR(offsets, 16); TOK_PTR(pieces, 20); TOK_PTR(sorted_ids, 24);
    TOK_PTR(ranks, 28); TOK_PTR(byte_ids, 32); TOK_PTR(special_ids, 48);
#undef TOK_PTR
    off = model->weights_offset;
    model->embedding = advance_q4(&off, c->vocab_size, c->dim);
    for (i = 0; i < c->n_layers; ++i) {
        FalconLayerOffsets *l = &model->layers[i];
        l->input_norm = advance_f32(&off, c->dim);
        l->wq = advance_q4(&off, c->n_heads * c->head_dim, c->dim);
        l->wk = advance_q4(&off, c->n_kv_heads * c->head_dim, c->dim);
        l->wv = advance_q4(&off, c->n_kv_heads * c->head_dim, c->dim);
        l->wo = advance_q4(&off, c->dim, c->n_heads * c->head_dim);
        l->q_norm = advance_f32(&off, c->head_dim);
        l->k_norm = advance_f32(&off, c->head_dim);
        l->ffn_norm = advance_f32(&off, c->dim);
        l->w_gate = advance_q4(&off, c->hidden_dim, c->dim);
        l->w_up = advance_q4(&off, c->hidden_dim, c->dim);
        l->w_down = advance_q4(&off, c->dim, c->hidden_dim);
    }
    model->final_norm = advance_f32(&off, c->dim);
    if (off != model->file_size) {
        set_error(error, error_size, "Qwen3 weight layout mismatch");
        falcon_model_close(model);
        return 0;
    }
    model->small_bytes = ((size_t)c->n_layers * (2 * c->dim + 2 * c->head_dim) +
                          c->dim) * sizeof(float);
    model->small_block = malloc(model->small_bytes);
    if (!model->small_block) {
        set_error(error, error_size, "cannot allocate Qwen3 norm weights");
        falcon_model_close(model);
        return 0;
    }
    p = model->small_block;
    for (i = 0; i < c->n_layers; ++i) {
        FalconLayerOffsets *l = &model->layers[i];
        FalconLayerSmall *s = &model->small[i];
        uint32_t offsets[] = {l->input_norm, l->q_norm, l->k_norm, l->ffn_norm};
        int counts[] = {c->dim, c->head_dim, c->head_dim, c->dim};
        const float **targets[] = {&s->input_norm, &s->q_norm, &s->k_norm, &s->ffn_norm};
        int j;
        for (j = 0; j < 4; ++j) {
            if (!seek_read(model->file, offsets[j], p, counts[j] * sizeof(float))) {
                set_error(error, error_size, "short read for Qwen3 norms");
                falcon_model_close(model);
                return 0;
            }
            *targets[j] = p;
            p += counts[j];
        }
    }
    model->final_norm_values = p;
    if (!seek_read(model->file, model->final_norm, p, c->dim * sizeof(float))) {
        set_error(error, error_size, "short read for Qwen3 final norm");
        falcon_model_close(model);
        return 0;
    }
    q4_lut_init();
    return 1;
}
