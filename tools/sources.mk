# 模型结构、层、算子、存储、运行管理和分词分别独立编译。
CORE_SOURCES := \
 src/model/qwen3.c \
 src/layers/decoder_layer.c src/layers/self_attention.c src/layers/mlp.c src/layers/basic.c \
 src/ops/math.c src/ops/kv_cache.c src/ops/q4.c src/ops/q4_vfpu.c src/ops/attention_vfpu.c \
 src/storage/binary.c src/storage/weights.c src/storage/model_loader.c \
 src/runtime/memory.c src/runtime/sampler.c src/runtime/generate.c \
 src/tokenizer/vocabulary.c src/tokenizer/bpe.c src/tokenizer/tokenizer.c \
 src/tokenizer/chat_template.c
