#ifndef QWEN_STORAGE_H
#define QWEN_STORAGE_H

#include "qwen_model.h"
#include "binary.h"

int weights_read(QwenModel *model, uint32_t offset, void *destination, size_t bytes);
const uint8_t *weights_cached_pointer(const QwenModel *model, uint32_t offset, size_t bytes);
int weights_copy_cached(const QwenModel *model, uint32_t offset, void *destination, size_t bytes);

#endif
