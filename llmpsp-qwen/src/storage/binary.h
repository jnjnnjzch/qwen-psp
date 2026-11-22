#ifndef QWEN_BINARY_H
#define QWEN_BINARY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

uint16_t read_u16_le(const uint8_t *bytes);
uint32_t read_u32_le(const uint8_t *bytes);
uint64_t read_u64_le(const uint8_t *bytes);
float read_f32_le(const uint8_t *bytes);
void error_format(char *buffer, size_t size, const char *format, ...);
int weights_read_file(FILE *file, uint32_t offset, void *destination, size_t bytes);

#endif
