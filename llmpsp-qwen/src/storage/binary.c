#include "binary.h"

#include <stdarg.h>
#include <string.h>

/* FHQ4 是原项目自己的文件格式；这些函数沿用它的小端字节读取方式。 */
uint16_t read_u16_le(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

uint32_t read_u32_le(const uint8_t *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

uint64_t read_u64_le(const uint8_t *bytes) {
    return (uint64_t)read_u32_le(bytes) | ((uint64_t)read_u32_le(bytes + 4) << 32);
}

float read_f32_le(const uint8_t *bytes) {
    uint32_t bits = read_u32_le(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void error_format(char *buffer, size_t size, const char *format, ...) {
    va_list arguments;
    if (!buffer || !size) return;
    va_start(arguments, format);
    vsnprintf(buffer, size, format, arguments);
    va_end(arguments);
}

int weights_read_file(FILE *file, uint32_t offset, void *destination, size_t bytes) {
    return fseek(file, (long)offset, SEEK_SET) == 0 && fread(destination, 1, bytes, file) == bytes;
}
