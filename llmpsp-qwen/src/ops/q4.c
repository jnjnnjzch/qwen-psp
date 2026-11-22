#include "ops.h"
#include "storage/binary.h"

#include <string.h>

/* 一块 Q4_0 保存 32 个权重：2 字节 FP16 scale + 16 字节整数。
 * 一个整数占 4 位，因此一个字节同时保存两个整数。 */
static float low_values[256], high_values[256];
static int lookup_ready;

size_t q4_row_bytes(int columns) {
    return (size_t)(columns / QWEN_Q4_BLOCK_SIZE) * 18u;
}

void q4_initialize_lookup(void) {
    int byte;
    if (lookup_ready) return;
    for (byte = 0; byte < 256; ++byte) {
        low_values[byte] = (float)((byte & 15) - 8);
        high_values[byte] = (float)((byte >> 4) - 8);
    }
    lookup_ready = 1;
}

static float half_to_float(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)half & 0x3ffu;
    uint32_t bits;
    float result;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x3ffu;
            bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    memcpy(&result, &bits, sizeof(result));
    return result;
}

void q4_decode_row(float *output, const uint8_t *row, int columns) {
    int block, index;
    for (block = 0; block < columns / QWEN_Q4_BLOCK_SIZE; ++block) {
        const uint8_t *bytes = row + block * 18;
        float scale = half_to_float(read_u16_le(bytes));
        int start = block * QWEN_Q4_BLOCK_SIZE;
        for (index = 0; index < 16; ++index) {
            uint8_t packed = bytes[2 + index];
            output[start + index] = scale * low_values[packed];
            output[start + index + 16] = scale * high_values[packed];
        }
    }
}

void q4_input_block_sums(const float *input, int columns, float *block_sums) {
    int block, index;
    for (block = 0; block < columns / QWEN_Q4_BLOCK_SIZE; ++block) {
        float sum = 0.0f;
        const float *values = input + block * QWEN_Q4_BLOCK_SIZE;
        for (index = 0; index < QWEN_Q4_BLOCK_SIZE; ++index) sum += values[index];
        block_sums[block] = sum;
    }
}

#if defined(_PSP_FW_VERSION)
float q4_dot_vfpu(const uint8_t *row, const float *input, int blocks, const float *block_sums);
#endif

/* 输入一行量化权重和一个向量，输出一个数：它们的点积。
 * PSP 构建链接独立的 q4_vfpu.c；电脑构建使用下面沿用原项目的标量循环。 */
float q4_dot(const uint8_t *row, const float *input, int columns, const float *block_sums) {
#if defined(_PSP_FW_VERSION)
    return q4_dot_vfpu(row, input, columns / QWEN_Q4_BLOCK_SIZE, block_sums);
#else
    int block, index;
    float sum = 0.0f;
    (void)block_sums;
    for (block = 0; block < columns / QWEN_Q4_BLOCK_SIZE; ++block) {
        const uint8_t *bytes = row + block * 18;
        const float *values = input + block * QWEN_Q4_BLOCK_SIZE;
        float scale = half_to_float(read_u16_le(bytes));
        float block_sum = 0.0f;
        for (index = 0; index < 16; ++index) {
            uint8_t packed = bytes[2 + index];
            block_sum += low_values[packed] * values[index];
            block_sum += high_values[packed] * values[index + 16];
        }
        sum += scale * block_sum;
    }
    return sum;
#endif
}
