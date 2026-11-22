#include "bench.h"

#include <string.h>

/* 沿用 LLMPSP 的 FP16 转换、查表和标量累加顺序。 */
static float q4_lut_low[256], q4_lut_high[256];

void bench_scalar_init(void) {
    for (int i = 0; i < 256; ++i) {
        q4_lut_low[i] = (float)((i & 15) - 8);
        q4_lut_high[i] = (float)((i >> 4) - 8);
    }
}

static float half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)h & 0x3ffu;
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

float bench_dot_scalar(const uint8_t *row, const float *x, int blocks, const float *block_sums) {
    float sum = 0.0f;
    (void)block_sums;
    for (int block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 18;
        const float *v = x + block * 32;
        float d = half_to_float((uint16_t)(b[0] | ((uint16_t)b[1] << 8)));
        float block_sum = 0.0f;
        for (int j = 0; j < 16; ++j) {
            uint8_t packed = b[2 + j];
            block_sum += q4_lut_low[packed] * v[j];
            block_sum += q4_lut_high[packed] * v[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}

void bench_matmul_scalar(const BenchTask *task) {
    for (uint32_t batch = 0; batch < task->batch; ++batch)
        for (uint32_t row = 0; row < task->rows; ++row)
            task->output[batch * task->rows + row] =
                bench_dot_scalar(task->weights + row * task->weight_row_bytes,
                                 task->input + batch * task->input_stride,
                                 (int)task->columns / 32, NULL);
}
