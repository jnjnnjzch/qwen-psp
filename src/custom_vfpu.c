#include "bench.h"

#include <stdlib.h>
#include <string.h>

#if BENCH_CUSTOM_VH2F
#define CUSTOM_ENTRY bench_matmul_custom_vh2f
#else
#define CUSTOM_ENTRY bench_matmul_custom
#endif

#if defined(_PSP_FW_VERSION)

/* 使用与原始 dot_q4 相同的整数位构造，直接传送 FP32 位模式。 */
#define INTEGER_SCALE(row, destination) \
        "lhu $8, 0(%[" row "])\n" \
        "andi $9, $8, 0x3FF\n" \
        "sll $9, $9, 13\n" \
        "srl $10, $8, 10\n" \
        "andi $10, $10, 0x1F\n" \
        "addiu $10, $10, 112\n" \
        "sll $10, $10, 23\n" \
        "addu $9, $9, $10\n" \
        "srl $10, $8, 15\n" \
        "sll $10, $10, 31\n" \
        "or $9, $9, $10\n" \
        "andi $10, $8, 0x7C00\n" \
        "sltu $10, $0, $10\n" \
        "negu $10, $10\n" \
        "and $9, $9, $10\n" \
        "mtv $9, " destination "\n"

/* Q4 位拆分使用整数指令，量化整数转换、缩放和乘法均在 VFPU 中完成。
 * 每次读取四个压缩字节，分别计算低四位与高四位对应的两个 K 小块。
 * C000/C010/C020/C030 跨完整 K 累计，最终每列对应一个输入向量。 */
static void matrix_tile(const uint8_t *w0, const uint8_t *w1,
                        const uint8_t *w2, const uint8_t *w3,
                        const float *x0, const float *x1,
                        const float *x2, const float *x3,
                        int blocks, float *output) {
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        ".set macro\n"
        "vzero.q C000\n"
        "vzero.q C010\n"
        "vzero.q C020\n"
        "vzero.q C030\n"
        "vone.q C410\n"
        "vfim.s S420, 8.0\n"
        "vscl.q C410, C410, S420\n"
        "lui $24, 0x0F0F\n"
        "ori $24, $24, 0x0F0F\n"
        "1:\n"
#if BENCH_CUSTOM_VH2F
        "lhu $8, 0(%[w0])\n"
        "lhu $9, 0(%[w1])\n"
        "lhu $10, 0(%[w2])\n"
        "lhu $11, 0(%[w3])\n"
        "sll $9, $9, 16\n"
        "or $8, $8, $9\n"
        "sll $11, $11, 16\n"
        "or $10, $10, $11\n"
        "mtv $8, S500\n"
        "mtv $10, S501\n"
        "vh2f.p C400, C500\n"
#else
        INTEGER_SCALE("w0", "S400")
        INTEGER_SCALE("w1", "S401")
        INTEGER_SCALE("w2", "S402")
        INTEGER_SCALE("w3", "S403")
#endif
        "li $25, 4\n"
        "2:\n"
        "ulw $8, 2(%[w0])\n"
        "ulw $9, 2(%[w1])\n"
        "ulw $10, 2(%[w2])\n"
        "ulw $11, 2(%[w3])\n"
        "and $12, $8, $24\n"
        "and $13, $9, $24\n"
        "and $14, $10, $24\n"
        "and $15, $11, $24\n"
        "mtv $12, S500\n"
        "mtv $13, S501\n"
        "mtv $14, S502\n"
        "mtv $15, S503\n"
        "vc2i.s C100, S500\n"
        "vc2i.s C110, S501\n"
        "vc2i.s C120, S502\n"
        "vc2i.s C130, S503\n"
        "vi2f.q C100, C100, 24\n"
        "vi2f.q C110, C110, 24\n"
        "vi2f.q C120, C120, 24\n"
        "vi2f.q C130, C130, 24\n"
        "lv.q C200, 0(%[x0])\n"
        "lv.q C210, 0(%[x1])\n"
        "lv.q C220, 0(%[x2])\n"
        "lv.q C230, 0(%[x3])\n"
        "vsub.q C100, C100, C410\n"
        "vsub.q C110, C110, C410\n"
        "vsub.q C120, C120, C410\n"
        "vsub.q C130, C130, C410\n"
        "srl $8, $8, 4\n"
        "and $12, $8, $24\n"
        "srl $9, $9, 4\n"
        "and $13, $9, $24\n"
        "srl $10, $10, 4\n"
        "and $14, $10, $24\n"
        "srl $11, $11, 4\n"
        "and $15, $11, $24\n"
        "vscl.q C100, C100, S400\n"
        "vscl.q C110, C110, S401\n"
        "vscl.q C120, C120, S402\n"
        "vscl.q C130, C130, S403\n"
        "mtv $12, S510\n"
        "mtv $13, S511\n"
        "mtv $14, S512\n"
        "mtv $15, S513\n"
        "vmmul.q M300, E100, M200\n"
        "vadd.q C000, C000, C300\n"
        "vadd.q C010, C010, C310\n"
        "vadd.q C020, C020, C320\n"
        "vadd.q C030, C030, C330\n"
        "vc2i.s C100, S510\n"
        "vc2i.s C110, S511\n"
        "vc2i.s C120, S512\n"
        "vc2i.s C130, S513\n"
        "vi2f.q C100, C100, 24\n"
        "vi2f.q C110, C110, 24\n"
        "vi2f.q C120, C120, 24\n"
        "vi2f.q C130, C130, 24\n"
        "lv.q C200, 64(%[x0])\n"
        "lv.q C210, 64(%[x1])\n"
        "lv.q C220, 64(%[x2])\n"
        "lv.q C230, 64(%[x3])\n"
        "vsub.q C100, C100, C410\n"
        "vsub.q C110, C110, C410\n"
        "vsub.q C120, C120, C410\n"
        "vsub.q C130, C130, C410\n"
        "addiu %[w0], %[w0], 4\n"
        "addiu %[w1], %[w1], 4\n"
        "addiu %[w2], %[w2], 4\n"
        "addiu %[w3], %[w3], 4\n"
        "vscl.q C100, C100, S400\n"
        "vscl.q C110, C110, S401\n"
        "vscl.q C120, C120, S402\n"
        "vscl.q C130, C130, S403\n"
        "addiu %[x0], %[x0], 16\n"
        "addiu %[x1], %[x1], 16\n"
        "addiu %[x2], %[x2], 16\n"
        "addiu %[x3], %[x3], 16\n"
        "vmmul.q M300, E100, M200\n"
        "vadd.q C000, C000, C300\n"
        "vadd.q C010, C010, C310\n"
        "vadd.q C020, C020, C320\n"
        "vadd.q C030, C030, C330\n"
        "addiu $25, $25, -1\n"
        "bnez $25, 2b\n"
        "nop\n"
        "addiu %[w0], %[w0], 2\n"
        "addiu %[w1], %[w1], 2\n"
        "addiu %[w2], %[w2], 2\n"
        "addiu %[w3], %[w3], 2\n"
        "addiu %[x0], %[x0], 64\n"
        "addiu %[x1], %[x1], 64\n"
        "addiu %[x2], %[x2], 64\n"
        "addiu %[x3], %[x3], 64\n"
        "addiu %[blocks], %[blocks], -1\n"
        "bnez %[blocks], 1b\n"
        "nop\n"
        "sv.q C000, 0(%[out])\n"
        "sv.q C010, 16(%[out])\n"
        "sv.q C020, 32(%[out])\n"
        "sv.q C030, 48(%[out])\n"
        ".set pop\n"
        : [w0] "+&r"(w0), [w1] "+&r"(w1), [w2] "+&r"(w2), [w3] "+&r"(w3),
          [x0] "+&r"(x0), [x1] "+&r"(x1), [x2] "+&r"(x2), [x3] "+&r"(x3),
          [blocks] "+&r"(blocks)
        : [out] "r"(output)
        : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$24", "$25", "memory");
}

#if !BENCH_CUSTOM_VH2F
int bench_custom_available(void) {
    return 1;
}
#endif

void CUSTOM_ENTRY(const BenchTask *task) {
    float accumulator[4 * 4] __attribute__((aligned(16)));
    for (uint32_t row = 0; row < task->rows; row += 4) {
        uint32_t rows = task->rows - row < 4 ? task->rows - row : 4;
        const uint8_t *w0 = task->weights + row * task->weight_row_bytes;
        const uint8_t *w1 = rows > 1 ? w0 + task->weight_row_bytes : w0;
        const uint8_t *w2 = rows > 2 ? w0 + 2 * task->weight_row_bytes : w0;
        const uint8_t *w3 = rows > 3 ? w0 + 3 * task->weight_row_bytes : w0;
        for (uint32_t batch = 0; batch < task->batch; batch += 4) {
            uint32_t batches = task->batch - batch < 4 ? task->batch - batch : 4;
            const float *x0 = task->input + batch * task->input_stride;
            /* 边界分量重复有效行或输入，最终仅写出真实切片中的元素。 */
            const float *x1 = batches > 1 ? x0 + task->input_stride : x0;
            const float *x2 = batches > 2 ? x0 + 2 * task->input_stride : x0;
            const float *x3 = batches > 3 ? x0 + 3 * task->input_stride : x0;
            matrix_tile(w0, w1, w2, w3, x0, x1, x2, x3, (int)task->columns / 32, accumulator);
            for (uint32_t b = 0; b < batches; ++b)
                memcpy(task->output + (batch + b) * task->rows + row,
                       accumulator + b * 4, rows * sizeof(float));
        }
    }
}

#else

#if !BENCH_CUSTOM_VH2F
int bench_custom_available(void) {
    return 0;
}
#endif

void CUSTOM_ENTRY(const BenchTask *task) {
    (void)task;
    abort();
}

#endif
