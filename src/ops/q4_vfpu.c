#include <stdint.h>

#if defined(_PSP_FW_VERSION)
/* 原项目 falcon_dot_q4_vfpu 的指令序列。
 * 每个 Q4_0 权重是 scale * (整数 - 8)。VFPU 累加整数部分，
 * 最后减去 8 * scale * input_block_sum，得到这一行的点积。
 * 输入浮点数组和 scratch 的起始地址都在 16 字节边界。 */
float q4_dot_vfpu(const uint8_t *row, const float *input,
                   int blocks, const float *block_sums) {
    static uint32_t scratch[12] __attribute__((aligned(16)));
    float result;
    const uint8_t *row_pointer = row;
    const float *input_pointer = input, *sum_pointer = block_sums;
    int remaining_blocks = blocks;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "vzero.q C000\n"
        "vzero.s S010\n"
        "lui   $24, 0x0F0F\n"
        "ori   $24, $24, 0x0F0F\n"
        "1:\n"
        "ulw   $8, 2(%[row])\n"
        "ulw   $9, 6(%[row])\n"
        "ulw   $10, 10(%[row])\n"
        "ulw   $11, 14(%[row])\n"
        "and   $12, $8, $24\n"
        "srl   $8, $8, 4\n"
        "and   $8, $8, $24\n"
        "and   $13, $9, $24\n"
        "srl   $9, $9, 4\n"
        "and   $9, $9, $24\n"
        "and   $14, $10, $24\n"
        "srl   $10, $10, 4\n"
        "and   $10, $10, $24\n"
        "and   $15, $11, $24\n"
        "srl   $11, $11, 4\n"
        "and   $11, $11, $24\n"
        "sw    $12, 0(%[sc])\n"
        "sw    $13, 4(%[sc])\n"
        "sw    $14, 8(%[sc])\n"
        "sw    $15, 12(%[sc])\n"
        "sw    $8, 16(%[sc])\n"
        "sw    $9, 20(%[sc])\n"
        "sw    $10, 24(%[sc])\n"
        "sw    $11, 28(%[sc])\n"
        /* 将当前块的 FP16 scale 转成 FP32；沿用原算子对非规格化数的处理。 */
        "lhu   $25, 0(%[row])\n"
        "andi  $2, $25, 0x3FF\n"
        "sll   $2, $2, 13\n"
        "srl   $3, $25, 10\n"
        "andi  $3, $3, 0x1F\n"
        "addiu $3, $3, 112\n"
        "sll   $3, $3, 23\n"
        "addu  $2, $2, $3\n"
        "srl   $3, $25, 15\n"
        "sll   $3, $3, 31\n"
        "or    $2, $2, $3\n"
        "andi  $3, $25, 0x7C00\n"
        "sltu  $3, $0, $3\n"
        "negu  $3, $3\n"
        "and   $2, $2, $3\n"
        "sw    $2, 32(%[sc])\n"
        "lv.s  S030, 32(%[sc])\n"
        "lv.s  S031, 0(%[bs])\n"

        // 用vh2f.s可以直接转换
        // "lhu    $25, 0(%[row])\n"  /* 读取 FP16 scale */
        // "andi   $3, $25, 0x7C00\n" /* 提取指数位 */
        // "movz   $25, $0, $3\n"    /* 指数为零时清零 */
        // "mtv    $25, S030\n"      /* 将 FP16 编码送入 VFPU */
        // "vh2f.s C020, S030\n"     /* S020 得到 FP32 scale，S021 得到 0 */
        // "lv.s   S031, 0(%[bs])\n"

        "lv.q  C100, 0(%[sc])\n"
        "lv.q  C110, 16(%[sc])\n"
        "vc2i.s C200, S100\n"
        "vc2i.s C210, S101\n"
        "vc2i.s C220, S102\n"
        "vc2i.s C230, S103\n"
        "vc2i.s C300, S110\n"
        "vc2i.s C310, S111\n"
        "vc2i.s C320, S112\n"
        "vc2i.s C330, S113\n"
        "vi2f.q C200, C200, 24\n"
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "lv.q  C400, 0(%[x])\n"
        "lv.q  C410, 16(%[x])\n"
        "lv.q  C420, 32(%[x])\n"
        "lv.q  C430, 48(%[x])\n"
        "lv.q  C500, 64(%[x])\n"
        "lv.q  C510, 80(%[x])\n"
        "lv.q  C520, 96(%[x])\n"
        "lv.q  C530, 112(%[x])\n"
        "vmul.q C600, C200, C400\n"
        "vmul.q C610, C210, C410\n"
        "vmul.q C620, C220, C420\n"
        "vmul.q C630, C230, C430\n"
        "vmul.q C700, C300, C500\n"
        "vmul.q C710, C310, C510\n"
        "vmul.q C720, C320, C520\n"
        "vmul.q C730, C330, C530\n"
        "vadd.q C600, C600, C610\n"
        "vadd.q C620, C620, C630\n"
        "vadd.q C700, C700, C710\n"
        "vadd.q C720, C720, C730\n"
        "vadd.q C600, C600, C620\n"
        "vadd.q C700, C700, C720\n"
        "vadd.q C600, C600, C700\n"
        "vscl.q C600, C600, S030\n"
        "vadd.q C000, C000, C600\n"
        "vmul.s S032, S030, S031\n"
        "vadd.s S010, S010, S032\n"
        "addiu %[row], %[row], 18\n"
        "addiu %[x], %[x], 128\n"
        "addiu %[bs], %[bs], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        "vfad.q S020, C000\n"
        "vfim.s S021, 8.0\n"
        "vmul.s S022, S010, S021\n"
        "vsub.s S020, S020, S022\n"
        "sv.s  S020, 0(%[out])\n"
        ".set pop\n"
        : [row] "+r"(row_pointer), [x] "+r"(input_pointer),
          [bs] "+r"(sum_pointer), [n] "+r"(remaining_blocks)
        : [sc] "r"(scratch), [out] "r"(&result)
        : "$2", "$3", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15",
          "$24", "$25", "memory");
    return result;
}
#endif
