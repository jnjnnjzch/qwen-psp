# PSP 算子验证结果

## Q4 GEMV 与 GEMM

基准使用 Qwen3-0.6B 的 `blk.0.attn_q.weight`、`blk.0.ffn_down.weight` 和真实文本的激活。电脑端核验了 59,908 个输出。PPSSPP 的 PSP-1000 配置运行了 21 个工作负载和 5 种内核，共 105 项比较，最大绝对误差为 `9.54e-6`。CSV 的计时环境为 PPSSPP。 [验证报告](ppsspp/q4-verification.json)

- [21 个工作负载](ppsspp/q4-suite.csv)
- [快速测试](ppsspp/q4-quick.csv)
- [固定次数测量，222 MHz](ppsspp/q4-count-222mhz.csv)
- [固定次数测量，266 MHz](ppsspp/q4-count-266mhz.csv)

| 内核 | 反汇编 |
| --- | --- |
| C / SCALAR | [标量 C](ppsspp/scalar-disassembly.txt) |
| LLMPSP / VFPU | [原始 VFPU](ppsspp/vfpu-disassembly.txt) |
| VH2F / VFPU | [`vh2f.s` 转换](ppsspp/dot-vh2f-disassembly.txt) |
| CUSTOM / VMMUL | [`vmmul.q` 矩阵分块](ppsspp/custom-disassembly.txt) |
| VH2F / VMMUL | [`vh2f.p` 与 `vmmul.q`](ppsspp/custom-vh2f-disassembly.txt) |

## 128 维注意力

`pytorch-style` 分支中的 Qwen3 推理程序在 PPSSPP 上核验了 82 次完整前向中的最后一层。全部 16 个查询头的 QK 最大绝对误差为 `5.72e-6`，AV 最大绝对误差为 `0`。三个生成场景的 token 均与电脑端参考结果一致。 [注意力验证报告](ppsspp/qwen-attention-verification.json)
