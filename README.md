# PSP VFPU 算子优化

基准使用 Qwen3-0.6B 第一层的 Q4_0 权重和真实文本产生的激活，运行五种 GEMV 和 GEMM 内核。源码包含标量 C、原始 LLMPSP VFPU、使用 `vh2f.s` 的 VFPU，以及两种使用 `vmmul.q` 的内核。

- [模型和数据准备](MODEL_SETUP.md)
- [算子源码、构建和运行](psp-dot-q4-bench/README.md)
- [数值验证、计时 CSV 和反汇编](results/README.md)

`pytorch-style` 分支包含 128 维 VFPU 注意力的代码。`original-c` 分支包含沿用 LLMPSP 结构的 C 推理引擎。
