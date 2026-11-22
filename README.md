# Qwen3-0.6B on PSP · 原始 C 结构

沿用 LLMPSP C 推理引擎结构，在 PSP-1000 上运行 Q4_0 量化的 Qwen3-0.6B。模型加载、权重缓存和完整前向计算位于 `LLMPSP/src/falcon_h1.c`，分词位于 `LLMPSP/src/falcon_tokenizer.c`，Q4 点积由 PSP VFPU 执行。

- [准备模型](MODEL_SETUP.md)
- [电脑端和 PSP 构建](LLMPSP/qwen/q4_0/README.md)

`pytorch-style` 分支提供按模型组件组织的 C 版本。`operator-optimization` 分支包含矩阵算子基准和验证结果。
