# Qwen3-0.6B on PSP

纯 C 推理程序，在 PSP-1000 的内存限制下运行 Q4_0 量化的 Qwen3-0.6B。模型按 PyTorch 的组件划分组织为 Embedding、DecoderLayer、Attention、MLP 和 RMSNorm。PSP 版本使用 VFPU 执行 Q4 点积与 128 维注意力。

- [准备模型](MODEL_SETUP.md)
- [电脑端和 PSP 构建](llmpsp-qwen/README.md)
- [从分词和模型组件开始阅读代码](llmpsp-qwen/START_HERE.md)

`original-c` 分支提供沿用 LLMPSP 结构的 C 版本。`operator-optimization` 分支包含矩阵算子基准和验证结果。
