# PSP 算子验证结果

## Q4 GEMV 与 GEMM

基准使用 Qwen3-0.6B 的 `blk.0.attn_q.weight`、`blk.0.ffn_down.weight` 和真实文本的激活。电脑端核验了 59,908 个输出。PPSSPP 的 PSP-1000 配置运行了 21 个工作负载和 5 种内核，共 105 项比较，最大绝对误差为 `9.54e-6`。CSV 的计时环境为 PPSSPP。 [完整工作负载结果](q4-suite.csv)

运行 `make test` 可重新生成完整验证报告、反汇编和计时数据。 [构建和验证](../README.md)

## 128 维注意力

`pytorch-style` 分支中的 Qwen3 推理程序在 PPSSPP 上核验了 82 次完整前向中的最后一层。全部 16 个查询头的 QK 最大绝对误差为 `5.72e-6`，AV 最大绝对误差为 `0`。三组测试的生成 token 均与电脑端参考一致。 [注意力验证报告](qwen-attention.json)
