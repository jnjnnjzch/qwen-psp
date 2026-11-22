# PSP VFPU 算子优化

用 Qwen3-0.6B 第一层的 Q4_0 权重和真实文本激活测量 GEMV 与 GEMM。五个内核接收相同的矩阵任务，计算 `Y[B,N] = X[B,K] @ W[N,K].T`。 [验证结果](results/README.md)

## 内核

| 界面名称 | 源文件 | 实现 |
| --- | --- | --- |
| C / SCALAR | `src/scalar.c` | 标量点积，编译时关闭自动向量化 |
| LLMPSP / VFPU | `src/llmpsp_vfpu.c` | 原始 LLMPSP Q4 点积，使用整数指令转换 scale |
| VH2F / VFPU | `src/dot_vh2f.c` | 保留点积累加顺序，使用 `vh2f.s` 转换 scale |
| CUSTOM / VMMUL | `src/custom_vfpu.c` | 使用 `vmmul.q` 累加 4×4 输出块，以整数指令转换 scale |
| VH2F / VMMUL | `src/custom_vfpu.c` | 使用相同的矩阵块，使用 `vh2f.p` 转换四个 scale |

`tests/cases.json` 定义 21 个矩阵形状和切片。基准先用 FP64 参考结果验证数值，再计量完成整个矩阵运算的耗时，覆盖解量化、数据准备、计算和写回。

## 准备和构建

在仓库根目录执行，需要 Python、Docker 和 DejaVu Sans 字体。 [模型来源](tools/model_source.json)

```bash
mkdir -p .work/tmp .work/python
export TMPDIR="$PWD/.work/tmp" PYTHONPATH="$PWD/.work/python"
python -m pip install --target .work/python -r requirements.txt
curl -fL -o .work/Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_0.gguf
curl -fL -o .work/tokenizer.json https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json
python tools/convert_model.py
make package
```

构建工具核对模型校验和，从第一层导出 Attention 和 FFN down 输入，生成基准数据、安装包和反汇编。将 `build/DOTQ4BENCH.iso` 放到记忆棒的 `ISO/`，或将 `build/PSP/GAME/DOTQ4BENCH/` 复制到 `PSP/GAME/`。两种安装包的最低系统版本均为 5.50。

## 操作和验证

- L 和 R 切换工作负载、设置和结果。方向键选择项目或修改设置。
- 交叉按钮执行当前项目，START 执行 QUICK 或 FULL 集合，圆圈按钮取消当前比较。
- 比较完成后自动写入 CSV。三角按钮可重新导出，方块按钮保存截图。文件保存在 `ms0:/PSP/SAVEDATA/DOTQ4BENCH/`。

在 `.work/ppsspp-build/` 放置 `PPSSPPHeadless` 后执行 `make test`。它验证电脑端标量计算、PSP 上的五种内核、ISO 与 PBP 启动以及 CSV 结果。测试日志、反汇编和完整报告生成在 `build/`。
