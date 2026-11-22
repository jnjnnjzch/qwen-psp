# PSP Q4 矩阵 benchmark

将 `build/DOTQ4BENCH.iso` 复制到 PSP 记忆棒 `/ISO/`，从游戏菜单启动。构建目标为 PSP-1000、5.50 Prome-4，ISO 和 PBP 的最低系统版本声明均为 5.50。也可以将 `build/PSP/GAME/DOTQ4BENCH` 整个目录复制到记忆棒 `/PSP/GAME/`，启动其中的 `EBOOT.PBP`。计算数据已经包含在这两种安装包中。

## 五个矩阵入口

| 界面名称 | 实现位置 | 编译方式 |
| --- | --- | --- |
| C / SCALAR | `src/scalar.c` | 完整矩阵入口内部使用标量点积，GCC `-O3`，关闭自动向量化 |
| LLMPSP / VFPU | `src/engine.c` 与 `../LLMPSP/src/falcon_q4_psp.c` | 完整矩阵入口计算 block sums，再循环调用原项目的 `falcon_dot_q4_vfpu` |
| VH2F / VFPU | `src/engine.c` 与 `src/dot_vh2f.c` | 保留原始点积的运算顺序，使用 `lhu`、`mtv`、`vh2f.s` 转换 scale |
| CUSTOM / VMMUL | `src/custom_vfpu.c`，`BENCH_CUSTOM_VH2F=0` | 使用整数位构造转换 scale，`vmmul.q` 累计四行权重、四个输入对应的 4×4 输出小块 |
| VH2F / VMMUL | `src/custom_vfpu.c`，`BENCH_CUSTOM_VH2F=1` | 使用 `vh2f.p` 一次转换四个 scale，矩阵遍历与计算代码和 CUSTOM / VMMUL 共用 |

五个入口均接收 `BenchTask` 中的完整矩阵、行跨度、输出与工作空间，计算相同的 `Y[B,N] = X[B,K] @ W[N,K].T`。每个实现自行安排矩阵遍历和数据准备。构建检查普通 C 目标文件中 VFPU 指令数量为零，并逐个检查其余四个实现的转换与计算指令。两组对应实现保留相同的 VFPU 运算指令顺序，差异限定在 scale 转换；两种 VMMUL 中的标量浮点运算指令数量均为零。程序在单个启用 VFPU 的用户线程上执行。

## 操作

| 按钮 | 操作 |
| --- | --- |
| L / R | 切换 WORKLOADS、SETTINGS、RESULTS |
| 方向键上下 | 选择测试项目或设置项目 |
| 方向键左右 | 修改 SETTINGS 中选中的参数 |
| 交叉按钮 | 执行当前项目的五个算子比较 |
| START | 执行 SETTINGS 中选择的 QUICK 或 FULL 集合 |
| 圆圈按钮 | 取消当前比较，在当前矩阵运算完成后响应 |
| 三角按钮 | 导出已经完成的结果 |
| 方块按钮 | 保存当前实际画面为 BMP |
| SELECT / HOME | 退出程序 |

默认每个算子采集 3 个样本，每个样本累计计算 250 ms，CPU 为 333 MHz。设置提供 TIME 和 COUNT 两种停止条件，时间可选 100、250、500、1000、2000 ms，执行次数可选 1、10、100、1000、10000，样本数量可选 1、3、5、7、9，CPU 可选 222、266、333 MHz。大矩阵单次运算可能超过目标时间，CSV 保存实际执行时间和次数。

START SUITE 默认选择 QUICK，包含 13 个小规模或点积项目；FULL 包含全部 21 项。选择任意单项后按交叉按钮均可执行，包括较大矩阵项目。默认选中的首项为 `4×16×32` GEMM。

每个项目完成后自动保存 CSV，文件位于 `ms0:/PSP/SAVEDATA/DOTQ4BENCH/`。同一个项目重新执行后，界面显示此次比较结果；每次自动导出的 CSV 都保存完成时的结果。截图保存为该目录下的 `screen-000.bmp` 等文件。

## 权重、输入与测试规模

数据文件为 3,692,192 字节，约 3.52 MiB，包含 Qwen3-0.6B 第一层的 `blk.0.attn_q.weight` 和 `blk.0.ffn_down.weight`、16 个实际文本 token 的激活，以及参考输出和误差范围。切片直接引用这些常驻数据，保留来源矩阵的物理行跨度。

Attention 矩阵保留来源 GGUF 的 Q4_0 字节。FFN down 在来源 GGUF 中为 Q4_1，使用 LLMPSP 现有量化器转换为 Q4_0；转换后的字节与已有 FHQ4 模型一致。准备工具核对两个模型文件的 SHA-256，并重新核对提取或转换后的字节。

Attention 输入来自 token embedding 经过第一层 RMSNorm 的输出。FFN down 输入来自同一段文本经过第一层 causal GQA、Q/K RMSNorm、RoPE、attention output、残差、FFN RMSNorm 和 SiLU gate/up 的输出。这些激活使用现有 FHQ4 模型的量化权重，在电脑上执行 FP32 计算后保存。来源版本、原始文本、token IDs、权重和数据 SHA-256 保存在安装包中的 `MANIFEST.json`。

| 项目 | B | N | K |
| --- | --- | --- | --- |
| QUICK attention GEMM | 4 | 16 | 32 |
| QUICK attention GEMV | 1 | 16 | 128 |
| QUICK attention GEMM | 4 | 32 | 128 |
| QUICK attention GEMM | 8 | 64 | 256 |
| QUICK FFN down GEMV | 1 | 16 | 256 |
| QUICK FFN down GEMM | 4 | 32 | 256 |
| QUICK attention slice，起始行 5、起始列 32 | 3 | 7 | 96 |
| QUICK FFN down slice，起始行 11、起始列 128 | 2 | 5 | 160 |
| DOT | 1 | 1 | 32、128、512、1024、3072 |
| attention GEMV | 1 | 128、512、2048 | 1024 |
| FFN down GEMV | 1 | 1024 | 3072 |
| attention GEMM | 4 | 512 | 1024 |
| attention GEMM | 16 | 2048 | 1024 |
| FFN down GEMM | 4、16 | 1024 | 3072 |

`cases.json` 中的 `rows`、`columns`、`batch` 分别指定 N、K、B，`row_start` 和 `column_start` 指定真实切片起点。列起点与 K 均须为 32 的倍数，以保留完整 Q4 block。`full: true` 将项目归入较大矩阵集合。修改配置并重新构建后，准备工具会为实际切片重新生成 FP64 参考结果。

两种 VMMUL 在一段汇编中处理四行权重和四个输入的完整 K。整数 `and/srl` 分离 Q4 高低四位，`mtv` 将位模式送入 VFPU，`vc2i.s` 和 `vi2f.q` 完成量化整数转换，`vsub.q` 与 `vscl.q` 计算 `d*(q-8)`。CUSTOM / VMMUL 使用原始 dot_q4 的整数算法构造 FP32 scale 位模式，通过 `mtv` 传送到 VFPU；VH2F / VMMUL 将四个原有 FP16 scale 打包后使用 `vh2f.p` 转换为 FP32。权重的浮点数值保存在 VFPU 寄存器中，输入通过 `lv.q` 直接从来源矩阵读取。

每个 K=32 小块执行八条 `vmmul.q`。16 个累计值跨完整 K 保留在 VFPU 寄存器中，计算结束后写出 64 字节输出小块。程序仅为这个最终结果保留一个 64 字节临时数组；编译器的局部存储用量记录在 `build/build-report.json`。不足四行或四个输入时，使用合法行或输入的重复指针填充无效分量，仅写出切片范围内的结果。GEMV 和 DOT 也由同一矩阵入口处理。

VFPU 指令语义和硬件周期资料见 [pspdev/vfpu-docs](https://github.com/pspdev/vfpu-docs)。`vh2f.p` 的独立指令吞吐约为一周期、结果延迟约为五周期，完整 scale 转换序列仍包含读取、打包和寄存器传送成本。真机效率通过实际矩阵任务的计时结果评估。

## 计时和结果含义

启动时一次读取整个数据文件，随后关闭文件。每次比较先检查全部输出，通过之后进行预热、批次数量校准和计时。所有项目统一测量完整矩阵入口，包含实现所需的 Q4 解码、缩放、输入准备、block sums、数据排列、计算和输出写入。COUNT 的一次运算表示完成整个矩阵任务。

每个样本由连续计算批次组成，程序累计批次内部时间。界面绘制、控制器读取、文件读取、正确性检查和 CSV 写入在测量窗口之外进行。不同样本循环改变实现的执行顺序。数据在 RAM 与 CPU 缓存之间的访问属于实际计算成本。五个实现反复使用相同的真实输入和 Q4 权重；custom 仅解码当前小块。

界面柱状图显示每次完整矩阵任务耗时的样本中位数，长度越短表示耗时越少。相对倍数为普通 C 耗时除以对应实现耗时。有效 GFLOP/s 使用 `2*B*N*K / time`，计数依据实际输出需要的乘加项。CSV 保存矩阵形状、行列切片起点、QUICK/FULL 分类、`matrix_q4` 接口名称、实际运行次数、累计时间、中位数、最短值、最长值、误差、数据 CRC32，以及请求与实际 CPU/总线频率。

CSV 的 `implementation_id` 标识实现版本，当前为 `five-kernels-1`。对同一项目计算 `T(LLMPSP / VFPU) / T(VH2F / VFPU)`，可以观察点积中 scale 转换的影响；计算 `T(CUSTOM / VMMUL) / T(VH2F / VMMUL)`，可以观察 VMMUL 中 scale 转换的影响。比值大于 1 表示对应的 VH2F 实现耗时更少。五种实现之间的绝对耗时和相对 C 的倍数同时保存在 CSV 中。

`vh2f` 减少了 scale 转换所需的指令，完整任务中的收益还取决于这部分成本的占比、指令依赖和矩阵规模。GEMV 的输入数量为 1，当前 VMMUL 会计算重复输入对应的列；GEMM 可以使用四个不同的输入。各实现的速度次序通过 PSP 实机计时确定。为了让五种实现分别占据一次样本起始位置，可以将 SAMPLES 设置为 5。

参考输出使用 GGUF 库解量化 Q4_0，执行 FP64 矩阵乘法，保存为 FP32。每个输出要求 `abs(error) <= 2e-5 + 2e-5 * sum(abs(x_i*w_i))`。发生数值错误时，程序显示错误并终止测试。取消的比较不产生计时结果。选中组件的 FP16 scale 没有非规格化值，符合原项目 VFPU 内核的 scale 转换范围。

## 编译

按仓库根目录的说明准备 GGUF、FHQ4 和官方 tokenizer。安装依赖并构建。 [模型准备](../MODEL_SETUP.md)

```bash
cd psp-dot-q4-bench
mkdir -p .work/tmp .work/python
TMPDIR="$PWD/.work/tmp" python -m pip install --target .work/python -r requirements.txt
bash build.sh
```

PSP 交叉编译在固定镜像 `ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb1d4a9cdd93718fb8bf652ec7bb48e54038a8b` 中执行，编译器为 GCC 15.2.0，SDK 路径为 `/usr/local/pspdev/psp/sdk`。容器中的项目路径为 `/workspace/psp-dot-q4-bench`，LLMPSP 来源目录通过只读挂载提供。编译容器禁用网络。

数据准备使用相邻目录的以下现有文件：

- `../LLMPSP/qwen/models/qwen3-0.6b/Qwen3-0.6B-Q4_0.gguf`
- `../LLMPSP/qwen/q4_0/model.fhq4` 及其 `.json` 元数据
- `../LLMPSP/qwen/q4_0/.work/tokenizer.json`
- `../LLMPSP/qwen/download-manifest-q4_0.json`

最终输出为 `build/DOTQ4BENCH.iso` 和 `build/PSP/GAME/DOTQ4BENCH/`。pycdlib 生成基础 ISO9660 目录，文件标识符使用 PSP 固件要求的完整名称，包括 `EBOOT.BIN` 和 `BENCH.DAT`。每次构建都会检查入口、目录和数据文件名称，并通过重新读取镜像核对程序与数据。构建报告和校验和分别位于 `build/build-report.json`、`build/CHECKSUMS.sha256`。所有中间结果都保存在本项目被 gitignore 的目录中。

## 修改 custom VFPU

编辑 `src/custom_vfpu.c` 中的 `matrix_tile` 或 `CUSTOM_ENTRY`，重新执行 `bash build.sh`。构建系统将这份源码分别编译为 `bench_matmul_custom` 和 `bench_matmul_custom_vh2f`。五个矩阵入口使用相同的任务描述：

```c
void bench_matmul_custom(const BenchTask *task);
```

`task->weights` 和 `task->input` 已指向切片起点，`weight_row_bytes` 与 `input_stride` 保存来源矩阵行跨度，后者以 float 元素为单位。`rows`、`columns`、`batch` 为 N、K、B，输出按 `output[b*N+n]` 连续写入。`sums` 为至多 16×96 个 FP32 的工作空间，可由具体实现使用。

每个 Q4 block 为 18 字节，前 2 字节为 FP16 scale，后 16 字节保存整数。低 4 位对应第 0–15 个元素，高 4 位对应第 16–31 个元素，解量化值为 `scale * (quant - 8)`。custom 可使用全部 VFPU 寄存器，并须遵循 C ABI。当前 PSP 实现已启用；电脑端明确显示该平台实现不可用，host 数值验证执行普通 C，PSP 模拟器验证五个原生入口。

## 运行验证

```bash
cd psp-dot-q4-bench
TMPDIR="$PWD/.work/tmp" PYTHONPATH="$PWD/.work/python" python tools/verify.py
```

将 PPSSPP 的 headless 构建放在 `../LLMPSP/qwen/q4_0/.work/ppsspp-build/`。验证脚本运行最终 ISO 和 PBP，通过模拟控制器操作读取程序生成的截图和 CSV。验证报告为 `build/verification.json`。发布仓库提供 PPSSPP 的数值检查、计时记录和反汇编。 [验证结果](../results/README.md)

安装包同时包含 LLMPSP 的 MIT 许可证和 Qwen 模型的 Apache-2.0 许可证。
