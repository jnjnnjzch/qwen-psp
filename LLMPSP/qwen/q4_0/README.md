# Qwen3-0.6B Q4_0 与 PSP-1000

应用目录：`build/PSP/GAME/LLMPSP-Qwen3/`。复制到存储卡的
`PSP/GAME/LLMPSP-Qwen3/` 后启动其中的 `EBOOT.PBP`。

程序使用 128 个 token 的上下文、int8 KV 缓存和 6 MiB 权重缓存。
编译设置为 `PSP_LARGE_MEMORY = 0`，配置关闭 volatile memory 和预取。
界面、按键、输入编辑和会话继续处理复用原项目。屏幕字体和键盘沿用原项目的 ASCII 支持。

模型文件为 **338,307,328 字节**，每个 token 的计算逐段读取文件中的权重。
存储卡读取速度与 CPU 计算速度都会影响实体设备的生成耗时。

## 代码入口

| 文件 | 功能 |
| --- | --- |
| `convert.py` | 使用 `gguf.GGUFReader` 读取 GGUF，复用原项目 Q4_0 量化器，生成 FHQ4 v2、TOK3 和权重位置记录 |
| `../../src/falcon_h1.c` | 加载 Qwen3 权重、维护运行状态并执行模型前向 |
| `../../src/falcon_tokenizer.c` | 使用 PCRE2 执行 Unicode 预分词，并完成 BPE 合并和词表查找 |
| `Makefile` | 编译电脑端 CLI 和数值检查工具 |
| `Makefile.psp`、`build_psp.sh` | 编译 PSP 交互应用和自动验证程序，生成存储卡应用目录 |
| `probe.c`、`verify.py` | 核验官方分词、聊天模板、完整模型输出、逐层数值和缓存一致性 |
| `autotest.c`、`verify_psp.py` | 在 PSP 程序中执行完整推理，核验生成的 token 和内存记录 |
| `psp1000.ini`、`qwen_test.txt` | PSP-1000 模拟配置与自动验证使用的真实提示文本 |

公共引擎继续负责 Q4_0 矩阵向量乘法、VFPU 点积、文件读取、权重缓存、RMSNorm、
SiLU、RoPE、int8 KV、注意力、采样和生成循环。Qwen3 的查询、键在 RoPE 之前执行
各自的 RMSNorm；注意力头维度为 128，使用公共引擎中的通用注意力循环。
PSP 的矩阵乘法使用原有 `falcon_dot_q4_vfpu`。

编译定义 `FALCON_QWEN3` 启用 PSP 界面的 Qwen3 配置。
Qwen3 构建使用 `FALCON_SCALAR_ATTENTION` 和 `FALCON_NO_PREFETCH`。
3072 列矩阵使用 96 个 Q4_0 块，注意力拼接结果分配 2048 个 FP32 元素。
会话模板使用官方 `enable_thinking=False` 对应的 assistant 前缀。

## 模型转换

下载来源和校验值保存在上级目录的 `download-manifest-q4_0.json`。
GGUF 中的 193 个 Q4_0 矩阵按原字节保存；Q6_K 词嵌入与三个 Q4_1 矩阵转换为 Q4_0。
113 个归一化张量保留 FP32。转换记录 `model.fhq4.json` 包含每个张量的位置、形状、
原始量化类型、输出类型及整个文件的 SHA-256。

按仓库根目录的说明准备模型，在本目录构建并运行电脑端 CLI。 [模型准备](../../../MODEL_SETUP.md)

```bash
make host
./build/qwen_cli model.fhq4 "What is 2+2?" 8 6
```

电脑端 C 编译需要 C99 编译器及 PCRE2 8 位开发文件。
Python 依赖见 `requirements.txt`。

## PSP 构建

构建依赖保存在本目录的 `.work/`。PCRE2 使用 10.49，源码目录为
`.work/pcre2-10.49/`。`pcre2-psp.patch` 记录 PSP 平台使用的两个局部变量声明：
`read_repeat_counts` 的 `min`、`max` 使用 `int`，与 `read_number` 的输出参数类型相同。
静态库以 `-G0` 编译，并启用 Unicode。

准备 PCRE2 源码后，从本目录使用 PSPDEV 容器构建并打包。 [PSP 依赖](../../../PSP_SETUP.md)

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$(cd ../.. && pwd):/work" \
  -w /work/qwen/q4_0 \
  -e TMPDIR=/work/qwen/q4_0/.work/tmp \
  ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb1d4a9cdd93718fb8bf652ec7bb48e54038a8b \
  sh build_psp.sh
```

交互程序位于 `build/PSP/GAME/LLMPSP-Qwen3/`。
自动验证程序位于 `build/Qwen3Test/`，从该目录读取 `qwen_test.txt`：
第一行是缓存 MiB 和生成 token 上限，第二行是提示文本。

## 电脑端核验

```bash
export TMPDIR="$PWD/.work/tmp"
python -m pip install -r requirements.txt
python verify.py --reference
```

核验使用完整模型，以及官方 tokenizer 和 Hugging Face `Qwen3ForCausalLM`。
参考实现加载转换后的权重，并使用相同的逐头 int8 KV 量化。
验证包括十种实际输入的分词、两个聊天模板、完整词表输出分布，以及生成 token 序列。
缓存为 0 和 6 MiB 时，电脑端的以下回答一致，并且与参考实现的生成 token 一致：

| 提示 | 回答 |
| --- | --- |
| `What is 2+2?` | `2 + 2 equals 4.` |
| `你好，请用中文回答：1加1等于多少？` | `1加1等于2。` |

`Hello world` 的输出分布 KL 为约 0.000355，最大 logit 绝对差为约 0.07449。
逐层记录显示，第二个输入 token 的第 5 层投影最大差约 0.0000801，导致一个
int8 value 缓存元素产生不同舍入。生成序列继续按 token 核验。
完整记录位于 `build/host-verification.json`。

## PSP-1000 模拟核验

```bash
python verify_psp.py
```

脚本使用 `psp1000.ini` 设置 `PSPModel = 0`，关闭忽略非法内存访问，运行真实 PSP
程序的 VFPU 点积和完整模型生成。它逐个比较实际输出 token 与电脑端参考记录，
并要求模型常驻表、权重缓存和运行数组的合计容量低于 20 MiB。
程序代码、线程、分词临时空间由剩余用户内存承担。

模拟器构建位于 `.work/ppsspp-build/`，源码版本为
`1a7c92107ddd49bb61edd7c36d7232fa98d227ec`，使用官方 headless 构建入口。
本机编译的 `Common/Profiler/Profiler.h` 显式引入 `<cstddef>`。
模拟器的图形开发文件位于 `.work/host-deps/`。
完整运行记录写入 `build/psp-verification.json` 和 `build/psp-*-cache*.log`。
模拟运行耗时记录使用模拟器时钟；实体设备速度需要在设备上测量。

完整模型的生成结果与常驻运行数据如下。

| PSP 权重缓存 | 提示 | 生成结果 | 常驻运行数据 |
| --- | --- | --- | ---: |
| 0 MiB | `What is 2+2?` | `2 + 2 equals 4.` | 10.88 MiB |
| 6 MiB | `你好，请用中文回答：1加1等于多少？` | `1加1等于2。` | 16.88 MiB |
| 6 MiB | `What is 2+2?` | `2 + 2 equals 4.` | 16.88 MiB |

三次运行的生成 token 均与电脑端参考记录一致。
应用目录的 `CHECKSUMS.sha256` 核验程序、模型和配置文件。

程序使用项目的 MIT 许可证；模型许可证和来源保存在 `MODEL-LICENSE.txt` 与
`MODEL_NOTICE.txt`。应用目录同时包含 PCRE2 许可证。
