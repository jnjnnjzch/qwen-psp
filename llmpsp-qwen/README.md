# 运行与构建

学习代码从 [START_HERE.md](START_HERE.md) 开始。它先演示真实词表中的 BPE 合并，
再按照 C 符号、分词、模型组件、底层运算的顺序介绍代码。

## 电脑端

按仓库根目录的说明准备模型。在本目录执行以下命令，需要 C99 编译器、Make、PCRE2-8 开发库和系统数学库。 [模型准备](../MODEL_SETUP.md)

```bash
make -j4
mkdir -p models
ln -s ../../LLMPSP/qwen/q4_0/model.fhq4 models/qwen3-0.6b-q4_0.fhq4
./build/tokenizer_demo models/qwen3-0.6b-q4_0.fhq4 "Hello world"
./build/qwen_cli models/qwen3-0.6b-q4_0.fhq4 "What is 2+2?" 8 6
```

最后一条命令中的 `8` 是最多生成的 token 数量，`6` 是权重缓存的 MiB 数量。
当前默认使用贪心选择，回答为 `2 + 2 equals 4.`。

模型链接指向相邻目录 `../LLMPSP/qwen/q4_0/model.fhq4`。这是已转换的
Qwen3-0.6B Q4_0，文件大小为 338307328 字节，配置为 128 个上下文位置。

## PSP 应用

已构建的应用目录为 `build/PSP/GAME/LLMPSP-Qwen3/`。
将整个 `LLMPSP-Qwen3` 文件夹复制到记忆棒的 `PSP/GAME/` 下即可安装。
其中包含 `EBOOT.PBP`、`model.fhq4`、配置、许可证和校验文件。

准备 PCRE2 源码后，在本目录运行 PSPDEV 容器。构建脚本会生成 PCRE2 静态库和 PSP 应用。 [PSP 依赖](../PSP_SETUP.md)

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$(dirname "$PWD"):/work" \
  -w /work/llmpsp-qwen \
  -e TMPDIR=/work/llmpsp-qwen/.work/tmp \
  ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb1d4a9cdd93718fb8bf652ec7bb48e54038a8b \
  sh build_psp.sh
```

模型、分词、层、算子、文件读取和生成的实现都位于本目录，并逐个编译成目标文件。
PSP 界面、按钮处理、配置读取与图标复用相邻项目。`compat/` 提供界面调用所需的函数声明和转接函数。
`Makefile.psp` 将原有界面源文件与本目录实现分别编译，再进行链接。

## 真实模型验证

先准备官方 tokenizer，构建本项目及 `LLMPSP` 中的参考程序，再运行电脑端核验。PSP 验证还需要完成上面的 PSP 构建和 PPSSPP 准备。

```bash
export TMPDIR="$PWD/.work/tmp"
python -m pip install -r ../LLMPSP/qwen/q4_0/requirements.txt
make -C ../LLMPSP/qwen/q4_0 -j4 host
(cd ../LLMPSP/qwen/q4_0 && python verify.py --reference)
python tests/verify_host.py
python tests/verify_psp.py
```

| 验证内容 | 实际结果 |
| --- | --- |
| 10 段中英文、空白、消息标记和其他 Unicode 文本 | token 编号与官方 tokenizer 一致 |
| 两个聊天提示的模板 | 与官方 `enable_thinking=False` 模板一致 |
| `Hello world` 的 28 层输出与完整词表分数 | 与原实现逐个二进制位一致 |
| 英文、中文生成，分别使用 0 MiB 和 6 MiB 缓存 | 与此前真实模型参考结果一致 |
| 继续会话与完整对话重放 | 后续生成的 8 个 token 一致 |
| PSP 的 128 维 VFPU Attention | 82 次真实 forward 的最后一层，全部 16 个 Q 头通过；QK 最大绝对差约 5.72e-6，AV 最大绝对差为 0 |
| PPSSPP 的 PSP-1000，32 MB 配置 | 英文、中文生成的 token 编号与电脑端一致 |
| PSP 界面启动 | 日志到达 `UI ready` |

完整报告保存在 `build/host-verification.json` 和 `build/psp-verification.json`。
使用 6 MiB 权重缓存时，PSP 测试统计的模型、缓存与运行缓冲区共 17694992 字节，约 16.88 MiB；
可执行程序、线程和临时分词空间还需要额外内存。运行验证使用 PPSSPP 的 PSP-1000 配置。

## 代码来源与资源

C 实现整理自相邻 `LLMPSP` 的模型计算、Qwen 加载、分词和生成代码，保留 MIT 许可证。
Q4 点积、VFPU 指令、int8 KV 缓存、权重读取和采样运算沿用已有计算方式。
`src/ops/attention_vfpu.c` 将 Attention 算子扩展为完整的 128 维接口：
QK 一次调用计算完整点积，AV 一次调用写入完整输出；分组加载和累加由算子内部完成。
`src/layers/` 以组件接口组织这些运算，`src/model/qwen3.c` 组合完整模型。

当前构建依赖以下已有资源：

| 相邻项目中的位置 | 用途 |
| --- | --- |
| `qwen/q4_0/model.fhq4` | 真实量化权重和内嵌词表 |
| `qwen/q4_0/.work/psp-deps/` | PSP PCRE2 头文件和静态库 |
| `src/main_psp.c`、`src/falcon_ui.c`、`src/falcon_prefetch_psp.c` | PSP 应用入口、界面和平台接口 |
| `include/`、`ICON0.PNG` | 界面声明和应用图标 |
| `qwen/q4_0/.work/`、`qwen/q4_0/build/` | 官方 tokenizer、PPSSPP 和本地核验生成的参考结果 |
| `qwen/q4_0/MODEL-LICENSE.txt`、`MODEL_NOTICE.txt` | 随应用分发的模型许可证和说明 |

PCRE2 执行官方预分词正则表达式。验证脚本通过 Transformers 读取官方 tokenizer，
通过 NumPy 比较每层结果。详细文件入口见[文件职责](docs/04-files.md)。
