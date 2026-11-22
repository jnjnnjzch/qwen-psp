# 查找文件

学习模型结构时，从 `src/model/qwen3.c` 阅读到 `src/layers/decoder_layer.c` 和
`src/layers/mlp.c`。学习分词时，从 `app/tokenizer_demo.c` 阅读到 `src/tokenizer/tokenizer.c`
和 `src/tokenizer/bpe.c`。其余文件可以在需要时查找。

## 应用入口与公开声明

| 文件 | 职责 |
| --- | --- |
| `app/main.c` | 电脑端入口：打开模型、编码用户消息、生成回答、释放资源 |
| `app/tokenizer_demo.c` | 打印分段、初始字节条目、每轮合并、最终编号和还原文本 |
| `include/qwen.h` | 应用需要引入的公开声明入口 |
| `include/qwen_types.h` | 模型配置、常量、基础类型和回调类型 |
| `include/qwen_layers.h` | Linear、Embedding、RMSNorm、Attention、MLP 和 DecoderLayer 的结构与接口 |
| `include/qwen_model.h` | 完整模型、模型打开和关闭、权重缓存与 forward 接口 |
| `include/qwen_tokenizer.h` | 词表结构、编码、按编号读取字节、聊天提示和观察回调 |
| `include/qwen_runtime.h` | 中间数组、KV 数据、统计和运行空间的生命周期 |
| `include/qwen_generation.h` | 采样器、生成循环与继续会话接口 |

## 模型组件与底层运算

| 文件 | 职责 |
| --- | --- |
| `src/model/qwen3.c` | Embedding → 28 个 DecoderLayer → RMSNorm → LM head |
| `src/layers/decoder_layer.c` | 组合 Attention、MLP、两次归一化与两条残差连接 |
| `src/layers/mlp.c` | gate/up 投影、SiLU、逐元素乘法、down 投影 |
| `src/layers/self_attention.c` | Q/K/V 投影、Q/K 归一化、RoPE、读取历史 KV、注意力与输出投影 |
| `src/layers/basic.c` | Embedding 按编号读取一行权重；Linear 按行执行点积；RMSNorm 调用归一化运算 |
| `src/ops/ops.h` | 各个底层运算的内部声明 |
| `src/ops/q4.c` | Q4_0 权重解码、标量点积和平台点积选择 |
| `src/ops/q4_vfpu.c` | PSP VFPU 量化点积实现 |
| `src/ops/attention_vfpu.c` | PSP VFPU 的完整 128 维注意力点积与 V 加权求和 |
| `src/ops/math.c` | RMSNorm、SiLU、Softmax、RoPE、残差相加 |
| `src/ops/kv_cache.c` | K/V 的 int8 编码、scale、查询与历史键的点积、历史值的加权求和 |

## 分词、存储与运行管理

| 文件 | 职责 |
| --- | --- |
| `src/tokenizer/tokenizer.c` | 识别特殊条目，使用 PCRE2 划分普通文本，再调用 BPE |
| `src/tokenizer/bpe.c` | 一段文本中的初始字节查找、优先级选择和反复合并 |
| `src/tokenizer/vocabulary.c` | 读取 TOK3 词表，按字节查编号、按编号查字节和 rank |
| `src/tokenizer/chat_template.c` | 用户消息和 assistant 前缀，关闭思考模式的聊天模板 |
| `src/tokenizer/tokenizer_internal.h` | 分词模块内部共享的函数声明 |
| `src/storage/model_loader.c` | 读取 FHQ4 v2 配置，为模型组件填写权重位置，加载归一化参数和词表 |
| `src/storage/weights.c` | 文件读取、分块缓存、缓存统计和释放 |
| `src/storage/storage.h` | 权重读取模块内部接口 |
| `src/storage/binary.c` | 自定义 FHQ4 文件的整数读取与连续区域分配 |
| `src/storage/binary.h` | 对应内部声明 |
| `src/runtime/memory.c` | 分配、初始化和释放中间数组、KV 数据与 RoPE 表 |
| `src/runtime/sampler.c` | 重复惩罚、温度、top-k、top-p 和 token 选择 |
| `src/runtime/generate.c` | 提示处理、反复 forward、输出回调、停止条件和 pending_token |

## PSP、构建与验证

| 文件 | 职责 |
| --- | --- |
| `compat/falcon_h1.h` | 向原 PSP 界面提供需要的类型与函数声明 |
| `compat/falcon_frontend.c` | 将界面的函数调用交给对应 Qwen 接口 |
| `sources.mk` | 本目录独立编译的核心 `.c` 文件列表 |
| `Makefile` | 构建电脑端应用、分词演示和验证程序 |
| `Makefile.psp` | 使用 PSPDEV 构建应用与自动运行测试 |
| `build_psp.sh` | 构建两个 PSP 程序，准备安装目录和校验文件 |
| `llmpsp_config.cfg` | PSP 默认配置，包含 6 MiB 权重缓存和 128 个上下文位置 |
| `psp1000.ini` | PPSSPP 的 PSP-1000 配置 |
| `tests/probe.c` | 输出 token 编号、聊天提示，以及逐层向量和完整词表分数 |
| `tests/multiturn.c` | 用真实对话验证继续会话和完整重放一致 |
| `tests/psp_autotest.c` | 在 PSP 上自动加载模型、生成，比较真实 Q/K/V 的 VFPU 与标量结果，报告 token 编号与内存 |
| `tests/qwen_test.txt` | 自动运行程序的缓存、生成长度和真实提示配置 |
| `tests/verify_host.py` | 与官方 tokenizer、参考输出和原实现的逐层数据进行比较 |
| `tests/verify_psp.py` | 启动 PPSSPP，并比较 PSP 生成编号与电脑端参考结果 |
| `.gitignore` | 排除构建产物、中间文件、共享模型和 Python 缓存 |
| `LICENSE` | C 实现的 MIT 许可证 |

`build/` 保存可执行文件和验证报告；`.work/tmp/` 用于构建中间文件。
`models/` 中的链接指向相邻项目已有的模型和转换说明。

## 学习文档

| 文件 | 内容 |
| --- | --- |
| `START_HERE.md` | 实际演示与阅读顺序 |
| `README.md` | 运行、构建、验证结果和资源来源 |
| `docs/00-c-basics.md` | 指针、`->`、地址、数组和函数参数 |
| `docs/01-tokenizer.md` | 预分词、BPE、真实合并过程和聊天模板 |
| `docs/02-model.md` | PyTorch 构件对应关系、模型与 DecoderLayer 流程图 |
| `docs/03-runtime.md` | 权重读取、算子边界、内存和继续会话 |
| `docs/04-files.md` | 本页，按职责查找文件 |
