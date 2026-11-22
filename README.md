# Qwen3-0.6B on PSP

在 PSP-1000 上运行 Q4_0 量化的 Qwen3-0.6B。推理代码按 PyTorch 的组件划分组织为 Embedding、DecoderLayer、Attention、MLP 和 RMSNorm，使用 C 实现。PSP 使用 VFPU 执行 Q4 点积和 128 维注意力。 [代码阅读入口](docs/README.md)

## 安装

构建后，将 `build/PSP/GAME/LLMPSP-Qwen3/` 复制到记忆棒的 `PSP/GAME/`，从游戏菜单启动。默认使用 128 个上下文位置、int8 KV 缓存和 6 MiB 权重缓存。

## 准备模型

在仓库根目录执行。转换程序生成 `model.fhq4` 和张量位置记录 `model.fhq4.json`。

```bash
mkdir -p .work/tmp
export TMPDIR="$PWD/.work/tmp"
python -m pip install gguf==0.19.0 'numpy>=2.0'
curl -fL -o .work/Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_0.gguf
printf '%s  %s\n' 33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb \
  .work/Qwen3-0.6B-Q4_0.gguf | sha256sum -c -
python tools/convert_model.py
```

## 构建

电脑端需要 C99 编译器、Make 和 PCRE2 8 位开发库。

```bash
make -j4 host
./build/qwen_cli model.fhq4 "What is 2+2?" 8 6
```

PSP 构建使用 PSPDEV 和 PCRE2 10.49。准备 PCRE2 源码后，在 PSPDEV 容器中执行 `make package psp-test`，生成安装包和模拟器测试程序。

```bash
curl -fL -o .work/pcre2-10.49.tar.gz \
  https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.49/pcre2-10.49.tar.gz
tar -xzf .work/pcre2-10.49.tar.gz -C .work
patch -d .work/pcre2-10.49 -p1 < tools/pcre2-psp.patch
PSP_IMAGE=ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb8bf652ec7bb48e54038a8b
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace "$PSP_IMAGE" make package psp-test
```

## 验证

将 PPSSPP 的 headless 构建放在 `.work/ppsspp-build/`。核验使用官方 tokenizer 和加载同一份量化权重的 `Qwen3ForCausalLM`，比较分词、完整词表分数和生成 token。PSP 自动测试还比较 VFPU 与标量注意力的结果。

```bash
python -m pip install -r requirements.txt
curl -fL -o .work/tokenizer.json https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json
curl -fL -o .work/tokenizer_config.json https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json
python tests/verify_host.py --reference
python tests/verify_psp.py
```

## 来源与许可

C 源码及 PSP 界面基于 LLMPSP，使用 MIT 许可证。模型使用 Apache License 2.0。 [代码许可证](LICENSE) · [模型来源](MODEL_NOTICE.txt)
