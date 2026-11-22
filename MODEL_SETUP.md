# 准备 Qwen3-0.6B 权重

在仓库根目录执行以下命令。模型来源、固定版本和原始文件校验值记录在 [下载清单](LLMPSP/qwen/download-manifest-q4_0.json) 中。

```bash
cd LLMPSP/qwen/q4_0
mkdir -p .work/tmp ../models/qwen3-0.6b
export TMPDIR="$PWD/.work/tmp"
python -m pip install gguf==0.19.0 'numpy>=2.0'
curl -fL -o ../models/qwen3-0.6b/Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_0.gguf
printf '%s  %s\n' 33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb \
  ../models/qwen3-0.6b/Qwen3-0.6B-Q4_0.gguf | sha256sum -c -
python convert.py
```

`convert.py` 生成 `model.fhq4` 和 `model.fhq4.json`。FHQ4 文件大小为 338,307,328 字节，SHA-256 为 `23740505a170b18257212aebfd6cec84b1c3dc727b7267f7341c49fbe69a0502`。

模型验证和矩阵基准需要官方 tokenizer。继续在 `LLMPSP/qwen/q4_0/` 执行以下命令。

```bash
curl -fL -o .work/tokenizer.json \
  https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json
curl -fL -o .work/tokenizer_config.json \
  https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json
```

模型使用 Apache License 2.0。 [模型许可证](LLMPSP/qwen/q4_0/MODEL-LICENSE.txt)
