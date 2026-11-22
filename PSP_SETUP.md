# 准备 PSP 构建依赖

PSP 程序使用 PSPDEV 容器中的交叉编译器。下载 PCRE2 10.49 源码后，构建脚本会编译静态库并将许可证随应用打包。

从仓库根目录执行以下命令。`patch` 将 `read_repeat_counts` 中传给 `read_number` 的两个变量改为 PSP 编译器要求的 `int` 类型。

```bash
cd LLMPSP/qwen/q4_0
mkdir -p .work
curl -fL -o .work/pcre2-10.49.tar.gz \
  https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.49/pcre2-10.49.tar.gz
tar -xzf .work/pcre2-10.49.tar.gz -C .work
patch -d .work/pcre2-10.49 -p1 < pcre2-psp.patch
```

模拟验证使用启用了 headless 目标的 PPSSPP。将 `PPSSPPHeadless` 放在 `LLMPSP/qwen/q4_0/.work/ppsspp-build/`，或在该位置创建指向已有构建目录的链接。
