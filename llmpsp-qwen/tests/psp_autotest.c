#include "qwen.h"
#include "ops/ops.h"

#include <math.h>
#include <pspkernel.h>
#include <psppower.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

PSP_MODULE_INFO("Qwen3Test", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-2048);

static FILE *log_file;
typedef struct {
    QwenRuntime *runtime;
    int query_heads, forwards;
    float score_max_abs, output_max_abs;
} AttentionCheck;
static void log_line(const char *format, ...) {
    char line[1024];
    va_list arguments;
    int length;
    va_start(arguments, format);
    length = vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (length <= 0 || length >= (int)sizeof(line)) exit(1);
    sceIoWrite(1, line, (unsigned)length);
    if (fwrite(line, 1, (size_t)length, log_file) != (size_t)length || fflush(log_file) != 0) exit(1);
}
static unsigned int clock_us(void) { return sceKernelGetSystemTimeLow(); }

/* 使用真实 forward 留下的 Q、K、V，对完整 128 维计算作独立标量比较。
 * 每次处理完最后一层时检查全部 Q 头，历史长度随真实生成逐步增加。 */
static void check_attention(AttentionCheck *check, int position) {
    const QwenRuntime *runtime = check->runtime;
    const KVCache *cache = &runtime->kv_cache;
    int head, time, index, kv_dim = cache->num_heads * cache->head_dim;
    int layer = cache->num_layers - 1;
    size_t layer_offset = (size_t)layer * cache->context * kv_dim;
    size_t scale_offset = (size_t)layer * cache->context * cache->num_heads;
    float inverse_sqrt = 1.0f / sqrtf((float)cache->head_dim);
    float scores[position + 1];
    float output[128] __attribute__((aligned(16)));
    for (head = 0; head < check->query_heads; ++head) {
        int kv_head = head / (check->query_heads / cache->num_heads);
        const float *query = runtime->query + head * cache->head_dim;
        attention_scores(cache, layer, kv_head, position, query, scores);
        for (time = 0; time <= position; ++time) {
            const int8_t *key = cache->keys + layer_offset + (size_t)time * kv_dim +
                                kv_head * cache->head_dim;
            float even = 0.0f, odd = 0.0f, expected, difference;
            for (index = 0; index < cache->head_dim; index += 2) {
                even += query[index] * (float)key[index];
                odd += query[index + 1] * (float)key[index + 1];
            }
            expected = (even + odd) * cache->key_scales[
                scale_offset + (size_t)time * cache->num_heads + kv_head] * inverse_sqrt;
            difference = fabsf(scores[time] - expected);
            if (!(difference <= 2.0e-5f * (1.0f + fabsf(expected)))) {
                log_line("[fail] QK position=%d head=%d time=%d actual=%.9g expected=%.9g\n",
                          position, head, time, scores[time], expected);
                exit(1);
            }
            if (difference > check->score_max_abs) check->score_max_abs = difference;
        }
        softmax_inplace(scores, position + 1);
        attention_weighted_sum(cache, layer, kv_head, position, scores, output);
        for (index = 0; index < cache->head_dim; ++index) {
            float expected = 0.0f, difference;
            for (time = 0; time <= position; ++time) {
                size_t offset = layer_offset + (size_t)time * kv_dim + kv_head * cache->head_dim;
                float weight = scores[time] * cache->value_scales[
                    scale_offset + (size_t)time * cache->num_heads + kv_head];
                expected += weight * (float)cache->values[offset + index];
            }
            difference = fabsf(output[index] - expected);
            if (!(difference <= 2.0e-5f * (1.0f + fabsf(expected)))) {
                log_line("[fail] AV position=%d head=%d index=%d actual=%.9g expected=%.9g\n",
                          position, head, index, output[index], expected);
                exit(1);
            }
            if (difference > check->output_max_abs) check->output_max_abs = difference;
        }
    }
    ++check->forwards;
}

static void progress(int position, int layer, int total, void *user) {
    scePowerTick(PSP_POWER_TICK_ALL);
    if (layer == 0 || layer == total) log_line("[progress] position=%d layer=%d/%d\n", position, layer, total);
    if (layer == total) check_attention(user, position);
}
static int output_token(int token, const uint8_t *bytes, size_t length, void *user) {
    (void)user;
    log_line("[token] id=%d bytes=%u text=%.*s\n", token, (unsigned)length,
              (int)length, bytes ? (const char *)bytes : "");
    return 1;
}

int main(void) {
    QwenModel model;
    QwenRuntime runtime;
    AttentionCheck attention_check = {0};
    Sampler sampler;
    FILE *settings;
    char error[256], prompt[512];
    int tokens[128], count, generated = 0, cache_mib, max_new;
    unsigned int fcr, started;
    size_t cached, resident;
    log_file = fopen("qwen_autotest.log", "wb");
    if (!log_file) return 1;
    log_line("[platform] free_partition_bytes=%u\n", sceKernelTotalFreeMemSize());
    __asm__ volatile ("cfc1 %0, $31" : "=r"(fcr));
    fcr = (fcr & ~0x00000f80u) | 0x01000000u;
    __asm__ volatile ("ctc1 %0, $31" : : "r"(fcr));
    scePowerSetClockFrequency(333, 333, 166);
    settings = fopen("qwen_test.txt", "rb");
    if (!settings || fscanf(settings, "%d %d\n", &cache_mib, &max_new) != 2 ||
        !fgets(prompt, sizeof(prompt), settings)) {
        log_line("[fail] cannot read qwen_test.txt\n"); return 1;
    }
    fclose(settings);
    prompt[strcspn(prompt, "\r\n")] = '\0';
    if (cache_mib < 0 || cache_mib > 6 || max_new < 1 || max_new > 32) return 1;
    if (!qwen_model_open(&model, "model.fhq4", error, sizeof(error)) ||
        !qwen_runtime_init(&runtime, &model, error, sizeof(error))) {
        log_line("[fail] %s\n", error); return 1;
    }
    runtime.clock_us = clock_us;
    attention_check.runtime = &runtime;
    attention_check.query_heads = model.config.num_attention_heads;
    qwen_runtime_set_progress(&runtime, progress, &attention_check);
    cached = qwen_model_cache_weights(&model, (size_t)cache_mib * 1048576u, error, sizeof(error));
    if (error[0] || cached != (size_t)cache_mib * 1048576u) {
        log_line("[fail] cache=%u: %s\n", (unsigned)cached, error); return 1;
    }
    resident = qwen_model_resident_bytes(&model) + qwen_runtime_resident_bytes(&runtime);
    log_line("[memory] model_bytes=%u runtime_bytes=%u cache_bytes=%u resident_bytes=%u\n",
              (unsigned)qwen_model_resident_bytes(&model), (unsigned)qwen_runtime_resident_bytes(&runtime),
              (unsigned)cached, (unsigned)resident);
    count = tokenizer_build_chat_prompt(&model, prompt, tokens, 128);
    if (count <= 0 || count + max_new > 128) {
        log_line("[fail] context exceeded\n"); return 1;
    }
    log_line("[prompt] text=%s count=%d\n", prompt, count);
    sampler_init(&sampler, 0, 1, 32, 1, 1);
    started = clock_us();
    if (!qwen_generate(&model, &runtime, &sampler, tokens, count, max_new,
                        output_token, NULL, &generated)) {
        log_line("[fail] generation io_error=%d\n", runtime.io_error); return 1;
    }
    log_line("[attention] head_dim=%d forwards=%d score_max_abs=%.9g output_max_abs=%.9g\n",
              model.config.head_dim, attention_check.forwards,
              attention_check.score_max_abs, attention_check.output_max_abs);
    log_line("[pass] generated=%d elapsed_us=%u streamed_bytes=%llu\n", generated,
              clock_us() - started, (unsigned long long)runtime.streamed_bytes);
    qwen_runtime_free(&runtime); qwen_model_close(&model);
    fclose(log_file);
    sceKernelExitGame();
    return 0;
}
