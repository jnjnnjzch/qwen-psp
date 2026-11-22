#include "falcon_h1.h"

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

static void log_line(const char *format, ...) {
    char line[1024];
    va_list args;
    int length;
    va_start(args, format);
    length = vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (length <= 0 || length >= (int)sizeof(line)) exit(1);
    sceIoWrite(1, line, (unsigned)length);
    if (fwrite(line, 1, (size_t)length, log_file) != (size_t)length ||
        fflush(log_file) != 0) exit(1);
}

static unsigned int clock_us(void) {
    return sceKernelGetSystemTimeLow();
}

static void progress(int position, int layer, int total, void *user) {
    (void)user;
    scePowerTick(PSP_POWER_TICK_ALL);
    if (layer == 0 || layer == total)
        log_line("[progress] position=%d layer=%d/%d\n", position, layer, total);
}

static int output_token(int token, const uint8_t *piece, size_t length, void *user) {
    (void)user;
    log_line("[token] id=%d bytes=%u text=%.*s\n", token, (unsigned)length,
              (int)length, piece ? (const char *)piece : "");
    return 1;
}

int main(void) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    FILE *settings;
    char error[256], prompt[512];
    int tokens[128], count, generated = 0, cache_mb, max_new, i;
    unsigned int fcr, start;
    size_t cached, resident;
    log_file = fopen("qwen_autotest.log", "wb");
    if (!log_file) return 1;
    log_line("[platform] free_partition_bytes=%u\n", sceKernelTotalFreeMemSize());
    __asm__ volatile ("cfc1 %0, $31" : "=r"(fcr));
    fcr = (fcr & ~0x00000f80u) | 0x01000000u;
    __asm__ volatile ("ctc1 %0, $31" : : "r"(fcr));
    scePowerSetClockFrequency(333, 333, 166);
    settings = fopen("qwen_test.txt", "rb");
    if (!settings || fscanf(settings, "%d %d\n", &cache_mb, &max_new) != 2 ||
        !fgets(prompt, sizeof(prompt), settings)) {
        log_line("[fail] qwen_test.txt must contain cache-MiB max-new and prompt\n");
        return 1;
    }
    fclose(settings);
    prompt[strcspn(prompt, "\r\n")] = '\0';
    if (cache_mb < 0 || cache_mb > 6 || max_new < 1 || max_new > 32) return 1;
    if (!falcon_model_open(&model, "model.fhq4", error, sizeof(error)) ||
        !falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
        log_line("[fail] %s\n", error);
        return 1;
    }
    runtime.clock_us = clock_us;
    falcon_runtime_set_progress(&runtime, progress, NULL);
    cached = falcon_model_cache_weights(&model, (size_t)cache_mb * 1048576u,
                                        error, sizeof(error));
    if (error[0] || cached != (size_t)cache_mb * 1048576u) {
        log_line("[fail] cache=%u: %s\n", (unsigned)cached, error);
        return 1;
    }
    resident = falcon_model_resident_bytes(&model) + falcon_runtime_resident_bytes(&runtime);
    log_line("[memory] model_bytes=%u runtime_bytes=%u cache_bytes=%u resident_bytes=%u\n",
              (unsigned)falcon_model_resident_bytes(&model),
              (unsigned)falcon_runtime_resident_bytes(&runtime),
              (unsigned)cached, (unsigned)resident);
    count = falcon_build_chat_prompt(&model, prompt, tokens, 128);
    if (count <= 0 || count + max_new > 128) {
        log_line("[fail] prompt exceeds context\n");
        return 1;
    }
    log_line("[prompt] text=%s count=%d\n", prompt, count);
    for (i = 0; i < count; ++i) log_line("[input] %d\n", tokens[i]);
    falcon_sampler_init(&sampler, 0, 1, 32, 1, 1);
    start = clock_us();
    if (!falcon_generate(&model, &runtime, &sampler, tokens, count, max_new,
                         output_token, NULL, &generated)) {
        log_line("[fail] generation io_error=%d\n", runtime.io_error);
        return 1;
    }
    log_line("[pass] generated=%d elapsed_us=%u streamed_bytes=%llu\n",
              generated, clock_us() - start,
              (unsigned long long)runtime.streamed_bytes);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    fclose(log_file);
    sceKernelExitGame();
    return 0;
}
