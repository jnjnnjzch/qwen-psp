#include "bench.h"

#include <math.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#if defined(_PSP_FW_VERSION)
extern float falcon_dot_q4_vfpu(const uint8_t *, const float *, int, const float *);
extern float bench_dot_q4_vh2f(const uint8_t *, const float *, int, const float *);
static void matmul_llmpsp(const BenchTask *task);
static void matmul_vh2f(const BenchTask *task);
#endif

const char *const bench_kernel_names[BENCH_KERNELS] = {
    "C / SCALAR", "LLMPSP / VFPU", "VH2F / VFPU", "CUSTOM / VMMUL", "VH2F / VMMUL"
};

static BenchMatmul kernel_function(int kernel) {
    if (kernel == 0) return bench_matmul_scalar;
#if defined(_PSP_FW_VERSION)
    if (kernel == 1) return matmul_llmpsp;
    if (kernel == 2) return matmul_vh2f;
    if (kernel == 3 && bench_custom_available()) return bench_matmul_custom;
    if (kernel == 4 && bench_custom_available()) return bench_matmul_custom_vh2f;
#endif
    abort();
}

static int kernel_available(int kernel) {
    if (kernel == 0) return 1;
#if defined(_PSP_FW_VERSION)
    if (kernel == 1 || kernel == 2) return 1;
    if (kernel == 3 || kernel == 4) return bench_custom_available();
#endif
    return 0;
}

static int span_valid(uint32_t offset, size_t length, size_t size) {
    return offset >= 64 && offset <= size && length <= size - offset;
}

int bench_load(BenchData *data, const char *path, char *error, size_t error_size) {
    FILE *file;
    long size;
    memset(data, 0, sizeof(*data));
    file = fopen(path, "rb");
    if (!file) { snprintf(error, error_size, "Cannot open BENCH.DAT"); return 0; }
    if (fseek(file, 0, SEEK_END) || (size = ftell(file)) < 64 || size > 8 * 1024 * 1024 ||
        fseek(file, 0, SEEK_SET)) {
        fclose(file); snprintf(error, error_size, "Invalid dataset size"); return 0;
    }
    data->size = (size_t)size;
    data->bytes = memalign(16, data->size);
    if (!data->bytes) { fclose(file); snprintf(error, error_size, "Dataset allocation failed"); return 0; }
    size_t received = fread(data->bytes, 1, data->size, file);
    int closed = fclose(file);
    if (received != data->size || closed) {
        bench_free(data); snprintf(error, error_size, "Dataset read failed"); return 0;
    }
    uint32_t *h = data->header = (uint32_t *)data->bytes;
    if (h[0] != 0x42345144 || h[1] != 2 || h[2] == 0 || h[2] > BENCH_MAX_CASES ||
        h[3] != 16 || h[4] != 64 || h[11] != 1024 || h[12] != 3072 || h[14] != data->size ||
        h[15] != sizeof(BenchCase) || h[6] != 2048 * 1024 / 32 * 18 || h[8] != 1024 * 3072 / 32 * 18 ||
        !span_valid(h[4], h[2] * sizeof(BenchCase), data->size) ||
        !span_valid(h[5], h[6], data->size) || !span_valid(h[7], h[8], data->size) ||
        !span_valid(h[9], 16 * 1024 * sizeof(float), data->size) ||
        !span_valid(h[10], 16 * 3072 * sizeof(float), data->size) ||
        (h[5] | h[7] | h[9] | h[10]) % 16 ||
        crc32(0, data->bytes + 64, data->size - 64) != h[13]) {
        bench_free(data); snprintf(error, error_size, "Dataset header or CRC32 invalid"); return 0;
    }
    data->case_count = (int)h[2];
    data->cases = (BenchCase *)(data->bytes + h[4]);
    for (int i = 0; i < data->case_count; ++i) {
        BenchCase *c = &data->cases[i];
        uint32_t limit = c->component == 0 ? 2048 : 1024;
        if (!memchr(c->name, 0, sizeof(c->name)) || c->component > 1 || c->rows == 0 ||
            c->row_start >= limit || c->rows > limit - c->row_start ||
            c->columns == 0 || c->columns % 32 || c->column_start % 32 ||
            c->column_start >= h[11 + c->component] ||
            c->columns > h[11 + c->component] - c->column_start ||
            c->batch == 0 || c->batch > 16 || c->full > 1 ||
            !span_valid(c->reference_offset, c->rows * c->batch * sizeof(float), data->size) ||
            !span_valid(c->bound_offset, c->rows * c->batch * sizeof(float), data->size) ||
            (c->reference_offset | c->bound_offset) % 16) {
            bench_free(data); snprintf(error, error_size, "Invalid case geometry"); return 0;
        }
    }
    data->output = memalign(16, BENCH_MAX_OUTPUTS * sizeof(float));
    data->sums = memalign(16, 16 * 96 * sizeof(float));
    if (!data->output || !data->sums) {
        bench_free(data); snprintf(error, error_size, "Work buffer allocation failed"); return 0;
    }
    bench_scalar_init();
    return 1;
}

void bench_free(BenchData *data) {
    free(data->bytes);
    free(data->output);
    free(data->sums);
    memset(data, 0, sizeof(*data));
}

#if defined(_PSP_FW_VERSION)
typedef float (*BenchDot)(const uint8_t *, const float *, int, const float *);

static void matmul_dot(const BenchTask *task, BenchDot dot) {
    for (uint32_t batch = 0; batch < task->batch; ++batch) {
        for (uint32_t block = 0; block < task->columns / 32; ++block) {
            float sum = 0;
            const float *x = task->input + batch * task->input_stride + block * 32;
            for (int j = 0; j < 32; ++j) sum += x[j];
            task->sums[batch * 96 + block] = sum;
        }
        for (uint32_t row = 0; row < task->rows; ++row)
            task->output[batch * task->rows + row] =
                dot(task->weights + row * task->weight_row_bytes,
                    task->input + batch * task->input_stride,
                    (int)task->columns / 32, task->sums + batch * 96);
    }
}

static void matmul_llmpsp(const BenchTask *task) {
    matmul_dot(task, falcon_dot_q4_vfpu);
}

static void matmul_vh2f(const BenchTask *task) {
    matmul_dot(task, bench_dot_q4_vh2f);
}
#endif

void bench_execute(BenchData *data, int case_index, int kernel) {
    const BenchCase *c = &data->cases[case_index];
    const uint8_t *weights = data->bytes + data->header[c->component ? 7 : 5];
    const float *input = (const float *)(data->bytes + data->header[9 + c->component]);
    uint32_t stride = data->header[11 + c->component], row_bytes = stride / 32 * 18;
    BenchTask task = {
        weights + c->row_start * row_bytes + c->column_start / 32 * 18,
        input + c->column_start, data->output, data->sums,
        c->rows, c->columns, c->batch, row_bytes, stride
    };
    kernel_function(kernel)(&task);
}

int bench_validate(BenchData *data, int case_index, int kernel, BenchResult *result,
                   char *error, size_t error_size) {
    const BenchCase *c = &data->cases[case_index];
    const float *reference = (const float *)(data->bytes + c->reference_offset);
    const float *bounds = (const float *)(data->bytes + c->bound_offset);
    memset(result, 0, sizeof(*result));
    result->available = kernel_available(kernel);
    if (!result->available) return 1;
    for (uint32_t i = 0; i < c->batch * c->rows; ++i) data->output[i] = NAN;
    bench_execute(data, case_index, kernel);
    for (uint32_t i = 0; i < c->batch * c->rows; ++i) {
        float delta = fabsf(data->output[i] - reference[i]);
        float normalized = delta / (2e-5f + 2e-5f * bounds[i]);
        if (!isfinite(data->output[i]) || !isfinite(normalized) || normalized > 1.0f) {
            snprintf(error, error_size, "%s: output %u failed (%.7g / %.7g)",
                     bench_kernel_names[kernel], (unsigned)i, data->output[i], reference[i]);
            return 0;
        }
        if (delta > result->max_absolute_error) result->max_absolute_error = delta;
        if (normalized > result->max_normalized_error) result->max_normalized_error = normalized;
    }
    result->checked = 1;
    return 1;
}

static void summarize(BenchResult *result) {
    double sorted[BENCH_MAX_SAMPLES];
    for (int i = 0; i < result->sample_count; ++i) {
        sorted[i] = result->samples[i].per_operation_us;
        for (int j = i; j > 0 && sorted[j] < sorted[j - 1]; --j) {
            double value = sorted[j]; sorted[j] = sorted[j - 1]; sorted[j - 1] = value;
        }
    }
    int count = result->sample_count;
    result->minimum_us = sorted[0];
    result->maximum_us = sorted[count - 1];
    result->median_us = count % 2 ? sorted[count / 2] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2;
    result->complete = 1;
}

int bench_compare(BenchData *data, int case_index, const BenchSettings *settings,
                  const BenchPlatform *platform, BenchComparison *comparison,
                  char *error, size_t error_size) {
    memset(comparison, 0, sizeof(*comparison));
    comparison->settings = *settings;
    for (int kernel = 0; kernel < BENCH_KERNELS; ++kernel) {
        if (!bench_validate(data, case_index, kernel, &comparison->results[kernel], error, error_size)) return 0;
        if (platform->progress(kernel, -1, 0, platform->user)) { comparison->cancelled = 1; return 1; }
    }
    for (int sample = 0; sample < settings->samples; ++sample) {
        for (int slot = 0; slot < BENCH_KERNELS; ++slot) {
            int kernel = (slot + sample) % BENCH_KERNELS;
            BenchResult *result = &comparison->results[kernel];
            if (!result->available) continue;
            bench_execute(data, case_index, kernel);
            /* 使用实际计算校准批次数量，使 DOT 的计时跨越多个时钟刻度。 */
            uint32_t chunk = 1;
            uint64_t calibration;
            do {
                uint64_t start = platform->clock_us();
                for (uint32_t i = 0; i < chunk; ++i) bench_execute(data, case_index, kernel);
                calibration = platform->clock_us() - start;
                if (calibration >= 2000 || chunk >= 65536) break;
                chunk *= 2;
            } while (1);
            if (platform->progress(kernel, sample, 0, platform->user)) {
                comparison->cancelled = 1; return 1;
            }
            BenchSample *measurement = &result->samples[sample];
            uint64_t target_us = (uint64_t)settings->duration_ms * 1000;
            do {
                uint32_t current = chunk;
                if (!settings->timed && current > (uint32_t)settings->repetitions - measurement->iterations)
                    current = (uint32_t)((uint64_t)settings->repetitions - measurement->iterations);
                uint64_t start = platform->clock_us();
                for (uint32_t i = 0; i < current; ++i) bench_execute(data, case_index, kernel);
                measurement->elapsed_us += platform->clock_us() - start;
                measurement->iterations += current;
                float fraction = settings->timed ? (float)measurement->elapsed_us / target_us :
                    (float)measurement->iterations / settings->repetitions;
                if (platform->progress(kernel, sample, fraction, platform->user)) {
                    comparison->cancelled = 1; return 1;
                }
            } while (settings->timed ? measurement->elapsed_us < target_us :
                     measurement->iterations < (uint64_t)settings->repetitions);
            if (measurement->elapsed_us == 0) {
                snprintf(error, error_size, "Below timer resolution; increase repetitions"); return 0;
            }
            measurement->per_operation_us = (double)measurement->elapsed_us / measurement->iterations;
            ++result->sample_count;
        }
    }
    for (int kernel = 0; kernel < BENCH_KERNELS; ++kernel)
        if (comparison->results[kernel].available) summarize(&comparison->results[kernel]);
    comparison->valid = 1;
    return 1;
}

int bench_export(const char *path, const BenchData *data, const BenchComparison *comparisons,
                 char *error, size_t error_size) {
    FILE *file = fopen(path, "w");
    if (!file) { snprintf(error, error_size, "Cannot create CSV on Memory Stick"); return 0; }
    fprintf(file, "dataset_crc32,case,kernel,B,N,K,row_start,column_start,suite,interface,implementation_id,mode,target_ms,target_repetitions,samples,requested_cpu_mhz,cpu_mhz,bus_mhz,"
                  "sample,iterations,elapsed_us,us_per_operation,median_us,min_us,max_us,gflops,speedup,"
                  "max_abs_error,max_normalized_error\n");
    for (int index = 0; index < data->case_count; ++index) {
        const BenchComparison *comparison = &comparisons[index];
        const BenchCase *c = &data->cases[index];
        if (!comparison->valid) continue;
        for (int kernel = 0; kernel < BENCH_KERNELS; ++kernel) {
            const BenchResult *r = &comparison->results[kernel];
            if (!r->complete) continue;
            double operations = 2.0 * c->batch * c->rows * c->columns;
            for (int sample = 0; sample < r->sample_count; ++sample) {
                const BenchSample *s = &r->samples[sample];
                fprintf(file, "%08x,%s,%s,%u,%u,%u,%u,%u,%s,matrix_q4," BENCH_IMPLEMENTATION_ID ",%s,%d,%d,%d,%d,%.6f,%.6f,%d,%llu,%llu,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.9g,%.9g\n",
                        (unsigned)data->header[13], c->name, bench_kernel_names[kernel],
                        (unsigned)c->batch, (unsigned)c->rows, (unsigned)c->columns,
                        (unsigned)c->row_start, (unsigned)c->column_start, c->full ? "FULL" : "QUICK",
                        comparison->settings.timed ? "TIME" : "COUNT", comparison->settings.duration_ms,
                        comparison->settings.repetitions, comparison->settings.samples, comparison->settings.cpu_mhz,
                        comparison->cpu_mhz, comparison->bus_mhz, sample + 1, (unsigned long long)s->iterations,
                        (unsigned long long)s->elapsed_us, s->per_operation_us, r->median_us, r->minimum_us,
                        r->maximum_us, operations / r->median_us / 1000.0,
                        comparison->results[0].median_us / r->median_us,
                        r->max_absolute_error, r->max_normalized_error);
            }
        }
    }
    int failed = ferror(file);
    int closed = fclose(file);
    if (failed || closed) { snprintf(error, error_size, "CSV write failed"); return 0; }
    return 1;
}
