#ifndef DOT_Q4_BENCH_H
#define DOT_Q4_BENCH_H

#include <stddef.h>
#include <stdint.h>

#define BENCH_MAX_CASES 32
#define BENCH_MAX_SAMPLES 9
#define BENCH_KERNELS 5
#define BENCH_MAX_OUTPUTS (16 * 2048)
#define BENCH_IMPLEMENTATION_ID "five-kernels-1"

typedef struct {
    const uint8_t *weights;
    const float *input;
    float *output, *sums;
    uint32_t rows, columns, batch, weight_row_bytes, input_stride;
} BenchTask;

typedef void (*BenchMatmul)(const BenchTask *);

typedef struct {
    char name[32];
    uint32_t component, rows, columns, batch, reference_offset, bound_offset;
    uint32_t row_start, column_start, full;
} BenchCase;

typedef struct {
    uint8_t *bytes;
    size_t size;
    uint32_t *header;
    BenchCase *cases;
    int case_count;
    float *output;
    float *sums;
} BenchData;

typedef struct {
    int timed, duration_ms, repetitions, samples, cpu_mhz, full_suite;
} BenchSettings;

typedef struct {
    uint64_t elapsed_us, iterations;
    double per_operation_us;
} BenchSample;

typedef struct {
    int available, checked, complete, sample_count;
    float max_absolute_error, max_normalized_error;
    double median_us, minimum_us, maximum_us;
    BenchSample samples[BENCH_MAX_SAMPLES];
} BenchResult;

typedef struct {
    int valid, cancelled;
    float cpu_mhz, bus_mhz;
    BenchSettings settings;
    BenchResult results[BENCH_KERNELS];
} BenchComparison;

typedef struct {
    uint64_t (*clock_us)(void);
    int (*progress)(int kernel, int sample, float fraction, void *user);
    void *user;
} BenchPlatform;

extern const char *const bench_kernel_names[BENCH_KERNELS];

void bench_scalar_init(void);
float bench_dot_scalar(const uint8_t *row, const float *x, int blocks, const float *block_sums);
void bench_matmul_scalar(const BenchTask *task);
int bench_custom_available(void);
void bench_matmul_custom(const BenchTask *task);
void bench_matmul_custom_vh2f(const BenchTask *task);
int bench_load(BenchData *data, const char *path, char *error, size_t error_size);
void bench_free(BenchData *data);
void bench_execute(BenchData *data, int case_index, int kernel);
int bench_validate(BenchData *data, int case_index, int kernel, BenchResult *result,
                   char *error, size_t error_size);
int bench_compare(BenchData *data, int case_index, const BenchSettings *settings,
                  const BenchPlatform *platform, BenchComparison *comparison,
                  char *error, size_t error_size);
int bench_export(const char *path, const BenchData *data, const BenchComparison *comparisons,
                 char *error, size_t error_size);

#endif
