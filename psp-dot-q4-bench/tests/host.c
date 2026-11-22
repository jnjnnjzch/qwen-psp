#include "bench.h"

#include <stdio.h>

int main(int argc, char **argv) {
    BenchData data;
    char error[256];
    if (argc != 2) return 2;
    if (!bench_load(&data, argv[1], error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
    unsigned int outputs = 0;
    for (int i = 0; i < data.case_count; ++i) {
        BenchResult result;
        if (!bench_validate(&data, i, 0, &result, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); bench_free(&data); return 1;
        }
        outputs += data.cases[i].rows * data.cases[i].batch;
        printf("[pass] case=%d name=%s max_abs=%.9g normalized=%.9g\n", i, data.cases[i].name,
               result.max_absolute_error, result.max_normalized_error);
    }
    printf("[host-pass] cases=%d outputs=%u custom=%d\n", data.case_count, outputs, bench_custom_available());
    bench_free(&data);
    return 0;
}
