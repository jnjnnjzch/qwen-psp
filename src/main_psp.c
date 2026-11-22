#include "bench.h"

#include <pspkernel.h>
#include <pspctrl.h>
#include <pspdebug.h>
#include <pspdisplay.h>
#include <pspge.h>
#include <psppower.h>
#include <psprtc.h>
#include <pspiofilemgr.h>

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

PSP_MODULE_INFO("DotQ4Bench", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_MAIN_THREAD_STACK_SIZE_KB(64);
PSP_HEAP_SIZE_KB(10 * 1024);

#define WIDTH 480
#define HEIGHT 272
#define STRIDE 512
#define RGB(r,g,b) ((uint32_t)((r) | ((g) << 8) | ((b) << 16) | 0xff000000u))
#define WHITE RGB(231,239,249)
#define MUTED RGB(140,163,190)
#define CYAN RGB(105,220,234)
#define GOLD RGB(232,194,125)
#define RED RGB(246,143,145)

static volatile int alive = 1;
static uint32_t *framebuffer, *background;
static int frame_index, page, selected, setting_row;
static unsigned int previous_buttons;
static BenchData data;
static BenchComparison comparisons[BENCH_MAX_CASES];
static BenchSettings settings = {1, 250, 100, 3, 333, 0};
static char status_text[96] = "Select a workload and press X";
static int busy, current_kernel, current_sample;
static float current_fraction;
static float cpu_mhz, bus_mhz;
static int free_memory_kib;
static unsigned int export_sequence;
static unsigned int screenshot_sequence;

static int exit_callback(int arg1, int arg2, void *user) {
    (void)arg1; (void)arg2; (void)user;
    alive = 0;
    return 0;
}

static int callback_thread(SceSize args, void *argp) {
    (void)args; (void)argp;
    int callback = sceKernelCreateCallback("ExitCallback", exit_callback, NULL);
    if (callback < 0 || sceKernelRegisterExitCallback(callback) < 0) return -1;
    sceKernelSleepThreadCB();
    return 0;
}

static void text_at(int x, int y, uint32_t color, const char *format, ...) {
    char message[160];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    for (size_t i = 0; message[i] && x + 7 <= WIDTH; ++i, x += 7)
        pspDebugScreenPutChar(x, y, color, (unsigned char)message[i]);
}

static void rectangle(int x, int y, int width, int height, uint32_t color) {
    for (int row = y; row < y + height && row < HEIGHT; ++row)
        for (int column = x; column < x + width && column < WIDTH; ++column)
            framebuffer[row * STRIDE + column] = color;
}

static void make_background(void) {
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < STRIDE; ++x) {
            float light = (float)(HEIGHT - y) / HEIGHT;
            background[y * STRIDE + x] = RGB((int)(9 + 7 * light), (int)(19 + 13 * light), (int)(36 + 21 * light));
        }
    for (int ribbon = 0; ribbon < 4; ++ribbon)
        for (int x = 0; x < WIDTH; ++x) {
            int y = 179 + ribbon * 13 + (int)(17 * sinf((float)x / 105 + ribbon * 0.55f));
            background[y * STRIDE + x] = RGB(30 + ribbon * 3, 52 + ribbon * 3, 76 + ribbon * 4);
        }
}

static void draw_workloads(void) {
    int first = selected > 2 ? selected - 2 : 0;
    if (first > data.case_count - 5) first = data.case_count - 5;
    if (first < 0) first = 0;
    for (int i = first; i < first + 5 && i < data.case_count; ++i) {
        const BenchCase *c = &data.cases[i];
        int y = 92 + (i - first) * 24;
        if (i == selected) {
            rectangle(16, y - 5, 448, 22, RGB(28,62,84));
            rectangle(16, y - 5, 3, 22, CYAN);
        }
        text_at(28, y, i == selected ? WHITE : MUTED, "%02d  %s", i + 1, c->name);
        text_at(294, y, i == selected ? CYAN : MUTED, "%ux%ux%u", (unsigned)c->batch,
                (unsigned)c->rows, (unsigned)c->columns);
        if (comparisons[i].valid) rectangle(451, y + 2, 4, 4, CYAN);
    }
    const BenchCase *slice = &data.cases[selected];
    text_at(23, 218, MUTED, "%s / row %u / col %u / %d of %d", slice->full ? "FULL" : "QUICK",
            (unsigned)slice->row_start, (unsigned)slice->column_start, selected + 1, data.case_count);
    text_at(23, 232, MUTED, "START: %s suite / %.2f MiB resident", settings.full_suite ? "FULL" : "QUICK", data.size / 1048576.0);
}

static void draw_settings(void) {
    const char *labels[] = {"STOP CONDITION", "TIME / SAMPLE", "OPERATIONS / SAMPLE", "SAMPLES", "CPU CLOCK", "START SUITE"};
    for (int row = 0; row < 6; ++row) {
        int y = 93 + row * 21;
        if (row == setting_row) {
            rectangle(16, y - 5, 448, 23, RGB(28,62,84));
            rectangle(16, y - 5, 3, 23, CYAN);
        }
        text_at(29, y, row == setting_row ? WHITE : MUTED, "%s", labels[row]);
        if (row == 0) text_at(290, y, CYAN, "< %s >", settings.timed ? "TIME" : "COUNT");
        if (row == 1) text_at(290, y, settings.timed ? CYAN : MUTED, "< %d ms >", settings.duration_ms);
        if (row == 2) text_at(290, y, settings.timed ? MUTED : CYAN, "< %d >", settings.repetitions);
        if (row == 3) text_at(290, y, CYAN, "< %d >", settings.samples);
        if (row == 4) text_at(290, y, CYAN, "< %d MHz >", settings.cpu_mhz);
        if (row == 5) text_at(290, y, CYAN, "< %s >", settings.full_suite ? "FULL" : "QUICK");
    }
    text_at(23, 227, MUTED, "LEFT/RIGHT edit     Free user RAM: %d KiB", free_memory_kib);
}

static void draw_results(void) {
    const BenchCase *c = &data.cases[selected];
    const BenchComparison *comparison = &comparisons[selected];
    text_at(23, 89, WHITE, "%s  |  %ux%ux%u", c->name,
            (unsigned)c->batch, (unsigned)c->rows, (unsigned)c->columns);
    double maximum = 0;
    for (int i = 0; i < BENCH_KERNELS; ++i)
        if (comparison->results[i].complete && comparison->results[i].median_us > maximum)
            maximum = comparison->results[i].median_us;
    const uint32_t colors[] = {
        RGB(156,177,202), RGB(83,174,208), CYAN, GOLD, RGB(196,155,231)
    };
    for (int i = 0; i < BENCH_KERNELS; ++i) {
        const BenchResult *r = &comparison->results[i];
        int y = 106 + i * 23;
        text_at(23, y, colors[i], "%s", bench_kernel_names[i]);
        if (i >= 3 && !bench_custom_available()) {
            text_at(207, y, GOLD, "UNIMPLEMENTED");
            rectangle(23, y + 12, 434, 3, RGB(44,50,63));
        } else if (comparison->valid && r->complete) {
            text_at(207, y, WHITE, "%.2f us   %.2fx", r->median_us,
                    comparison->results[0].median_us / r->median_us);
            rectangle(23, y + 12, 434, 3, RGB(35,49,66));
            int bar_width = (int)(434 * r->median_us / maximum);
            rectangle(23, y + 12, bar_width > 0 ? bar_width : 1, 3, colors[i]);
        } else text_at(207, y, MUTED, "%s", busy && i == current_kernel ? "RUNNING" : "NO RESULT");
    }
    if (comparison->valid) {
        text_at(23, 224, MUTED, "%d samples / %.1f MHz / speedup vs C",
                comparison->settings.samples, comparison->cpu_mhz);
    } else text_at(23, 224, MUTED, "LATENCY / median microseconds / shorter is faster");
}

static void draw(void) {
    /* 使用与 PSPSDK debug screen 相同的 uncached VRAM 地址。 */
    framebuffer = (uint32_t *)((uintptr_t)sceGeEdramGetAddr() | 0x40000000u) + frame_index * STRIDE * HEIGHT;
    memcpy(framebuffer, background, STRIDE * HEIGHT * 4);
    pspDebugScreenSetBase(framebuffer);
    pspDebugScreenSetOffset(0);
    pspDebugScreenEnableBackColor(0);
    text_at(22, 12, MUTED, "PSP / COMPUTE LAB");
    text_at(316, 12, MUTED, "%.1f / %.1f MHz", cpu_mhz, bus_mhz);
    text_at(22, 33, WHITE, "DOT / Q4");
    text_at(164, 33, MUTED, "REAL SLICES / MATRIX TASKS");
    const char *tabs[] = {"WORKLOADS", "SETTINGS", "RESULTS"};
    for (int i = 0; i < 3; ++i) {
        text_at(23 + i * 150, 63, i == page ? WHITE : MUTED, "%s", tabs[i]);
        if (i == page) rectangle(23 + i * 150, 77, 91, 2, CYAN);
    }
    if (page == 0) draw_workloads();
    else if (page == 1) draw_settings();
    else draw_results();
    if (busy) {
        rectangle(16, 209, 448, 32, RGB(14,31,49));
        text_at(23, 213, WHITE, "%s / %s", current_sample < 0 ? "CHECK" : "MEASURE",
                bench_kernel_names[current_kernel]);
        text_at(23, 227, MUTED, "Sample %d / %d    O Cancel after operation", current_sample + 1, settings.samples);
        float fraction = current_fraction > 1 ? 1 : current_fraction;
        rectangle(297, 214, 157, 4, RGB(38,62,82));
        rectangle(297, 214, (int)(157 * fraction), 4, CYAN);
    }
    rectangle(0, 247, WIDTH, 25, RGB(10,19,32));
    text_at(16, 251, MUTED, "X RUN  START SUITE  L/R PAGE  TRIANGLE CSV");
    text_at(16, 263, CYAN, "%.64s", status_text);
    sceDisplayWaitVblankStart();
    sceDisplaySetFrameBuf(framebuffer, STRIDE, PSP_DISPLAY_PIXEL_FORMAT_8888, PSP_DISPLAY_SETBUF_IMMEDIATE);
    frame_index ^= 1;
}

static void fatal_error(const char *error) {
    fprintf(stderr, "[fail] %s\n", error);
    snprintf(status_text, sizeof(status_text), "FAILED: %.83s", error);
    busy = 0;
    while (alive) { draw(); sceKernelDelayThread(16000); }
    sceKernelExitGame();
    exit(1);
}

static uint64_t clock_us(void) {
    return sceKernelGetSystemTimeWide();
}

static int progress(int kernel, int sample, float fraction, void *user) {
    (void)user;
    current_kernel = kernel;
    current_sample = sample;
    current_fraction = fraction;
    SceCtrlData pad;
    sceCtrlPeekBufferPositive(&pad, 1);
    previous_buttons = pad.Buttons;
    if (!alive || (pad.Buttons & PSP_CTRL_CIRCLE)) return 1;
    /* 界面更新限制在测量批次之外，每次更新间隔至少 50 ms。 */
    static uint64_t last_draw;
    uint64_t now = clock_us();
    if (sample < 0 || now - last_draw >= 50000) {
        scePowerTick(PSP_POWER_TICK_ALL);
        draw();
        last_draw = clock_us();
    }
    return 0;
}

static void ensure_directory(const char *path) {
    struct stat info;
    if (stat(path, &info) == 0) {
        if (!S_ISDIR(info.st_mode)) fatal_error("Result directory path is a file");
        return;
    }
    if (errno != ENOENT || mkdir(path, 0777) != 0) fatal_error("Cannot create result directory");
}

static void export_results(void) {
    char path[192], error[256];
    ScePspDateTime time;
    ensure_directory("ms0:/PSP");
    ensure_directory("ms0:/PSP/SAVEDATA");
    ensure_directory("ms0:/PSP/SAVEDATA/DOTQ4BENCH");
    if (sceRtcGetCurrentClockLocalTime(&time) < 0) fatal_error("Cannot read local clock");
    snprintf(path, sizeof(path), "ms0:/PSP/SAVEDATA/DOTQ4BENCH/%04d%02d%02d-%02d%02d%02d-%u.csv",
             time.year, time.month, time.day, time.hour, time.minute, time.second, export_sequence++);
    if (!bench_export(path, &data, comparisons, error, sizeof(error))) fatal_error(error);
    snprintf(status_text, sizeof(status_text), "CSV saved / PSP/SAVEDATA/DOTQ4BENCH");
    printf("[export] %s\n", path);
}

static int run_case(void) {
    char error[256];
    BenchPlatform platform = {clock_us, progress, NULL};
    if (scePowerSetClockFrequency(settings.cpu_mhz, settings.cpu_mhz, settings.cpu_mhz / 2) < 0)
        fatal_error("Cannot set CPU clock");
    cpu_mhz = scePowerGetCpuClockFrequencyFloat();
    bus_mhz = scePowerGetBusClockFrequencyFloat();
    busy = 1;
    page = 2;
    current_sample = -1;
    current_kernel = 0;
    current_fraction = 0;
    snprintf(status_text, sizeof(status_text), "%s", data.cases[selected].name);
    memset(&comparisons[selected], 0, sizeof(comparisons[selected]));
    draw();
    scePowerTick(PSP_POWER_TICK_ALL);
    printf("[case] index=%d name=%s mode=%s requested_cpu=%d cpu=%.6f bus=%.6f\n", selected,
           data.cases[selected].name, settings.timed ? "TIME" : "COUNT", settings.cpu_mhz, cpu_mhz, bus_mhz);
    if (!bench_compare(&data, selected, &settings, &platform, &comparisons[selected], error, sizeof(error)))
        fatal_error(error);
    busy = 0;
    comparisons[selected].cpu_mhz = cpu_mhz;
    comparisons[selected].bus_mhz = bus_mhz;
    if (comparisons[selected].cancelled) {
        snprintf(status_text, sizeof(status_text), "Cancelled / incomplete comparison discarded");
        printf("[cancel] index=%d\n", selected);
        return 0;
    }
    for (int kernel = 0; kernel < BENCH_KERNELS; ++kernel) {
        const BenchResult *r = &comparisons[selected].results[kernel];
        if (r->available) printf("[result] case=%d kernel=%d median_us=%.6f error=%.9g normalized=%.9g samples=%d\n",
                                selected, kernel, r->median_us, r->max_absolute_error,
                                r->max_normalized_error, r->sample_count);
        else printf("[unimplemented] case=%d kernel=%d\n", selected, kernel);
    }
    return 1;
}

static int cycle_value(int value, int direction, const int *choices, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (choices[i] == value) return choices[(i + count + direction) % count];
    abort();
}

static void edit_setting(int direction) {
    const int durations[] = {100,250,500,1000,2000};
    const int repetitions[] = {1,10,100,1000,10000};
    const int samples[] = {1,3,5,7,9};
    const int clocks[] = {222,266,333};
    if (setting_row == 0) settings.timed = !settings.timed;
    if (setting_row == 1) settings.duration_ms = cycle_value(settings.duration_ms, direction, durations, 5);
    if (setting_row == 2) settings.repetitions = cycle_value(settings.repetitions, direction, repetitions, 5);
    if (setting_row == 3) settings.samples = cycle_value(settings.samples, direction, samples, 5);
    if (setting_row == 4) settings.cpu_mhz = cycle_value(settings.cpu_mhz, direction, clocks, 3);
    if (setting_row == 5) settings.full_suite = !settings.full_suite;
    snprintf(status_text, sizeof(status_text), "Settings apply to the next comparison");
}

int main(int argc, char **argv) {
    char dataset_path[512], error[256];
    pspDebugScreenInit();
    sceDisplaySetMode(0, WIDTH, HEIGHT);
    background = malloc(STRIDE * HEIGHT * sizeof(uint32_t));
    if (!background) { fprintf(stderr, "[fail] Background allocation failed\n"); sceKernelExitGame(); return 1; }
    make_background();
    int callback = sceKernelCreateThread("CallbackThread", callback_thread, 0x11, 0x1000, 0, NULL);
    if (callback < 0 || sceKernelStartThread(callback, 0, NULL) < 0) fatal_error("Exit callback failed");
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_DIGITAL);
    cpu_mhz = scePowerGetCpuClockFrequencyFloat();
    bus_mhz = scePowerGetBusClockFrequencyFloat();
    if (argc < 1 || !argv[0]) fatal_error("Executable path missing");
    if (strncmp(argv[0], "disc0:", 6) == 0 || strncmp(argv[0], "umd0:", 5) == 0)
        snprintf(dataset_path, sizeof(dataset_path), "disc0:/PSP_GAME/USRDIR/BENCH.DAT");
    else {
        const char *slash = strrchr(argv[0], '/');
        if (!slash || slash - argv[0] > 450) fatal_error("Executable directory invalid");
        snprintf(dataset_path, sizeof(dataset_path), "%.*s/BENCH.DAT", (int)(slash - argv[0]), argv[0]);
    }
    if (!bench_load(&data, dataset_path, error, sizeof(error))) fatal_error(error);
    free_memory_kib = sceKernelTotalFreeMemSize() / 1024;
    printf("[boot] cases=%d dataset_bytes=%u crc32=%08x free_user_kib=%d path=%s custom=%d implementation=%s\n",
           data.case_count, (unsigned)data.size, (unsigned)data.header[13], free_memory_kib,
           dataset_path, bench_custom_available(), BENCH_IMPLEMENTATION_ID);
    while (alive) {
        draw();
        SceCtrlData pad;
        sceCtrlReadBufferPositive(&pad, 1);
        unsigned int pressed = pad.Buttons & ~previous_buttons;
        previous_buttons = pad.Buttons;
        if (pressed & PSP_CTRL_SELECT) alive = 0;
        if (pressed & PSP_CTRL_LTRIGGER) page = (page + 2) % 3;
        if (pressed & PSP_CTRL_RTRIGGER) page = (page + 1) % 3;
        if (pressed & PSP_CTRL_UP) {
            if (page == 1) setting_row = (setting_row + 5) % 6;
            else selected = (selected + data.case_count - 1) % data.case_count;
        }
        if (pressed & PSP_CTRL_DOWN) {
            if (page == 1) setting_row = (setting_row + 1) % 6;
            else selected = (selected + 1) % data.case_count;
        }
        if (page == 1 && (pressed & PSP_CTRL_LEFT)) edit_setting(-1);
        if (page == 1 && (pressed & PSP_CTRL_RIGHT)) edit_setting(1);
        if (pressed & PSP_CTRL_CROSS) { if (run_case()) export_results(); }
        if (pressed & PSP_CTRL_START) {
            int finished = 1, completed = 0, last_completed = selected;
            for (selected = 0; selected < data.case_count && alive; ++selected) {
                if (data.cases[selected].full && !settings.full_suite) continue;
                if (!run_case()) { finished = 0; break; }
                last_completed = selected;
                ++completed;
            }
            if (selected >= data.case_count) selected = last_completed;
            if (alive) export_results();
            if (finished) {
                snprintf(status_text, sizeof(status_text), "Suite complete / CSV saved");
                printf("[suite-complete] cases=%d scope=%s\n", completed, settings.full_suite ? "FULL" : "QUICK");
            }
        }
        if (pressed & PSP_CTRL_TRIANGLE) export_results();
        if (pressed & PSP_CTRL_SQUARE) {
            char screenshot_path[128];
            ensure_directory("ms0:/PSP");
            ensure_directory("ms0:/PSP/SAVEDATA");
            ensure_directory("ms0:/PSP/SAVEDATA/DOTQ4BENCH");
            snprintf(screenshot_path, sizeof(screenshot_path),
                     "ms0:/PSP/SAVEDATA/DOTQ4BENCH/screen-%03u.bmp", screenshot_sequence++);
            if (pspScreenshotSave(screenshot_path) != 0) fatal_error("Screenshot failed");
            snprintf(status_text, sizeof(status_text), "Screenshot saved / PSP/SAVEDATA/DOTQ4BENCH");
            printf("[screenshot] %s case=%d valid=%d\n", screenshot_path, selected, comparisons[selected].valid);
        }
        scePowerTick(PSP_POWER_TICK_ALL);
    }
    bench_free(&data);
    free(background);
    printf("[exit]\n");
    sceKernelExitGame();
    return 0;
}
