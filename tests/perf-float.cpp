#include "ggml-cpu.h"
#include "ggml.h"
#include "vec.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>

static constexpr int      WARMUP_ITERS     = 10;
static constexpr int64_t  DEFAULT_ITERS    = 200;

template <typename T> static void generate_vector(T * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        const float f_val = 2.0f * std::cos(i * 0.1f) - 0.1f;
        if constexpr (std::is_same_v<T, float>) {
            data[i] = f_val;
        } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
            data[i] = ggml_fp32_to_fp16(f_val);
        } else if constexpr (std::is_same_v<T, ggml_bf16_t>) {
            data[i] = ggml_fp32_to_bf16(f_val);
        } else {
            static_assert(!sizeof(T *), "Unsupported type for generate_vector");
        }
    }
}

template <typename T>
struct AlignedBuffer {
    T *    ptr   = nullptr;
    size_t bytes = 0;

    AlignedBuffer() = default;

    explicit AlignedBuffer(size_t n) {
        bytes = n * sizeof(T);
        ptr = (T *) ggml_aligned_malloc(bytes);
    }

    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer & operator=(const AlignedBuffer &) = delete;

    AlignedBuffer(AlignedBuffer && other) noexcept {
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }

    AlignedBuffer & operator=(AlignedBuffer && other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~AlignedBuffer() {
        reset();
    }

    void reset() {
        if (ptr != nullptr) {
            ggml_aligned_free(ptr, bytes);
            ptr = nullptr;
            bytes = 0;
        }
    }

    T * data() { return ptr; }
    const T * data() const { return ptr; }
};

static inline void do_not_optimize(float v) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(v) : "memory");
#else
    volatile float sink = v;
    (void) sink;
#endif
}

static inline int64_t time_now_ns() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

struct TimeStats {
    double min_ns  = 0.0;
    double mean_ns = 0.0;
    double std_ns  = 0.0;
};

struct Welford {
    int64_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x) {
        n++;
        const double delta = x - mean;
        mean += delta / (double) n;
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance_sample() const {
        return n > 1 ? (m2 / (double) (n - 1)) : 0.0;
    }
};

template <typename Func>
static TimeStats benchmark_kernel_ns(int64_t iters, Func && func) {
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        const float out = func();
        do_not_optimize(out);
    }

    int64_t min_ns = std::numeric_limits<int64_t>::max();
    Welford w;

    for (int64_t i = 0; i < iters; ++i) {
        const int64_t start_ns = time_now_ns();
        const float out = func();
        const int64_t end_ns = time_now_ns();

        do_not_optimize(out);

        const int64_t dt = end_ns - start_ns;
        if (dt < min_ns) {
            min_ns = dt;
        }
        w.add((double) dt);
    }

    TimeStats s;
    s.min_ns  = (double) min_ns;
    s.mean_ns = w.mean;
    s.std_ns  = std::sqrt(w.variance_sample());
    return s;
}

static bool parse_size(const char * s, size_t & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long long v = strtoull(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    if (v > (unsigned long long) SIZE_MAX) {
        return false;
    }
    out = (size_t) v;
    return true;
}

static bool parse_i64(const char * s, int64_t & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    const long long v = strtoll(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = (int64_t) v;
    return true;
}

static void print_usage(const char * argv0) {
    printf("Usage:\n");
    printf("  %s --list\n", argv0);
    printf("  %s --size N [--size N2 ...] [--sweep start,end,step] [--iter I]\n", argv0);
    printf("  %s --fn <name> [--fn <name> ...] --size N [--size N2 ...] [--sweep start,end,step] [--iter I]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --list         List supported kernel names and exit\n");
    printf("  --all          Benchmark all supported kernels (optional)\n");
    printf("  --fn <name>    Benchmark a specific kernel (repeatable)\n");
    printf("  --size N       Vector size (repeatable)\n");
    printf("  --sweep a,b,c  Add sizes from start,end,step\n");
    printf("  --iter I       Measured iterations (default: %lld)\n", (long long) DEFAULT_ITERS);
}

static volatile float g_scale_v = 1.0f;
static volatile float g_mad_v   = 0.0f;

static TimeStats bench_vec_dot_f16(size_t size, int64_t iters) {
    AlignedBuffer<ggml_fp16_t> x(size);
    AlignedBuffer<ggml_fp16_t> y(size);
    generate_vector(x.data(), size);
    generate_vector(y.data(), size);

    float result = 0.0f;
    auto call = [&]() -> float {
        ggml_vec_dot_f16((int) size, &result, 0, x.data(), 0, y.data(), 0, 1);
        return result;
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_vec_dot_bf16(size_t size, int64_t iters) {
    AlignedBuffer<ggml_bf16_t> x(size);
    AlignedBuffer<ggml_bf16_t> y(size);
    generate_vector(x.data(), size);
    generate_vector(y.data(), size);

    float result = 0.0f;
    auto call = [&]() -> float {
        ggml_vec_dot_bf16((int) size, &result, 0, x.data(), 0, y.data(), 0, 1);
        return result;
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_vec_dot_f16_unroll(size_t size, int64_t iters) {
    AlignedBuffer<ggml_fp16_t> xv((size_t) GGML_VEC_DOT_UNROLL * size);
    AlignedBuffer<ggml_fp16_t> y(size);
    generate_vector(xv.data(), (size_t) GGML_VEC_DOT_UNROLL * size);
    generate_vector(y.data(), size);

    const int xs = (int) (size * sizeof(ggml_fp16_t));
    float result[GGML_VEC_DOT_UNROLL] = {};

    auto call = [&]() -> float {
        ggml_vec_dot_f16_unroll((int) size, xs, result, xv.data(), y.data());
        return result[0];
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_vec_scale_f16(size_t size, int64_t iters) {
    AlignedBuffer<ggml_fp16_t> y(size);
    generate_vector(y.data(), size);

    auto call = [&]() -> float {
        const float v = g_scale_v;
        ggml_vec_scale_f16((int) size, y.data(), v);
        return ggml_fp16_to_fp32(y.data()[0]);
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_vec_mad_f16(size_t size, int64_t iters) {
    AlignedBuffer<ggml_fp16_t> x(size);
    AlignedBuffer<ggml_fp16_t> y(size);
    generate_vector(x.data(), size);
    generate_vector(y.data(), size);

    auto call = [&]() -> float {
        const float v = g_mad_v;
        ggml_vec_mad_f16((int) size, y.data(), x.data(), v);
        return ggml_fp16_to_fp32(y.data()[0]);
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_cpu_bf16_to_fp32(size_t size, int64_t iters) {
    AlignedBuffer<ggml_bf16_t> x(size);
    AlignedBuffer<float>       y(size);
    generate_vector(x.data(), size);

    auto call = [&]() -> float {
        ggml_cpu_bf16_to_fp32(x.data(), y.data(), (int64_t) size);
        return y.data()[0];
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_cpu_fp16_to_fp32(size_t size, int64_t iters) {
    AlignedBuffer<ggml_fp16_t> x(size);
    AlignedBuffer<float>       y(size);
    generate_vector(x.data(), size);

    auto call = [&]() -> float {
        ggml_cpu_fp16_to_fp32(x.data(), y.data(), (int64_t) size);
        return y.data()[0];
    };

    return benchmark_kernel_ns(iters, call);
}

static TimeStats bench_vec_silu_f32(size_t size, int64_t iters) {
    AlignedBuffer<float> x(size);
    AlignedBuffer<float> y(size);
    generate_vector(x.data(), size);

    auto call = [&]() -> float {
        ggml_vec_silu_f32((int) size, y.data(), x.data());
        return y.data()[0];
    };

    return benchmark_kernel_ns(iters, call);
}

static int64_t ops_dot(size_t n) {
    return 2 * (int64_t) n;
}

static int64_t ops_dot_unroll(size_t n) {
    return (int64_t) GGML_VEC_DOT_UNROLL * 2 * (int64_t) n;
}

static int64_t ops_scale(size_t n) {
    return (int64_t) n;
}

static int64_t ops_mad(size_t n) {
    return 2 * (int64_t) n;
}

static int64_t ops_convert(size_t n) {
    return (int64_t) n;
}

static int64_t ops_silu(size_t n) {
    return 33 * (int64_t) n;
}

struct KernelBench {
    const char * name;
    int64_t    (*ops_per_call)(size_t size);
    TimeStats (*run)(size_t size, int64_t iters);
};

static const KernelBench KERNELS[] = {
    { "ggml_vec_dot_f16",        ops_dot,       bench_vec_dot_f16        },
    { "ggml_vec_dot_bf16",       ops_dot,       bench_vec_dot_bf16       },
    { "ggml_vec_dot_f16_unroll", ops_dot_unroll,bench_vec_dot_f16_unroll },
    { "ggml_vec_scale_f16",      ops_scale,     bench_vec_scale_f16      },
    { "ggml_vec_mad_f16",        ops_mad,       bench_vec_mad_f16        },
    { "ggml_cpu_bf16_to_fp32",   ops_convert,   bench_cpu_bf16_to_fp32   },
    { "ggml_cpu_fp16_to_fp32",   ops_convert,   bench_cpu_fp16_to_fp32   },
    { "ggml_vec_silu_f32",       ops_silu,      bench_vec_silu_f32       },
};

static void print_supported() {
    for (const auto & k : KERNELS) {
        printf("%s\n", k.name);
    }
}

static const KernelBench * find_kernel(const std::string & name) {
    for (const auto & k : KERNELS) {
        if (name == k.name) {
            return &k;
        }
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();

    bool list_only = false;
    bool run_all = false;
    int64_t iters = DEFAULT_ITERS;
    std::vector<size_t> sizes;
    std::vector<const KernelBench *> selected;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--fn") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --fn requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            const std::string name = argv[++i];
            const KernelBench * k = find_kernel(name);
            if (!k) {
                fprintf(stderr, "Error: unknown kernel '%s'\n\n", name.c_str());
                printf("Supported kernels:\n");
                print_supported();
                return 2;
            }
            selected.push_back(k);
        } else if (arg == "--size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --size requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            size_t n = 0;
            if (!parse_size(argv[++i], n) || n == 0) {
                fprintf(stderr, "Error: invalid --size value\n");
                print_usage(argv[0]);
                return 2;
            }
            sizes.push_back(n);
        } else if (arg == "--sweep") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --sweep requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            size_t start = 0, end = 0, step = 0;
            if (sscanf(argv[++i], "%zu,%zu,%zu", &start, &end, &step) != 3 || step == 0) {
                fprintf(stderr, "Error: invalid --sweep value (expected start,end,step)\n");
                print_usage(argv[0]);
                return 2;
            }
            if (start == 0 || end == 0 || start > end) {
                fprintf(stderr, "Error: invalid --sweep range\n");
                print_usage(argv[0]);
                return 2;
            }
            for (size_t s = start; s <= end; s += step) {
                sizes.push_back(s);
                if (end - s < step) {
                    break; // avoid size_t overflow
                }
            }
        } else if (arg == "--iter") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --iter requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            if (!parse_i64(argv[++i], iters) || iters <= 0) {
                fprintf(stderr, "Error: invalid --iter value\n");
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_only) {
        print_supported();
        return 0;
    }

    if (run_all && !selected.empty()) {
        fprintf(stderr, "Error: specify either --all or --fn, not both\n");
        print_usage(argv[0]);
        return 2;
    }
    if (!run_all && selected.empty()) {
        run_all = true;
    }
    if (sizes.empty()) {
        fprintf(stderr, "Error: at least one --size (or --sweep) is required\n");
        print_usage(argv[0]);
        return 2;
    }

    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    if (run_all) {
        selected.clear();
        for (const auto & k : KERNELS) {
            selected.push_back(&k);
        }
    }

    const char * table_header =
        "|   Size   |      Best ns |       Avg ns | Best M-Ops/s |  Avg M-Ops/s |\n"
        "|:--------:|-------------:|-------------:|-------------:|-------------:|\n";

    printf("Warmup iters: %d\n", WARMUP_ITERS);
    printf("Measured iters: %lld\n", (long long) iters);

    for (const auto * k : selected) {
        printf("\n### Kernel: %s\n", k->name);
        printf("%s", table_header);
        for (const auto size : sizes) {
            const TimeStats s = k->run(size, iters);
            const int64_t ops = k->ops_per_call(size);
            const double best_s = s.min_ns  > 0 ? (s.min_ns  * 1e-9) : 0.0;
            const double avg_s  = s.mean_ns > 0 ? (s.mean_ns * 1e-9) : 0.0;
            const double best_mops = best_s > 0 ? ((double) ops / best_s / 1e6) : 0.0;
            const double avg_mops  = avg_s  > 0 ? ((double) ops / avg_s  / 1e6) : 0.0;
            printf("| %8zu | %12.2f | %12.2f | %12.4f | %12.4f |\n",
                   size, s.min_ns, s.mean_ns, best_mops, avg_mops);
        }
    }

    return 0;
}
