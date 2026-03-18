#include "ggml-cpu.h"
#include "ggml.h"
#include "vec.h"

#undef NDEBUG
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define DOT_F16_ERROR_THRESHOLD   0.00001f
#define DOT_BF16_ERROR_THRESHOLD  0.00010f
#define MAD_F16_ERROR_THRESHOLD   0.00500f
#define SCALE_F16_ERROR_THRESHOLD 0.00250f
#define DOT_F16_UNROLL_THRESHOLD  0.00001f

#define CPU_BF16_TO_FP32_THRESHOLD 0
#define CPU_FP16_TO_FP32_THRESHOLD 0

#define SILU_F32_THRESHOLD 0.00001f

template <typename T> float to_float(T x);
template <> float to_float(float x) {
    return x;
}
template <> float to_float(ggml_fp16_t x) {
    return ggml_fp16_to_fp32(x);
}
template <> float to_float(ggml_bf16_t x) {
    return ggml_bf16_to_fp32(x);
}

template <typename T> T from_float(float x);
template <> float from_float(float x) {
    return x;
}

template <> ggml_fp16_t from_float(float x) {
    return ggml_fp32_to_fp16(x);
}

template <> ggml_bf16_t from_float(float x) {
    return ggml_fp32_to_bf16(x);
}

static constexpr uint32_t DEFAULT_SEED = 0xC0FFEEu;

template <typename T>
static std::vector<T> generate_cosine(size_t n, std::mt19937 & rng) {
    std::vector<T> data(n);
    constexpr float kPi = 3.14159265358979323846f;
    std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * kPi);
    const float phase = phase_dist(rng);
    for (size_t i = 0; i < n; ++i) {
        const float val = 2.0f * std::cos(i * 0.1f + phase) - 0.1f;
        data[i] = from_float<T>(val);
    }
    return data;
}

template <typename T>
static float max_abs_diff(size_t n, const T * ref, const T * opt) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float err = std::fabs(to_float(ref[i]) - to_float(opt[i]));
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Scalar reference implementations.
static inline void ggml_vec_mad_f16_reference(const int                         n,
                                              ggml_fp16_t * GGML_RESTRICT       y,
                                              const ggml_fp16_t * GGML_RESTRICT x,
                                              const float                       v) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) + ggml_fp16_to_fp32(x[i]) * v);
    }
}

static void ggml_vec_mad_f32_reference(const int                   n,
                                       float * GGML_RESTRICT       y,
                                       const float * GGML_RESTRICT x,
                                       const float                 v) {
    for (int i = 0; i < n; ++i) {
        y[i] += x[i] * v;
    }
}

static void ggml_vec_dot_bf16_reference(int                         n,
                                        float * GGML_RESTRICT       s,
                                        size_t                      bs,
                                        ggml_bf16_t * GGML_RESTRICT x,
                                        size_t                      bx,
                                        ggml_bf16_t * GGML_RESTRICT y,
                                        size_t                      by,
                                        int                         nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int        i    = 0;
    ggml_float sumf = 0;
    for (; i < n; ++i) {
        sumf += (ggml_float) (ggml_bf16_to_fp32(x[i]) * ggml_bf16_to_fp32(y[i]));
    }
    *s = sumf;
}

static inline void ggml_vec_scale_f16_reference(const int n, ggml_fp16_t * y, const float v) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) * v);
    }
}

static void ggml_vec_dot_f16_reference(int                         n,
                                       float * GGML_RESTRICT       s,
                                       size_t                      bs,
                                       ggml_fp16_t * GGML_RESTRICT x,
                                       size_t                      bx,
                                       ggml_fp16_t * GGML_RESTRICT y,
                                       size_t                      by,
                                       int                         nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;

    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (ggml_fp16_to_fp32(x[i]) * ggml_fp16_to_fp32(y[i]));
    }

    *s = sumf;
}

inline static void ggml_vec_dot_f16_unroll_reference(const int                   n,
                                                     const int                   xs,
                                                     float * GGML_RESTRICT       s,
                                                     void * GGML_RESTRICT        xv,
                                                     ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i * xs);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float) (ggml_fp16_to_fp32(x[j][i]) * ggml_fp16_to_fp32(y[i]));
        }
    }

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float) sumf[i];
    }
}

static void ggml_vec_silu_f32_reference(const int n, float * y, const float * x) {
    int i = 0;
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

static void ggml_cpu_bf16_to_fp32_reference(const ggml_bf16_t * x, float * y, int64_t n) {
    int i = 0;
    for (; i < n; i++) {
        y[i] = ggml_bf16_to_fp32(x[i]);
    }
}

static void ggml_cpu_fp16_to_fp32_reference(const ggml_fp16_t * x, float * y, int64_t n) {
    int i = 0;
    for (; i < n; i++) {
        y[i] = ggml_fp16_to_fp32(x[i]);
    }
}

static float normalize_by_n(float err, size_t n) {
    return n > 0 ? (err / (float) n) : err;
}

static float run_vec_dot_f16(size_t n, std::mt19937 & rng) {
    auto x = generate_cosine<ggml_fp16_t>(n, rng);
    auto y = generate_cosine<ggml_fp16_t>(n, rng);

    float s_ref = 0.0f;
    float s_opt = 0.0f;
    ggml_vec_dot_f16_reference((int) n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);
    ggml_vec_dot_f16((int) n, &s_opt, 0, x.data(), 0, y.data(), 0, 1);
    return normalize_by_n(std::fabs(s_ref - s_opt), n);
}

static float run_vec_dot_bf16(size_t n, std::mt19937 & rng) {
    auto x = generate_cosine<ggml_bf16_t>(n, rng);
    auto y = generate_cosine<ggml_bf16_t>(n, rng);

    float s_ref = 0.0f;
    float s_opt = 0.0f;

    ggml_vec_dot_bf16_reference((int) n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);
    ggml_vec_dot_bf16((int) n, &s_opt, 0, x.data(), 0, y.data(), 0, 1);
    return normalize_by_n(std::fabs(s_ref - s_opt), n);
}

static float run_vec_scale_f16(size_t n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scalar_dist(-2.0f, 2.0f);
    const float v = scalar_dist(rng);

    auto y_ref = generate_cosine<ggml_fp16_t>(n, rng);
    auto y_opt = y_ref;

    ggml_vec_scale_f16_reference((int) n, y_ref.data(), v);
    ggml_vec_scale_f16((int) n, y_opt.data(), v);

    return max_abs_diff(n, y_ref.data(), y_opt.data());
}

static float run_vec_mad_f16(size_t n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scalar_dist(-2.0f, 2.0f);
    const float v = scalar_dist(rng);

    auto x = generate_cosine<ggml_fp16_t>(n, rng);
    auto y_ref = generate_cosine<ggml_fp16_t>(n, rng);
    auto y_opt = y_ref;

    ggml_vec_mad_f16_reference((int) n, y_ref.data(), x.data(), v);
    ggml_vec_mad_f16((int) n, y_opt.data(), x.data(), v);
    return max_abs_diff(n, y_ref.data(), y_opt.data());
}

static float run_vec_dot_f16_unroll(size_t n, std::mt19937 & rng) {
    const int xs = (int) (n * sizeof(ggml_fp16_t));
    std::vector<char> xv_data((size_t) xs * GGML_VEC_DOT_UNROLL);
    void * xv = xv_data.data();

    auto y = generate_cosine<ggml_fp16_t>(n, rng);

    for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
        ggml_fp16_t * xv_k = (ggml_fp16_t *) ((char *) xv + (size_t) k * (size_t) xs);
        auto tmp = generate_cosine<ggml_fp16_t>(n, rng);
        memcpy(xv_k, tmp.data(), n * sizeof(ggml_fp16_t));
    }

    float s_ref[GGML_VEC_DOT_UNROLL];
    float s_opt[GGML_VEC_DOT_UNROLL];

    ggml_vec_dot_f16_unroll_reference((int) n, xs, s_ref, xv, (ggml_fp16_t *) y.data());
    ggml_vec_dot_f16_unroll((int) n, xs, s_opt, xv, (ggml_fp16_t *) y.data());

    float max_err = 0.0f;
    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        const float err = std::fabs(s_ref[i] - s_opt[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return normalize_by_n(max_err, n);
}

static float run_cpu_bf16_to_fp32(size_t n, std::mt19937 & rng) {
    auto x = generate_cosine<ggml_bf16_t>(n, rng);
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_cpu_bf16_to_fp32_reference(x.data(), y_ref.data(), (int64_t) n);
    ggml_cpu_bf16_to_fp32(x.data(), y_opt.data(), (int64_t) n);

    return max_abs_diff(n, y_ref.data(), y_opt.data());
}

static float run_cpu_fp16_to_fp32(size_t n, std::mt19937 & rng) {
    auto x = generate_cosine<ggml_fp16_t>(n, rng);
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_cpu_fp16_to_fp32_reference(x.data(), y_ref.data(), (int64_t) n);
    ggml_cpu_fp16_to_fp32(x.data(), y_opt.data(), (int64_t) n);
    return max_abs_diff(n, y_ref.data(), y_opt.data());
}

static float run_vec_silu_f32(size_t n, std::mt19937 & rng) {
    auto x = generate_cosine<float>(n, rng);
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_vec_silu_f32_reference((int) n, y_ref.data(), x.data());
    ggml_vec_silu_f32((int) n, y_opt.data(), x.data());
    return max_abs_diff(n, y_ref.data(), y_opt.data());
}

struct KernelTest {
    const char * name;
    float        limit;
    float      (*run)(size_t n, std::mt19937 & rng);
};

static const KernelTest KERNELS[] = {
    { "ggml_vec_dot_f16",        DOT_F16_ERROR_THRESHOLD,        run_vec_dot_f16        },
    { "ggml_vec_dot_bf16",       DOT_BF16_ERROR_THRESHOLD,       run_vec_dot_bf16       },
    { "ggml_vec_scale_f16",      SCALE_F16_ERROR_THRESHOLD,      run_vec_scale_f16      },
    { "ggml_vec_mad_f16",        MAD_F16_ERROR_THRESHOLD,        run_vec_mad_f16        },
    { "ggml_vec_dot_f16_unroll", DOT_F16_UNROLL_THRESHOLD,       run_vec_dot_f16_unroll },
    { "ggml_cpu_bf16_to_fp32",   CPU_BF16_TO_FP32_THRESHOLD,     run_cpu_bf16_to_fp32   },
    { "ggml_cpu_fp16_to_fp32",   CPU_FP16_TO_FP32_THRESHOLD,     run_cpu_fp16_to_fp32   },
    { "ggml_vec_silu_f32",       SILU_F32_THRESHOLD,             run_vec_silu_f32       },
};

static void print_usage(const char * argv0) {
    printf("Usage:\n");
    printf("  %s --list\n", argv0);
    printf("  %s --size N [--size N ...] [--seed S]\n", argv0);
    printf("  %s --fn <name> [--fn <name> ...] --size N [--size N ...] [--seed S]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --list         List supported function names and exit\n");
    printf("  --all          Test all supported functions (optional)\n");
    printf("  --fn <name>    Test a specific function (repeatable)\n");
    printf("  --size N       Vector size (required, repeatable)\n");
    printf("  --seed S       RNG seed override (default: 0x%08X)\n", DEFAULT_SEED);
}

static void print_supported_fns() {
    for (const auto & k : KERNELS) {
        printf("%s\n", k.name);
    }
}

static const KernelTest * find_kernel(const std::string & name) {
    for (const auto & k : KERNELS) {
        if (name == k.name) {
            return &k;
        }
    }
    return nullptr;
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

static bool parse_u32(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long v = strtoul(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();

    bool list_only = false;
    bool run_all = false;
    std::vector<size_t> sizes;
    uint32_t seed = DEFAULT_SEED;
    std::vector<const KernelTest *> selected;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --size requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            size_t parsed_size = 0;
            if (!parse_size(argv[++i], parsed_size) || parsed_size == 0) {
                fprintf(stderr, "Error: invalid --size value\n");
                print_usage(argv[0]);
                return 2;
            }
            sizes.push_back(parsed_size);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --seed requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            if (!parse_u32(argv[++i], seed)) {
                fprintf(stderr, "Error: invalid --seed value\n");
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--fn") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --fn requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
            const std::string name = argv[++i];
            const KernelTest * k = find_kernel(name);
            if (!k) {
                fprintf(stderr, "Error: unknown function '%s'\n\n", name.c_str());
                printf("Supported functions:\n");
                print_supported_fns();
                return 2;
            }
            selected.push_back(k);
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
        print_supported_fns();
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
        fprintf(stderr, "Error: at least one --size is required and each must be > 0\n");
        print_usage(argv[0]);
        return 2;
    }

    if (run_all) {
        selected.clear();
        for (const auto & k : KERNELS) {
            selected.push_back(&k);
        }
    }

    printf("Executing Kernel Tests.\n");
    printf("Seed: %u\n\n", seed);

    std::mt19937 rng(seed);
    int failures = 0;
    for (size_t size : sizes) {
        printf("Size: %zu\n", size);
        printf("\n");

        for (const auto * k : selected) {
            const float err = k->run(size, rng);
            const bool pass = err <= k->limit;
            printf("%-24s : %s (err: %g, limit: %g)\n", k->name, pass ? "PASSED" : "FAILED", err, k->limit);
            if (!pass) {
                failures++;
            }
        }

        printf("\n");
    }

    printf("\nTotal Failures: %d\n", failures);
    return failures;
}
