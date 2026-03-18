#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Error limits grouped by quantization family.
constexpr float MAX_QUANTIZATION_TOTAL_ERROR             = 0.0030f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_4BITS       = 0.0100f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS       = 0.0200f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS       = 0.0350f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY     = 0.0400f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_1BIT        = 0.0450f;

constexpr float MAX_DOT_PRODUCT_ERROR                    = 0.0200f;
constexpr float MAX_DOT_PRODUCT_ERROR_4BITS              = 0.0250f;
constexpr float MAX_DOT_PRODUCT_ERROR_3BITS              = 0.0450f;
constexpr float MAX_DOT_PRODUCT_ERROR_2BITS              = 0.0500f;
constexpr float MAX_DOT_PRODUCT_ERROR_1BIT               = 0.0950f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY            = 0.1500f;

static const char* RESULT_STR[] = {"PASSED", "FAILED"};

static void print_usage(const char * argv0) {
    printf("Usage:\n");
    printf("  %s --help\n", argv0);
    printf("  %s --list\n", argv0);
    printf("  %s --size N [--size N2 ...] [--sweep start,end,step]\n", argv0);
    printf("  %s --type <TYPE> --size N [--size N2 ...] [--sweep start,end,step]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --help, -h         Show this help\n");
    printf("  --list             List quantizable GGML types\n");
    printf("  --all              Test all quantizable types (optional)\n");
    printf("  --type <TYPE>      Test only the given type name (e.g. iq2_xxs)\n");
    printf("  --size N           Test vector size (repeatable; required for runs)\n");
    printf("  --sweep a,b,c      Add sizes from start,end,step (required for runs if no --size)\n");
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

static bool is_quantizable_type(ggml_type type) {
    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
    if (qfns == NULL || qfns_cpu == NULL) {
        return false;
    }
    return qfns->blck_size > 0 && qfns->to_float != NULL && qfns_cpu->vec_dot != NULL;
}

static void print_quantizable_type_list() {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type type = (ggml_type) i;
        const char * name = ggml_type_name(type);
        if (name == NULL) {
            continue;
        }
        if (!is_quantizable_type(type)) {
            continue;
        }
        printf("%s\n", name);
    }
}

static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) dst[i] = 0.1 + 2*cosf(i + offset);
}

static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) { double diff = a1[i] - a2[i]; sum += diff * diff; }
    return sqrtf(sum) / n;
}

static void quantize_with_fallback(
        ggml_type type,
        const ggml_type_traits_cpu * qfns_cpu,
        const float * src, void * dst, size_t n) {
    if (qfns_cpu->from_float != NULL) {
        qfns_cpu->from_float(src, dst, n);
    } else {
        // IQ paths require an importance matrix.
        std::vector<float> imatrix(n, 1.0f);
        ggml_quantize_chunk(type, src, dst, 0, 1, n, imatrix.data());
    }
}

static float total_quantization_error(
        ggml_type type,
        const ggml_type_traits * qfns,
        const ggml_type_traits_cpu * qfns_cpu,
        size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size + 256);
    std::vector<float> tmp_out(test_size);

    quantize_with_fallback(type, qfns_cpu, test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    return array_rmse(test_data, tmp_out.data(), test_size);
}

static float dot_product_error(
        ggml_type type,
        const ggml_type_traits * qfns,
        const ggml_type_traits_cpu * qfns_cpu,
        size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);
    std::vector<uint8_t> tmp_q1(2*test_size + 256);
    std::vector<uint8_t> tmp_q2(2*test_size + 256);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    quantize_with_fallback(type, qfns_cpu, test_data1, tmp_q1.data(), test_size);

    if (vdot->from_float != NULL) {
        vdot->from_float(test_data2, tmp_q2.data(), test_size);
    } else {
        std::vector<float> imatrix(test_size, 1.0f);
        ggml_quantize_chunk(qfns_cpu->vec_dot_type, test_data2, tmp_q2.data(),
                            0, 1, test_size, imatrix.data());
    }

    float result = 0;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    double sum = 0;
    for (size_t i = 0; i < test_size; i++) sum += (double)test_data1[i] * test_data2[i];

    return fabsf(result - (float)sum) / test_size;
}

int main(int argc, char * argv[]) {
    std::vector<size_t> test_sizes;
    std::string target_type = "";
    bool run_all = false;
    bool list_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--list") {
            list_only = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--size") {
            if (++i < argc) {
                size_t n = 0;
                if (!parse_size(argv[i], n) || n == 0) {
                    fprintf(stderr, "Error: invalid --size value\n");
                    print_usage(argv[0]);
                    return 2;
                }
                test_sizes.push_back(n);
            } else {
                fprintf(stderr, "Error: --size requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--sweep") {
            if (++i < argc) {
                size_t start, end, step;
                if (sscanf(argv[i], "%zu,%zu,%zu", &start, &end, &step) == 3 && step != 0) {
                    if (start == 0 || end == 0 || start > end) {
                        fprintf(stderr, "Error: invalid --sweep range\n");
                        print_usage(argv[0]);
                        return 2;
                    }
                    for (size_t s = start; s <= end; s += step) {
                        test_sizes.push_back(s);
                        if (end - s < step) {
                            break; // avoid size_t overflow
                        }
                    }
                } else {
                    fprintf(stderr, "Error: invalid --sweep value (expected start,end,step)\n");
                    print_usage(argv[0]);
                    return 2;
                }
            } else {
                fprintf(stderr, "Error: --sweep requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--type") {
            if (++i < argc) {
                target_type = argv[i];
            } else {
                fprintf(stderr, "Error: --type requires a value\n");
                print_usage(argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    ggml_cpu_init();

    if (list_only) {
        print_quantizable_type_list();
        return 0;
    }

    if (run_all && !target_type.empty()) {
        fprintf(stderr, "Error: specify either --all or --type, not both\n");
        print_usage(argv[0]);
        return 2;
    }
    if (!run_all && target_type.empty()) {
        run_all = true;
    }

    if (test_sizes.empty()) {
        fprintf(stderr, "Error: at least one --size (or --sweep) is required\n");
        print_usage(argv[0]);
        return 2;
    }

    if (!target_type.empty()) {
        bool exists = false;
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            const char * name = ggml_type_name((ggml_type)i);
            if (name && target_type == name) { exists = true; break; }
        }
        if (!exists) {
            fprintf(stderr, "Error: Type '%s' does not exist in GGML.\n", target_type.c_str());
            return 1;
        }
    }

    std::sort(test_sizes.begin(), test_sizes.end());
    test_sizes.erase(std::unique(test_sizes.begin(), test_sizes.end()), test_sizes.end());

    size_t max_size = test_sizes.back();
    std::vector<float> test_data(max_size);
    std::vector<float> test_data2(max_size);
    generate_data(0.0, max_size, test_data.data());
    generate_data(1.0, max_size, test_data2.data());

    int total_failures = 0;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const char * type_name = ggml_type_name(type);

        if (type_name == NULL) continue;
        if (!target_type.empty() && target_type != type_name) continue;

        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        bool has_from_float = (qfns_cpu->from_float != NULL);
        bool has_quantize_chunk = (qfns->blck_size > 0);

        if (qfns->blck_size == 0 || qfns->to_float == NULL || qfns_cpu->vec_dot == NULL) {
            if (!target_type.empty()) {
                fprintf(stderr, "Error: Type '%s' is not a quantizable type.\n", type_name);
                return 1;
            }
            continue;
        }

        if (!has_from_float && !has_quantize_chunk) {
            if (!target_type.empty()) {
                fprintf(stderr, "Error: Type '%s' has no quantization path.\n", type_name);
                return 1;
            }
            continue;
        }

        ggml_quantize_init(type);

        ggml_quantize_init(qfns_cpu->vec_dot_type);

        printf("%s\n", type_name);

        for (size_t current_size : test_sizes) {
            printf("  Size: %zu\n", current_size);

            float err = total_quantization_error(type, qfns, qfns_cpu, current_size, test_data.data());
            float dot_err = dot_product_error(type, qfns, qfns_cpu, current_size, test_data.data(), test_data2.data());

            float max_err = MAX_QUANTIZATION_TOTAL_ERROR;
            float max_dot = MAX_DOT_PRODUCT_ERROR;

            if (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
                max_err = MAX_QUANTIZATION_TOTAL_ERROR_1BIT;
                max_dot = MAX_DOT_PRODUCT_ERROR_1BIT;
            } else if (type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) {
                max_err = MAX_QUANTIZATION_TOTAL_ERROR_TERNARY;
                max_dot = MAX_DOT_PRODUCT_ERROR_TERNARY;
            } else if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS) {
                max_err = MAX_QUANTIZATION_TOTAL_ERROR_2BITS;
                max_dot = MAX_DOT_PRODUCT_ERROR_2BITS;
            } else if (type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
                max_err = MAX_QUANTIZATION_TOTAL_ERROR_3BITS;
                max_dot = MAX_DOT_PRODUCT_ERROR_3BITS;
            } else if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q4_K ||
                       type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS ||
                       type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) {
                max_err = MAX_QUANTIZATION_TOTAL_ERROR_4BITS;
                max_dot = MAX_DOT_PRODUCT_ERROR_4BITS;
            }

            bool q_failed = !(err < max_err);
            bool d_failed = !(dot_err < max_dot);

            printf("              Quantization: %s (err: %f, limit: %f)\n", RESULT_STR[q_failed], err, max_err);
            printf("              Dot Product:  %s (err: %f, limit: %f)\n", RESULT_STR[d_failed], dot_err, max_dot);

            if (q_failed || d_failed) total_failures++;
        }
        printf("\n");
    }

    if (total_failures > 0) {
        printf("Tests Complete. Total Failures: %d\n", total_failures);
    } else {
        printf("Tests Complete. All PASSED\n");
    }

    return total_failures > 0;
}
