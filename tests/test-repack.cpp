#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <string>
#include <vector>

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

static bool parse_i64_csv(const char * s, std::vector<int64_t> & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }

    std::string token;
    for (const char * p = s; ; ++p) {
        const char ch = *p;
        if (ch == ',' || ch == '\0') {
            if (token.empty()) {
                return false;
            }
            int64_t v = 0;
            if (!parse_i64(token.c_str(), v) || v <= 0) {
                return false;
            }
            out.push_back(v);
            token.clear();
            if (ch == '\0') {
                break;
            }
        } else {
            token.push_back(ch);
        }
    }

    return true;
}

static void print_usage(const char * argv0) {
    printf("Usage:\n");
    printf("  %s [--m M[,M2..]] [--n N[,N2..]] [--k K[,K2..]] [--threads T]\n", argv0);
    printf("  %s --type <TYPE> [--m M[,M2..]] [--n N[,N2..]] [--k K[,K2..]] [--threads T]\n", argv0);
    printf("  %s --list\n", argv0);
}

// Quantized types we can execute end-to-end on CPU.
static bool is_quantizable_type(ggml_type type) {
    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
    if (qfns == nullptr || qfns_cpu == nullptr) {
        return false;
    }
    return qfns->blck_size > 0 && qfns->to_float != nullptr && qfns_cpu->vec_dot != nullptr;
}

static void list_types(void) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * name = ggml_type_name(type);
        if (name == nullptr) {
            continue;
        }
        if (!is_quantizable_type(type)) {
            continue;
        }
        printf("%s\n", name);
    }
}

static void fill_data(float * dst, int64_t rows, int64_t cols, float seed) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const float v = 0.1f + 2.0f * cosf((float) c * 0.17f + (float) r * 0.11f + seed);
            dst[r * cols + c] = v;
        }
    }
}

static void reference_matmul(const float * act, const float * w, float * out, int64_t M, int64_t N, int64_t K) {
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += act[r * K + k] * w[c * K + k];
            }
            out[r * N + c] = sum;
        }
    }
}

static bool quantize_row_with_fallback(
        ggml_type type,
        const ggml_type_traits_cpu * qfns_cpu,
        const float * src,
        void * dst,
        int64_t K,
        size_t row_size) {
    if (qfns_cpu->from_float != nullptr) {
        qfns_cpu->from_float(src, (uint8_t *) dst, K);
        return true;
    }

    std::vector<float> imatrix((size_t) K, 1.0f);
    const size_t n_written = ggml_quantize_chunk(type, src, (uint8_t *) dst, 0, 1, K, imatrix.data());
    return n_written == row_size;
}

static bool quantize_rows_with_fallback(
        ggml_type type,
        const ggml_type_traits_cpu * qfns_cpu,
        const float * src,
        void * dst,
        int64_t rows,
        int64_t K,
        size_t row_size) {
    for (int64_t r = 0; r < rows; ++r) {
        if (!quantize_row_with_fallback(type, qfns_cpu, src + r*K, (uint8_t *) dst + r*row_size, K, row_size)) {
            return false;
        }
    }
    return true;
}

static bool reference_matmul_vec_dot(
        ggml_type type,
        const ggml_type_traits_cpu * qfns_cpu,
        const uint8_t * weights_q,
        const float * acts_f,
        float * out,
        int64_t M,
        int64_t N,
        int64_t K,
        size_t w_row_size) {
    const ggml_type vdot_type = qfns_cpu->vec_dot_type;
    const auto * vdot_traits = ggml_get_type_traits_cpu(vdot_type);
    if (vdot_traits == nullptr) {
        return false;
    }

    const size_t vdot_row_size = ggml_row_size(vdot_type, K);
    std::vector<uint8_t> act_q((size_t) M * vdot_row_size);
    if (!quantize_rows_with_fallback(vdot_type, vdot_traits, acts_f, act_q.data(), M, K, vdot_row_size)) {
        return false;
    }

    GGML_UNUSED(type);
    for (int64_t r = 0; r < M; ++r) {
        const uint8_t * act_row = act_q.data() + r * vdot_row_size;
        for (int64_t c = 0; c < N; ++c) {
            const uint8_t * w_row = weights_q + c * w_row_size;
            float sum = 0.0f;
            qfns_cpu->vec_dot(K, &sum, 0, w_row, 0, act_row, 0, 1);
            out[r * N + c] = sum;
        }
    }
    return true;
}

static int run_one_type(ggml_type type, int64_t M, int64_t N, int64_t K, int n_threads) {
    const char * type_name = ggml_type_name(type);
    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!is_quantizable_type(type)) {
        return 0;
    }

    if (K <= 0 || N <= 0 || M <= 0 || (K % qfns->blck_size) != 0) {
        printf("[M=%lld K=%lld N=%lld] %s: SKIP (invalid shape for block size %lld)\n",
               (long long) M, (long long) K, (long long) N, type_name, (long long) qfns->blck_size);
        return 0;
    }

    std::vector<float> weights_f((size_t) N * K);
    std::vector<float> acts_f((size_t) M * K);
    std::vector<float> ref_f32((size_t) M * N);
    std::vector<float> ref_vecdot((size_t) M * N);
    std::vector<float> out((size_t) M * N);

    fill_data(weights_f.data(), N, K, 0.0f);
    fill_data(acts_f.data(), M, K, 1.0f);
    reference_matmul(acts_f.data(), weights_f.data(), ref_f32.data(), M, N, K);

    ggml_quantize_init(type);
    ggml_quantize_init(qfns_cpu->vec_dot_type);

    const size_t row_size = ggml_row_size(type, K);
    const size_t w_bytes = row_size * (size_t) N;
    std::vector<uint8_t> weights_q(w_bytes);

    if (!quantize_rows_with_fallback(type, qfns_cpu, weights_f.data(), weights_q.data(), N, K, row_size)) {
        printf("[M=%lld K=%lld N=%lld] %s: FAIL (quantization failed)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        return 1;
    }

    if (!reference_matmul_vec_dot(type, qfns_cpu, weights_q.data(), acts_f.data(), ref_vecdot.data(), M, N, K, row_size)) {
        printf("[M=%lld K=%lld N=%lld] %s: FAIL (vec-dot reference failed)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        printf("[M=%lld K=%lld N=%lld] %s: FAIL (ggml_init)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        return 1;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, a);

    ggml_backend_buffer_type_t repack_bt = ggml_backend_cpu_repack_buffer_type();
    if (repack_bt == nullptr) {
        printf("[M=%lld K=%lld N=%lld] %s: SKIP (no repack buffer type)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        ggml_free(ctx);
        return 0;
    }

    ggml_backend_buffer_t w_buf = repack_bt->iface.alloc_buffer(repack_bt, ggml_nbytes(w));
    if (w_buf == nullptr) {
        printf("[M=%lld K=%lld N=%lld] %s: SKIP (repack buffer alloc failed)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        ggml_free(ctx);
        return 0;
    }

    w->buffer = w_buf;
    w->data = w_buf->context;
    w_buf->iface.init_tensor(w_buf, w);

    // No repack kernel for this type/shape on this CPU.
    if (w->extra == nullptr) {
        w_buf->iface.free_buffer(w_buf);
        ggml_free(ctx);
        return 0;
    }

    w_buf->iface.set_tensor(w_buf, w, weights_q.data(), 0, w_bytes);

    ggml_backend_buffer_t host_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (host_buf == nullptr) {
        printf("[M=%lld K=%lld N=%lld] %s: FAIL (host buffer alloc)\n",
               (long long) M, (long long) K, (long long) N, type_name);
        w_buf->iface.free_buffer(w_buf);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_set(a, acts_f.data(), 0, (size_t) M * K * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        printf("[M=%lld K=%lld N=%lld] %s: FAIL (compute failed: %s)\n",
               (long long) M, (long long) K, (long long) N, type_name, ggml_status_to_string(st));
        ggml_backend_buffer_free(host_buf);
        w_buf->iface.free_buffer(w_buf);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_get(y, out.data(), 0, (size_t) M * N * sizeof(float));

    float max_norm_abs_err = 0.0f;
    int vecdot_mismatch = 0;
    float max_vecdot_abs_err = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        const float e_float = fabsf(out[(size_t) i] - ref_f32[(size_t) i]) / (float) K;
        max_norm_abs_err = std::max(max_norm_abs_err, e_float);

        const float e_vecdot = fabsf(out[(size_t) i] - ref_vecdot[(size_t) i]) ;
        max_vecdot_abs_err = std::max(max_vecdot_abs_err, e_vecdot);
        if (fabs(out[(size_t) i] - ref_vecdot[(size_t) i]) > 0.00007f) {
            ++vecdot_mismatch;
        }
    }

    printf("[M=%lld K=%lld N=%lld] %s: %s (vecdot_ref_error=%f, float_ref_error=%f)\n",
           (long long) M, (long long) K, (long long) N,
           type_name,
           vecdot_mismatch == 0 ? "PASS" : "FAIL",
           max_vecdot_abs_err,
           max_norm_abs_err);

    ggml_backend_buffer_free(host_buf);
    w_buf->iface.free_buffer(w_buf);
    ggml_free(ctx);

    return vecdot_mismatch == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    bool list_only = false;
    bool run_all = false;
    std::string target_type;

    std::vector<int64_t> Ms = { 4 };
    std::vector<int64_t> Ns = { 16 };
    std::vector<int64_t> Ks = { 256 };
    bool seen_m = false;
    bool seen_n = false;
    bool seen_k = false;
    int64_t threads = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--type") {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            target_type = argv[i];
        } else if (arg == "--m") {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            if (!seen_m) {
                Ms.clear();
                seen_m = true;
            }
            if (!parse_i64_csv(argv[i], Ms)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--n") {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            if (!seen_n) {
                Ns.clear();
                seen_n = true;
            }
            if (!parse_i64_csv(argv[i], Ns)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--k") {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            if (!seen_k) {
                Ks.clear();
                seen_k = true;
            }
            if (!parse_i64_csv(argv[i], Ks)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--threads") {
            if (++i >= argc || !parse_i64(argv[i], threads) || threads <= 0) {
                print_usage(argv[0]);
                return 2;
            }
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    ggml_log_set(
        [](enum ggml_log_level level, const char * text, void * user_data) {
            (void) user_data;
            if (level == GGML_LOG_LEVEL_DEBUG) {
                return;
            }
            fputs(text, stderr);
            fflush(stderr);
        },
        nullptr);

    ggml_cpu_init();

    if (list_only) {
        list_types();
        return 0;
    }

    if (run_all && !target_type.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    if (!run_all && target_type.empty()) {
        run_all = true;
    }

    auto dedup_keep_order = [](std::vector<int64_t> & values) {
        std::vector<int64_t> unique;
        unique.reserve(values.size());
        for (const int64_t v : values) {
            if (std::find(unique.begin(), unique.end(), v) == unique.end()) {
                unique.push_back(v);
            }
        }
        values.swap(unique);
    };

    dedup_keep_order(Ms);
    dedup_keep_order(Ns);
    dedup_keep_order(Ks);

    int failures = 0;
    for (const int64_t M : Ms) {
        for (const int64_t N : Ns) {
            for (const int64_t K : Ks) {
                if (!target_type.empty()) {
                    bool found = false;
                    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                        const ggml_type type = (ggml_type) i;
                        const char * name = ggml_type_name(type);
                        if (name != nullptr && target_type == name) {
                            failures += run_one_type(type, M, N, K, (int) threads);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        fprintf(stderr, "repacking not supported: %s\n", target_type.c_str());
                        return 2;
                    }
                } else {
                    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                        const ggml_type type = (ggml_type) i;
                        const char * name = ggml_type_name(type);
                        if (name == nullptr) {
                            continue;
                        }
                        if (!is_quantizable_type(type)) {
                            continue;
                        }
                        failures += run_one_type(type, M, N, K, (int) threads);
                    }
                }
            }
        }
    }

    return failures > 0 ? 1 : 0;
}
