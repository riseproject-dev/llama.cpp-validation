Here's the rewritten README:

---

# llama.cpp RISE: Testing and Benchmarking

RISE provides a focused test and benchmark suite for validating and profiling llama.cpp GGML kernel implementations on RISC-V targets which covers quantized vector dot products, FP16/BF16 utilities, GEMM/GEMV repack operations and core backend ops (MUL_MAT, FLASH_ATTN_EXT).

## Getting Started

Clone the repository and initialize the llama.cpp submodule:

```bash
git submodule init
git submodule update
```

To use an existing llama.cpp checkout instead, pass `-DLLAMA_CPP_DIR=/path/to/llama.cpp` at configure time.

---

## Tests and Benchmarks

| Binary | Description |
|---|---|
| `test-quantize` | Correctness tests for quantized `vecdot` and `quantize_row` functions. |
| `test-float` | Correctness tests for FP utility functions and `vecdot` across FP16 and BF16. |
| `perf-float` | Benchmarks for FP utility functions and `vecdot` across FP16 and BF16. |
| `test-repack` | Correctness tests for repack GEMM/GEMV operations. |
| `perf-repack` | Benchmarks for repack GEMM/GEMV operations. |
| `test-quantize-perf` | Benchmarks for quantized `vecdot` and `quantize_row` functions. |
| `test-backend-ops` | Correctness tests for the SGEMM (MUL_MAT) and flash attention (FLASH_ATTN_EXT) kernels. |

---

## Building

### Native Build

```bash
cmake --preset riscv-release
cmake --build --preset riscv-release
```

### Cross-Compilation

Before configuring, set the following variables in `cmake/riscv64-linux-gcc.cmake` to match your local setup:

- `RISCV_TOOLCHAIN_PATH` — path to your RISC-V toolchain
- `CMAKE_SYSROOT` — path to the target sysroot
- `RV_QEMU_BIN` — path to the QEMU binary

Then configure and build:

```bash
cmake --preset riscv-cross -DRV_VLEN=<VLEN>
cmake --build --preset riscv-cross
```

When cross-compiling, `CMAKE_CROSSCOMPILING_EMULATOR` is set automatically during configure. Only correctness-oriented tests are registered as CTest targets — perf tests and `model-bench-all` are intentionally excluded since cross-compilation runs under QEMU emulation and performance numbers would not be meaningful:

- `test-quantize`
- `test-float`
- `test-repack`
- `test-backend-ops-mul-mat`
- `test-backend-ops-flash-attn`

---

## Running Tests

### Preset-Based Tests (Recommended)

CMake presets are provided to run correctness tests at fixed VLEN values via QEMU, independent of the VLEN the project was compiled with:

```bash
ctest --preset riscv-128-tests
ctest --preset riscv-256-tests
ctest --preset riscv-512-tests
ctest --preset riscv-1024-tests
```

To also run the model-level benchmark after preset tests:

```bash
ctest --test-dir build-riscv-release -R model-bench-all -V
```

On native RISC-V hardware, use the native preset instead:

```bash
ctest --preset riscv-native-tests
ctest --test-dir build-riscv-release -R model-bench-all -V
```

### Running Individual Tests via CTest

For more granular control, tests can be invoked directly using CTest against the build directory. If cross-compiled, tests will execute under QEMU using the VLEN specified at configure time.

List all registered tests:

```bash
ctest --test-dir build-riscv-release -N
```

Run individual tests:

```bash
ctest --test-dir build-riscv-release -R test-quantize -V
ctest --test-dir build-riscv-release -R test-float -V
ctest --test-dir build-riscv-release -R test-repack -V
ctest --test-dir build-riscv-release -R perf-float -V
ctest --test-dir build-riscv-release -R perf-repack -V
ctest --test-dir build-riscv-release -R test-backend-ops-mul-mat -V
ctest --test-dir build-riscv-release -R perf-backend-ops-mul-mat -V
ctest --test-dir build-riscv-release -R test-backend-ops-flash-attn -V
ctest --test-dir build-riscv-release -R perf-backend-ops-flash-attn -V
ctest --test-dir build-riscv-release -R model-bench-all -V
```