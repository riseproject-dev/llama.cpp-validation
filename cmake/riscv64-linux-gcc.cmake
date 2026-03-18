list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RV_VLEN)
if(NOT DEFINED RVV_VLEN)
    message(FATAL_ERROR "RVV_VLEN is mandatory for RISC-V cross-compilation! Please specify it (e.g., via a CMake preset or by passing -DRV_VLEN=256).")
endif()
if(NOT DEFINED RV_TOOLCHAIN_BIN)
    set(RV_TOOLCHAIN_BIN "/home/rehan-10xe/Downloads/riscv64-glibc-ubuntu-22.04-gcc/riscv/bin")
endif()

if(NOT DEFINED RV_SYSROOT)
    set(RV_SYSROOT "/home/rehan-10xe/Downloads/riscv64-glibc-ubuntu-22.04-gcc/riscv/sysroot")
endif()

if(NOT DEFINED RV_QEMU_BIN)
    set(RV_QEMU_BIN "/home/rehan-10xe/Downloads/riscv64-glibc-ubuntu-22.04-llvm-nightly-2024.08.03-nightly/riscv/bin/qemu-riscv64")
endif()
# ==============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(target riscv64-unknown-elf)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

set(CMAKE_SYSROOT "${RV_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "")
set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "")
set(CMAKE_ASM_IMPLICIT_INCLUDE_DIRECTORIES "")

set(CMAKE_C_COMPILER    ${RV_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER  ${RV_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-g++)

set(CMAKE_C_COMPILER_TARGET   ${target})
set(CMAKE_CXX_COMPILER_TARGET ${target})

set(DEFAULT_QEMU_CPU "rv64,v=true,vlen=${RVV_VLEN},zfh=true,zvfh=true,zifencei=true,vext_spec=v1.0")
set(CMAKE_CROSSCOMPILING_EMULATOR
        "sh" "-c"
        "QEMU_CPU=\"\${QEMU_CPU:-${DEFAULT_QEMU_CPU}}\" exec ${RV_QEMU_BIN} -L ${RV_SYSROOT} \"$0\" \"$@\""
)