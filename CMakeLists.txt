cmake_minimum_required(VERSION 3.14)
project(rise-ggml-tests LANGUAGES C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(LLAMA_BUILD_TOOLS    ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    ON CACHE BOOL "" FORCE)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to llama.cpp source tree")

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

set(GGML_SRC_DIR       ${LLAMA_CPP_DIR}/ggml/src)
set(GGML_CPU_SRC_DIR   ${GGML_SRC_DIR}/ggml-cpu)
set(GGML_RVV_DIR       ${GGML_CPU_SRC_DIR}/arch/riscv)

set(RISE_INTERNAL_INCLUDES
        ${GGML_SRC_DIR}
        ${GGML_CPU_SRC_DIR}
)

# Add a test executable and register it with CTest.
function(add_rise_test name source)
    cmake_parse_arguments(RISE "ARCH_FLAGS;RVV;PERF" "" "LIBS;TEST_ARGS" ${ARGN})

    add_executable(${name} ${source})
    set_property(GLOBAL APPEND PROPERTY RISE_TEST_TARGETS ${name})

    target_include_directories(${name} PRIVATE ${RISE_INTERNAL_INCLUDES})

    target_include_directories(${name} PRIVATE ${GGML_RVV_DIR})

    set(_libs ggml-cpu ggml-base)
    target_link_libraries(${name} PRIVATE ${_libs})

    # Mirror ggml-cpu arch flags (e.g., -march, AVX) for parity.
    target_compile_options(${name} PRIVATE
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_OPTIONS>
    )
    target_compile_definitions(${name} PRIVATE
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_DEFINITIONS>
    )

    if(NOT (RISE_PERF AND CMAKE_CROSSCOMPILING_EMULATOR))
        add_test(NAME ${name} COMMAND ${name} ${RISE_TEST_ARGS})
    endif()
endfunction()

function(add_root_ctest name command)
    cmake_parse_arguments(RISE_CTEST "PERF" "WORKING_DIRECTORY" "ARGS" ${ARGN})

    if(RISE_CTEST_PERF AND CMAKE_CROSSCOMPILING_EMULATOR)
        return()
    endif()

    if(NOT DEFINED RISE_CTEST_WORKING_DIRECTORY)
        set(RISE_CTEST_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    set(RISE_CTEST_COMMAND ${command})
    if(CMAKE_CROSSCOMPILING_EMULATOR)
        set(RISE_CTEST_COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${RISE_CTEST_COMMAND})
    endif()

    add_test(
        NAME ${name}
        WORKING_DIRECTORY ${RISE_CTEST_WORKING_DIRECTORY}
        COMMAND ${RISE_CTEST_COMMAND} ${RISE_CTEST_ARGS})
endfunction()

enable_testing()

add_rise_test(test-quantize tests/test-quantize.cpp ARCH_FLAGS TEST_ARGS --size 256 --size 512 --size 1024 --size 2048 --size 5632)
add_rise_test(test-float tests/test-float.cpp ARCH_FLAGS TEST_ARGS --size 256 --size 512 --size 1024 --size 2048 --size 5632)
add_rise_test(perf-float tests/perf-float.cpp PERF ARCH_FLAGS TEST_ARGS --size 256 --size 512 --size 1024 --size 2048 --size 5632)
add_rise_test(test-repack tests/test-repack.cpp ARCH_FLAGS TEST_ARGS --m 1,256 --k 256 --n 2048 --threads 8)
add_rise_test(perf-repack tests/perf-repack.cpp PERF ARCH_FLAGS TEST_ARGS --m 1,256 --k 256 --n 2048 --threads 8)

add_root_ctest(
        test-quantize-perf
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-quantize-perf)
add_root_ctest(
    test-backend-ops-mul-mat
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-ops
    ARGS test -o MUL_MAT -b CPU -p "(^|,)m=([2-9]|[1-9][0-9]+)(,|$)")
add_root_ctest(
    test-backend-ops-flash-attn
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-ops
    ARGS test -o FLASH_ATTN_EXT -b CPU)
add_root_ctest(
    perf-backend-ops-mul-mat
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-ops
    PERF
    ARGS perf -o MUL_MAT -b CPU -p "(^|,)m=([2-9]|[1-9][0-9]+)(,|$)")
add_root_ctest(
    perf-backend-ops-flash-attn
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-ops
    PERF
    ARGS perf -o FLASH_ATTN_EXT -b CPU)

if(NOT CMAKE_CROSSCOMPILING_EMULATOR)
    add_root_ctest(
        model-bench-all
        ${CMAKE_COMMAND}
        ARGS
            -DLLAMA_BENCH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-bench
            -DMODELS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/models
            -DTHREADS=8
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run-model-bench-all.cmake)
endif()
