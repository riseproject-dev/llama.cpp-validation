if(NOT DEFINED LLAMA_BENCH OR LLAMA_BENCH STREQUAL "")
    message(FATAL_ERROR "LLAMA_BENCH is required")
endif()

if(NOT DEFINED MODELS_DIR OR MODELS_DIR STREQUAL "")
    message(FATAL_ERROR "MODELS_DIR is required")
endif()

if(NOT DEFINED THREADS OR THREADS STREQUAL "")
    set(THREADS 8)
endif()

file(GLOB model_files
    "${MODELS_DIR}/*.gguf"
    "${MODELS_DIR}/*.GGUF")
list(SORT model_files)

list(LENGTH model_files model_count)
if(model_count EQUAL 0)
    message(FATAL_ERROR "No GGUF models found in ${MODELS_DIR}")
endif()

foreach(model_path IN LISTS model_files)
    get_filename_component(model_name "${model_path}" NAME)
    message(STATUS "Benchmarking ${model_name}")

    execute_process(
        COMMAND "${LLAMA_BENCH}" -m "${model_path}" -t "${THREADS}" -p 32,64,128,256,512 -n 0
        RESULT_VARIABLE bench_status_1)
    if(NOT bench_status_1 EQUAL 0)
        message(FATAL_ERROR "llama-bench failed for ${model_name} with prompt sweep")
    endif()

    execute_process(
        COMMAND "${LLAMA_BENCH}" -m "${model_path}" -t "${THREADS}" -p 32 -n 10,16,32,64,100
        RESULT_VARIABLE bench_status_2)
    if(NOT bench_status_2 EQUAL 0)
        message(FATAL_ERROR "llama-bench failed for ${model_name} with generation sweep")
    endif()
endforeach()
