
add_library(
    eval
    STATIC
    gpu_fx_eval.cuh
    gpu_fx_eval.cu
    gpu_signal_graph_eval.cuh
    gpu_signal_graph_eval.cu
    kernel_eval.cuh
    kernel_vector_load_eval.cuh
)
# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    eval 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
    ${CMAKE_SOURCE_DIR}/src/audio
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CUDA_INCLUDE_DIRS}
)
target_compile_definitions(eval PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(eval PRIVATE utils utils_cuda audio cuda spdlog::spdlog ${CUDA_LIBRARIES} )
set_target_properties(eval PROPERTIES LINKER_LANGUAGE CUDA)
