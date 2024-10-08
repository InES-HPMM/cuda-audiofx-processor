add_library(
    utils_cuda
    STATIC
    operators.cuh
    coding.cuh
    coding.cu
    gpu.cuh
    gpu.cu
    block_buffer.cuh
    block_buffer.cu
    cuda_ext.cuh
    cuda_ext.cu
    rmsd.cuh
    rmsd.cu
    buffer.cuh
    buffer.cu
    kernels.cuh
    kernels.cu
)
# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    utils_cuda
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/utils
)
target_compile_definitions(utils_cuda PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(utils_cuda PRIVATE utils spdlog::spdlog CUDA::cusolver_static )
set_target_properties(utils_cuda PROPERTIES LINKER_LANGUAGE CUDA)
