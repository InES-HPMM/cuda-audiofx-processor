
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" BINARY_NAME ${DIR_NAME})

add_library(
    ${BINARY_NAME}
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
    ${BINARY_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/utils
)
target_compile_definitions(${BINARY_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(${BINARY_NAME} PRIVATE utils spdlog::spdlog CUDA::cusolver_static )
set_target_properties(${BINARY_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
