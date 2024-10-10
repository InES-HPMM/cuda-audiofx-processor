
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" BINARY_NAME ${DIR_NAME})

add_library(
    ${BINARY_NAME}
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
    ${BINARY_NAME} 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
    ${CMAKE_SOURCE_DIR}/src/audio
    ${CMAKE_SOURCE_DIR}/src/audio_cuda
    ${CUDA_INCLUDE_DIRS}
)
target_compile_definitions(${BINARY_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(${BINARY_NAME} PRIVATE utils utils_cuda audio cuda spdlog::spdlog ${CUDA_LIBRARIES} )
set_target_properties(${BINARY_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
