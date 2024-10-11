
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" BINARY_NAME ${DIR_NAME})

add_library(
    ${BINARY_NAME}
    STATIC
    gpu_signal_graph.cuh
    gpu_signal_graph.cu
    gpu_fx.cuh
    gpu_fx.cu
    fx_conv_fd.cu
    fx_eq.cu
    fx_nam.cu
    fx_map.cu
    fx_util.cu
    fx_gate.cu
    onnx_trt_engine_nam.cuh
    )
# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    ${BINARY_NAME} 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
    ${CMAKE_SOURCE_DIR}/src/audio
    ${CMAKE_SOURCE_DIR}/src/trt
    ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
)
target_compile_definitions(${BINARY_NAME}  PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(${BINARY_NAME}  PRIVATE cufft utils utils_cuda audio trt spdlog::spdlog  Eigen3::Eigen ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} )
set_target_properties(${BINARY_NAME}  PROPERTIES LINKER_LANGUAGE CUDA)
