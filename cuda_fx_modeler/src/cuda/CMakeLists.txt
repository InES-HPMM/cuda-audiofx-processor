
add_library(
    cuda
    STATIC
    gpu_signal_graph.cuh
    gpu_signal_graph.cu
    gpu_fx.cuh
    gpu_fx.cu
    fx_conv_fd.cu
    fx_eq.cu
    fx_trt.cu
    fx_map.cu
    fx_util.cu
    fx_gate.cu
    onnx_trt_engine_nam.cuh
    )
# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    cuda 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
    ${CMAKE_SOURCE_DIR}/src/audio
    ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
)
target_compile_definitions(cuda PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(cuda PRIVATE cufft utils utils_cuda audio spdlog::spdlog  Eigen3::Eigen ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} )
set_target_properties(cuda PROPERTIES LINKER_LANGUAGE CUDA)
