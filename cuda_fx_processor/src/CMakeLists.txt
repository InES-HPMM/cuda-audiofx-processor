
add_subdirectory(utils)
add_subdirectory(utils_cuda)
add_subdirectory(audio)
add_subdirectory(audio_cuda)
add_subdirectory(eval)


add_executable(${PROJECT_NAME} main.cpp)
# enables pathless includes (in main from listed libraries)
target_include_directories(${PROJECT_NAME} PRIVATE utils utils_cuda audio audio_cuda)
target_link_libraries(${PROJECT_NAME} PRIVATE utils utils_cuda audio audio_cuda jack asound)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $ENV{DIST})

set(EXECUTABLE_NAME "test")
add_executable(${EXECUTABLE_NAME} test.cpp)
# enables pathless includes (in main from listed libraries)
target_include_directories(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval jack asound)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $ENV{DIST})


set(EXECUTABLE_NAME "scratch")
add_executable(${EXECUTABLE_NAME} scratch.cu ${CUDA_LIBRARIES})
target_compile_definitions(${EXECUTABLE_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_include_directories(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval ${CUDA_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval cufft spdlog::spdlog ${CUDA_LIBRARIES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $ENV{DIST})


set(EXECUTABLE_NAME "fx_performance_test")
add_executable(${EXECUTABLE_NAME} fx_performance_test.cu  ${CUDA_LIBRARIES})
target_compile_definitions(${EXECUTABLE_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_include_directories(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval ${CUDA_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval cufft spdlog::spdlog ${CUDA_LIBRARIES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $ENV{DIST})


set(EXECUTABLE_NAME "signal_graph_performance_test")
add_executable(${EXECUTABLE_NAME} signal_graph_performance_test.cu  ${CUDA_LIBRARIES})
target_compile_definitions(${EXECUTABLE_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_include_directories(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval ${CUDA_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE utils utils_cuda audio audio_cuda eval cufft spdlog::spdlog ${CUDA_LIBRARIES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $ENV{DIST})
