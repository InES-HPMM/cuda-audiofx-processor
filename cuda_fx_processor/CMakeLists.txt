# CMake example:https://github.com/sun1211/cmake_with_add_subdirectory/tree/master

cmake_minimum_required(VERSION 3.28.0)

get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" DIR_NAME ${DIR_NAME})
set(PROJECT_NAME ${DIR_NAME})
project(${PROJECT_NAME} LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 87)#https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17) # highest gcc standard supported by cuda toolkit 11.4
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${EIGEN3_INCLUDE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "/home/nvidia/git/spdlog/cmake")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package (Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.6.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # this is needed since the opencv4 libs are for some reason nested in a opencv2 subfolder which fucks with the include paths
find_package(CUDAToolkit 11.4 REQUIRED)
find_package(TensorRT REQUIRED)
find_library(NVINFER NAMES libnvinfer.so)
find_package(spdlog REQUIRED)



if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)



