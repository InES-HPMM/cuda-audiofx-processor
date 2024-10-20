
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" BINARY_NAME ${DIR_NAME})

add_library(${BINARY_NAME} 
    STATIC 
    signal_generator.hpp
    signal_generator.cpp
    signal.hpp
    signal.cpp
    fx.hpp
    signal_graph.hpp
    signal_graph.cpp
    driver.hpp
    jack_driver.cpp
    signal_driver.cpp
)

# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    ${BINARY_NAME}  
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
)

target_compile_definitions(${BINARY_NAME}  PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(${BINARY_NAME}  PUBLIC utils utils_cuda spdlog::spdlog  jack asound)
