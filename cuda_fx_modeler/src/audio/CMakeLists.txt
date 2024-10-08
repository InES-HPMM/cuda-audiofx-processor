add_library(audio 
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
    jack_driver_async.cpp
    file_driver.cpp
    signal_driver.cpp
)

# enables pathless includes from other libraries (note that the libraries from the listed directories also need to be linked)
target_include_directories(
    audio 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/utils_cuda
)

target_compile_definitions(audio PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(audio PUBLIC utils utils_cuda spdlog::spdlog  jack asound)
