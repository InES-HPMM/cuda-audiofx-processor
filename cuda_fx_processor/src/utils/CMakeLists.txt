
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" BINARY_NAME ${DIR_NAME})
add_library(
    ${BINARY_NAME}
    STATIC
    global.hpp
    log.hpp
    log.cpp
    convert.hpp
    convert.cpp
    math_ext.hpp
    math_ext.cpp
    evaluator.hpp
    enums.hpp
    enums.cpp
    path.hpp
    path.cpp
)

 
target_compile_definitions(${BINARY_NAME} PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(${BINARY_NAME} PRIVATE spdlog::spdlog)
