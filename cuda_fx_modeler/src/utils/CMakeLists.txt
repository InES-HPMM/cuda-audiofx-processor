add_library(
    utils 
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

 
target_compile_definitions(utils PUBLIC SPDLOG_COMPILED_LIB) # prevent using the header-only version of spdlog (should speed up compilation)
target_link_libraries(utils PRIVATE spdlog::spdlog)
