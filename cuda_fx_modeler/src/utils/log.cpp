#include "log.hpp"

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

void setup_spdlog() {
    spdlog::init_thread_pool(8192, 1);
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path::out("log.txt"), 1024 * 1024 * 10, 3);
    std::vector<spdlog::sink_ptr> sinks{stdout_sink, rotating_sink};
    auto logger = std::make_shared<spdlog::async_logger>("loggername", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v ");
}

LogSeverity Log::severity = LogSeverity::INFO;

void Log::setSeverity(const LogSeverity severity) noexcept {
    Log::severity = severity;
}

void Log::message(std::ostream& os, const char* style, const LogSeverity severity, const std::string& id, const char* msg) noexcept {
    if (severity > Log::severity) {
        return;
    }
    auto now = std::chrono::system_clock::now();
    auto dt = std::chrono::system_clock::to_time_t(now);
    const char* type;
    switch (severity) {
        case LogSeverity::ERROR:
            type = ESC(31; 1) "E";
            break;
        case LogSeverity::WARNING:
            type = ESC(33) "W";
            break;
        case LogSeverity::INFO:
            type = "I";
            break;
        default:
            type = "D";
            break;
    }
    os << ESC(37; 2) << type << " "
       << std::put_time(std::localtime(&dt), "%F %T") << ESC(0) << ESC(37)
       << style << ESC(1; 2) " [" << id << ESC(0) << style << ESC(1; 2) "] " ESC(0) << ESC(37);

    os << style << msg << EOL << std::flush;
}

void Log::debug(const std::string& id, const char* fmt, ...) noexcept {
    int rc;
    char buffer[256];

    assert(fmt);
    va_list args;
    va_start(args, fmt);
    rc = vsnprintf(buffer, 255, fmt, args);
    assert(0 <= rc);
    va_end(args);
    assert(rc > 0);

    message(std::cout, ESC(37), LogSeverity::DEBUG, id, buffer);
}

void Log::info(const std::string& id, const char* fmt, ...) noexcept {
    int rc;
    char buffer[256];

    assert(fmt);
    va_list args;
    va_start(args, fmt);
    rc = vsnprintf(buffer, 255, fmt, args);
    assert(0 <= rc);
    va_end(args);
    assert(rc > 0);

    message(std::cout, ESC(37), LogSeverity::INFO, id, buffer);
}

void Log::warn(const std::string& id, const char* fmt, ...) noexcept {
    int rc;
    char buffer[256];

    assert(fmt);
    va_list args;
    va_start(args, fmt);
    rc = vsnprintf(buffer, 255, fmt, args);
    assert(0 <= rc);
    va_end(args);
    assert(rc > 0);

    message(std::cerr, ESC(33; 1), LogSeverity::WARNING, id, buffer);
}

void Log::error(const std::string& id, const char* fmt, ...) noexcept {
    int rc;
    char buffer[256];

    assert(fmt);
    va_list args;
    va_start(args, fmt);
    rc = vsnprintf(buffer, 255, fmt, args);
    assert(0 <= rc);
    va_end(args);
    assert(rc > 0);

    message(std::cerr, ESC(31; 1; 2), LogSeverity::ERROR, id, buffer);
}

void Log::newline() noexcept {
    std::cout << std::string(22, ' ');
    std::cout << std::endl;
}

void Log::newline(const char* fmt, ...) noexcept {
    int rc;
    char buffer[256];

    assert(fmt);
    va_list args;
    va_start(args, fmt);
    rc = vsnprintf(buffer, 255, fmt, args);
    assert(0 <= rc);
    va_end(args);
    assert(rc > 0);

    std::cout << std::string(22, ' ');
    std::cout << buffer;
    std::cout << std::endl;
}
