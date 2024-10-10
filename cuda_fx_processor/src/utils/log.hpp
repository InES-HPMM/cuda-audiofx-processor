#pragma once
#include <cmath>
#include <cstdarg>
#include <iostream>
#include <string>

#include "path.hpp"
void setup_spdlog();

#ifdef ESC
#error ESC already defined
#else
#define ESC(n) "\x1b[" #n "m"
#endif

#ifdef EOL
#error EOL already defined
#else
#define EOL ESC(0) "\n"
#endif

inline std::string escapeRgb(uint8_t r, uint8_t g, uint8_t b) {
    char* buffer = (char*)alloca(32);
    b = pow(b / 256., 0.5) * 0xff;
    g = pow(g / 256., 0.5) * 0xff;
    r = pow(r / 256., 0.5) * 0xff;

    sprintf(buffer, "\x1b[38;2;%d;%d;%dm", r, g, b);
    return std::string(buffer);
}

enum class LogSeverity : int32_t {
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    DEBUG = 3
};
class Log {
   protected:
    static LogSeverity severity;
    static void message(
        std::ostream& os,
        const char* style,
        const LogSeverity severity,
        const std::string& id,
        const char* msg) noexcept;

   public:
    static void setSeverity(const LogSeverity severity) noexcept;
    static void debug(const std::string& id, const char* fmt, ...) noexcept;
    static void info(const std::string& id, const char* fmt, ...) noexcept;
    static void warn(const std::string& id, const char* fmt, ...) noexcept;
    static void error(const std::string& id, const char* fmt, ...) noexcept;
    static void log(const LogSeverity severity, const std::string& id, const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        switch (severity) {
            case LogSeverity::INFO:
                info(id, fmt);
                break;
            case LogSeverity::WARNING:
                warn(id, fmt);
                break;
            case LogSeverity::ERROR:
                error(id, fmt);
                break;
            default:
                throw std::runtime_error("Invalid log severity");
        }
        va_end(args);
    }
    static void newline() noexcept;
    static void newline(const char* fmt, ...) noexcept;

    static void lock() noexcept;
    static void unlock() noexcept;
};
