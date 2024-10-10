#include "convert.hpp"

char* int_to_char_ptr(char* dst, void* src, size_t size) {
    std::memcpy(dst, src, size);
    return dst;
}

template <typename T>
std::string int_to_hex(T i) {
    std::stringstream stream;
    stream << "0x"
           << std::setfill('0') << std::setw(sizeof(T) * 2)
           << std::hex << i;
    return stream.str();
}

template std::string int_to_hex<size_t>(size_t i);