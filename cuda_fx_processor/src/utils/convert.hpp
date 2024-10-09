#pragma once
#include <cstring>
#include <iomanip>

char* int_to_char_ptr(char* dst, void* src, size_t size);

template <typename T>
std::string int_to_hex(T i);
