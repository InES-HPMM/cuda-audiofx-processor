#include "math_ext.hpp"

size_t roundUpToPow2(size_t n) {
    return pow(2, ceil(log(n) / log(2)));
}