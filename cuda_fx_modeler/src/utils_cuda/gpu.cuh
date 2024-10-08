#pragma once

#include <cuda_runtime.h>

#include <string>

#define gpuErrChk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// #define warpSize 32
#define warpMask 0x1F
#define warpIdx(thread) ((thread) >> 5)

void selectGpu();
