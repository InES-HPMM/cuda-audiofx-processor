#include <cuda_runtime.h>

#include "kernels.cuh"
#include "operators.cuh"

__global__ void spin_kernel(float* dst, const float* src, const size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_samples; s += stride.x) {
        dst[s] = src[s] + src[s];
    }
}