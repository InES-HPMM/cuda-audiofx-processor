#pragma once

__global__ void spin_kernel(float* dst, const float* src, const size_t n_samples);

__global__ void fff_mix(float* dst, const float* src1, const float* src2, size_t n, float ratio);