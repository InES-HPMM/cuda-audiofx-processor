#pragma once

__global__ void spin_kernel(float* dst, const float* src, const size_t n_samples);
