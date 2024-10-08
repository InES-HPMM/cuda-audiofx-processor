#pragma once


__global__ static void f4_rmse(float* rmse, const float4* src1, const float4* src2, size_t size);

__global__ void spin_kernel(float* dst, const float* src, const size_t n_samples);
