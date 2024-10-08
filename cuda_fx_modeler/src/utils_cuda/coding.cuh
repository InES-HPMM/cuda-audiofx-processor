#pragma once

#include <cuda_runtime.h>

#include <enums.hpp>

#define INT16_SMPL_MAX 0x8000
#define INT32_SMPL_MAX 0x80000000
#define INT16_SMPL_MAX_F (float)INT16_SMPL_MAX
#define INT32_SMPL_MAX_F (float)INT32_SMPL_MAX

void pcm_to_float2(float2 *dst, const char *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream);
void pcm_to_float_interleaved(float *dst, const char *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream);
void float2_to_pcm(char *dst, const float2 *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream);
void float_interleaved_to_pcm(char *dst, const float *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream);
void pcm_to_float(void *dst, const char *src, size_t n_samples, size_t n_channels, size_t n_bytes, size_t bits_per_sample, ChannelOrder channel_order);
void float_to_pcm(char *dst, const void *src, size_t n_samples, size_t n_channels, size_t n_bytes, size_t bits_per_sample, ChannelOrder channel_order);