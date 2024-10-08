
#include <cassert>
#include <stdexcept>

#include "coding.cuh"
#include "cuda_ext.cuh"
#include "gpu.cuh"
#include "operators.cuh"

// The following function encode and decode using the transparent int-float-int conversion method sacrificing marginal full scale range in float
// see option 1 in http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
// explanation of the translation issue: https://stackoverflow.com/a/59068796/24234025

__global__ static void int16_to_float2(float2 *dst, const short2 *src, size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples; s += stride.x) {
        short2 sample_2ch = src[s];
        dst[s] = make_float2(sample_2ch.x / INT16_SMPL_MAX_F, sample_2ch.y / INT16_SMPL_MAX_F);
    }
}

__global__ static void int16_to_float(float *dst, const short *src, size_t n_samples, size_t n_channels) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples * n_channels; s += stride.x) {
        dst[s] = src[s] / INT16_SMPL_MAX_F;
    }
}

// 24-bit samples read from wav as 3 sequential bytes (LSB first) per channel
// this function combines every 3 bytes into a 24-bit sample
// as there is no native 24-bit type, we use 32-bit int to store the sample
// the bytes are stored into the upper 3 bytes of the int (the lower byte is 0) to naturally adopt the sign of the MSB
// as a result the final float conversion scaling is done using the 32-bit max value
// https://stackoverflow.com/questions/24151973/reading-24-bit-samples-from-a-wav-file
__global__ static void int24_to_float2(float2 *dst, const uint8_t *src, size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples; s += stride.x) {
        int32_t sample_ch1 = 0;
        int32_t sample_ch2 = 0;
        for (size_t i = 0; i < 3; i++) {
            sample_ch1 |= (src[s * 6 + i] << ((i + 1) * 8));
            sample_ch2 |= (src[s * 6 + i + 3] << ((i + 1) * 8));
        }

        dst[s] = make_float2(
            sample_ch1 / INT32_SMPL_MAX_F,
            sample_ch2 / INT32_SMPL_MAX_F);
    }
}
__global__ static void int24_to_float(float *dst, const uint8_t *src, size_t n_samples, size_t n_channels) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples * n_channels; s += stride.x) {
        int32_t sample = 0;
        for (size_t i = 0; i < 3; i++) {
            sample |= (src[s * 3 + i] << ((i + 1) * 8));
        }

        dst[s] = sample / INT32_SMPL_MAX_F;
    }
}

__global__ static void float2_to_int16(short2 *dst, const float2 *src, size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples; s += stride.x) {
        float2 sample_2ch = src[s];
        // TODO: ensure that the float is in the range [-1, 1]
        dst[s] = make_short2(
            sample_2ch.x * INT16_SMPL_MAX_F,
            sample_2ch.y * INT16_SMPL_MAX_F);
    }
}

__global__ static void float_to_int16(short *dst, const float *src, size_t n_samples, size_t n_channels) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples * n_channels; s += stride.x) {
        // TODO: ensure that the float is in the range [-1, 1]
        dst[s] = src[s] * INT16_SMPL_MAX_F;
    }
}

__global__ static void float2_to_int24(uint8_t *dst, const float2 *src, size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (size_t s = offset.x; s < n_samples; s += stride.x) {
        float2 sample_2ch = src[s];
        int32_t sample_ch1 = sample_2ch.x * INT32_SMPL_MAX_F;
        int32_t sample_ch2 = sample_2ch.y * INT32_SMPL_MAX_F;

        // TODO: ensure that the float is in the range [-1, 1]
        for (size_t i = 0; i < 3; i++) {
            dst[s * 6 + i] = (sample_ch1 >> ((i + 1) * 8)) & 0xFF;
            dst[s * 6 + i + 3] = (sample_ch2 >> ((i + 1) * 8)) & 0xFF;
        }
    }
}
__global__ static void float_to_int24(uint8_t *dst, const float *src, int n_samples, size_t n_channels) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (int s = offset.x; s < n_samples * n_channels; s += stride.x) {
        int32_t sample = src[s] * INT32_SMPL_MAX_F;

        // TODO: ensure that the float is in the range [-1, 1]
        for (int i = 0; i < 3; i++) {
            dst[s * 3 + i] = (sample >> ((i + 1) * 8)) & 0xFF;
        }
    }
}

// __global__ static void int16_to_float(float **dst, const char *src, int n_channels, int n_samples) {
//     auto stride = gridDim * blockDim;
//     auto offset = blockDim * blockIdx + threadIdx;

//     for (int s = offset.x; s < n_samples; s += stride.x) {
//         for (int c = 0; c < n_channels; c++) {
//             dst[c][s] = ((short *)src)[s * n_channels + c] / INT16_SMPL_MAX_F;
//         }
//     }
// }

// // 24-bit samples read from wav as 3 sequential bytes (LSB first) per channel
// // this function combines every 3 bytes into a 24-bit sample
// // as there is no native 24-bit type, we use 32-bit int to store the sample
// // the bytes are stored into the upper 3 bytes of the int (the lower byte is 0) to naturally adopt the sign of the MSB
// // as a result the final float conversion scaling is done using the 32-bit max value
// // https://stackoverflow.com/questions/24151973/reading-24-bit-samples-from-a-wav-file
// __global__ static void int24_to_float(float **dst, const char *src, int n_channels, int n_samples) {
//     auto stride = gridDim * blockDim;
//     auto offset = blockDim * blockIdx + threadIdx;

//     for (int s = offset.x; s < n_samples; s += stride.x) {
//         int32_t sample = 0;
//         for (int c = 0; c < n_channels; c++) {
//             for (int i = 0; i < 3; i++) {
//                 sample |= (src[s * c * 3 + i] << ((i + 1) * 8));
//             }
//             dst[c][s] = sample / INT32_SMPL_MAX_F;
//         }
//     }
// }

// __global__ static void float_to_int16(char *dst, const float *const *src, int n_channels, int n_samples) {
//     auto stride = gridDim * blockDim;
//     auto offset = blockDim * blockIdx + threadIdx;

//     for (int s = offset.x; s < n_samples; s += stride.x) {
//         for (int c = 0; c < n_channels; c++) {
//             ((short *)dst)[s * n_channels + c] = src[c][s] * INT16_SMPL_MAX_F;
//         }
//     }
// }

// __global__ static void float_to_int24(char *dst, const float *const *src, int n_channels, int n_samples) {
//     auto stride = gridDim * blockDim;
//     auto offset = blockDim * blockIdx + threadIdx;

//     for (int s = offset.x; s < n_samples; s += stride.x) {
//         for (int c = 0; c < n_channels; c++) {
//             int32_t sample = src[c][s] * INT32_SMPL_MAX_F;
//             for (int i = 0; i < 3; i++) {
//                 dst[s * c * 3 + i] = (sample >> ((i + 1) * 8)) & 0xFF;
//             }
//         }
//     }
// }

void pcm_to_float2(float2 *dst, const char *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream) {
    assert(2 == n_channels);
    if (bits_per_sample == 16) {
        int16_to_float2<<<16, 256, 0, stream>>>(dst, (short2 *)src, n_samples);
    } else if (bits_per_sample == 24) {
        int24_to_float2<<<16, 256, 0, stream>>>(dst, (uint8_t *)src, n_samples);
    } else {
        throw std::runtime_error("Unsupported bits per sample");
    }
}
void pcm_to_float_interleaved(float *dst, const char *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream) {
    if (bits_per_sample == 16) {
        int16_to_float<<<16, 256, 0, stream>>>(dst, (short *)src, n_samples, n_channels);
    } else if (bits_per_sample == 24) {
        int24_to_float<<<16, 256, 0, stream>>>(dst, (uint8_t *)src, n_samples, n_channels);
    } else {
        throw std::runtime_error("Unsupported bits per sample");
    }
}

void float2_to_pcm(char *dst, const float2 *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream) {
    assert(2 == n_channels);
    if (bits_per_sample == 16) {
        float2_to_int16<<<16, 256, 0, stream>>>((short2 *)dst, src, n_samples);
    } else if (bits_per_sample == 24) {
        float2_to_int24<<<16, 256, 0, stream>>>((uint8_t *)dst, src, n_samples);
    } else {
        throw std::runtime_error("Unsupported bits per sample");
    }
}

void float_interleaved_to_pcm(char *dst, const float *src, size_t n_samples, size_t n_channels, size_t bits_per_sample, cudaStream_t stream) {
    if (bits_per_sample == 16) {
        float_to_int16<<<16, 256, 0, stream>>>((short *)dst, src, n_samples, n_channels);
    } else if (bits_per_sample == 24) {
        float_to_int24<<<16, 256, 0, stream>>>((uint8_t *)dst, src, n_samples, n_channels);
    } else {
        throw std::runtime_error("Unsupported bits per sample");
    }
}

void pcm_to_float(void *dst, const char *src, size_t n_samples, size_t n_channels, size_t n_bytes, size_t bits_per_sample, ChannelOrder channel_order) {
    char *dev_bytes;
    float *dev_samples;
    gpuErrChk(cudaMalloc(&dev_bytes, n_bytes * sizeof(char)));
    gpuErrChk(cudaMalloc(&dev_samples, n_samples * n_channels * sizeof(float)));

    cudaStream_t stream;
    gpuErrChk(cudaStreamCreate(&stream));
    gpuErrChk(cudaMemcpyAsync(dev_bytes, src, n_bytes * sizeof(char), cudaMemcpyHostToDevice, stream));

    if (bits_per_sample == 16) {
        int16_to_float<<<16, 256, 0, stream>>>(dev_samples, (short *)dev_bytes, n_samples, n_channels);
    } else if (bits_per_sample == 24) {
        int24_to_float<<<16, 256, 0, stream>>>(dev_samples, (uint8_t *)dev_bytes, n_samples, n_channels);
    } else {
        assert(false);
    }

    if (channel_order == ChannelOrder::INTERLEAVED) {
        gpuErrChk(cudaMemcpyAsync(dst, dev_samples, n_samples * n_channels * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else {
        IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Interleaved2Segmented, dst, dev_samples, sizeof(float), n_samples, n_channels, {}, cudaMemcpyDeviceToHost, stream);
    }
    gpuErrChk(cudaStreamSynchronize(stream));
    gpuErrChk(cudaStreamDestroy(stream));
    gpuErrChk(cudaFree(dev_bytes));
    gpuErrChk(cudaFree(dev_samples));
}

void float_to_pcm(char *dst, const void *src, size_t n_samples, size_t n_channels, size_t n_bytes, size_t bits_per_sample, ChannelOrder channel_order) {
    char *dev_bytes;
    float *dev_samples;
    gpuErrChk(cudaMalloc(&dev_bytes, n_bytes * sizeof(char)));
    gpuErrChk(cudaMalloc(&dev_samples, n_samples * n_channels * sizeof(float)));
    cudaStream_t stream;
    gpuErrChk(cudaStreamCreate(&stream));

    if (channel_order == ChannelOrder::INTERLEAVED) {
        gpuErrChk(cudaMemcpyAsync(dev_samples, src, n_samples * n_channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    } else {
        IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Segmented2Interleaved, dev_samples, src, sizeof(float), n_samples, n_channels, {}, cudaMemcpyHostToDevice, stream);
    }

    if (bits_per_sample == 16) {
        float_to_int16<<<16, 256, 0, stream>>>((short *)dev_bytes, dev_samples, n_samples, n_channels);
    } else if (bits_per_sample == 24) {
        float_to_int24<<<16, 256, 0, stream>>>((uint8_t *)dev_bytes, dev_samples, n_samples, n_channels);
    } else {
        assert(false);
    }

    gpuErrChk(cudaMemcpyAsync(dst, dev_bytes, n_bytes * sizeof(char), cudaMemcpyDeviceToHost, stream));
    gpuErrChk(cudaStreamSynchronize(stream));
    gpuErrChk(cudaStreamDestroy(stream));
    gpuErrChk(cudaFree(dev_bytes));
    gpuErrChk(cudaFree(dev_samples));
}