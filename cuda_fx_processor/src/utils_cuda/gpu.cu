#include <assert.h>
#include <cuda_runtime.h>

#include <log.hpp>

#include "gpu.cuh"
#define TAG "GPU"

// static int smToCores(int major, int minor) {
//     switch ((major << 4) | minor) {
//         case (9999 << 4 | 9999):
//             return 1;
//         case 0x30:
//         case 0x32:
//         case 0x35:
//         case 0x37:
//             return 192;
//         case 0x50:
//         case 0x52:
//         case 0x53:
//             return 128;
//         case 0x60:
//             return 64;
//         case 0x61:
//         case 0x62:
//             return 128;
//         case 0x70:
//         case 0x72:
//         case 0x75:
//             return 64;
//         case 0x80:
//             return 64;
//         case 0x86:
//             return 128;
//         default:
//             return 0;
//     };
// }

// copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
// TODO: get includes working instead of copy-pasting
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

void selectGpu() {
    int rc;
    int maxId = -1;
    uint16_t maxScore = 0;
    int count = 0;
    cudaDeviceProp prop;

    rc = cudaGetDeviceCount(&count);
    assert(cudaSuccess == rc);
    assert(count > 0);

    for (int id = 0; id < count; id++) {
        rc = cudaGetDeviceProperties(&prop, id);
        assert(cudaSuccess == rc);
        Log::info(TAG, "GPU Compute Mode: %i", prop.computeMode);
        if (prop.computeMode == cudaComputeModeProhibited) {
            Log::warn(TAG, "GPU %d: (%s) is prohibited", id, prop.name);
            continue;
        }
        int sm_per_multiproc = _ConvertSMVer2Cores(prop.major, prop.minor);

        Log::info(TAG, "GPU %d", id);
        Log::newline(ESC(1) "%s" ESC(0), prop.name);
        Log::newline("Compute capability: " ESC(1) "%d.%d" ESC(0), prop.major, prop.minor);
        Log::newline("Multiprocessors:    " ESC(1) "%d" ESC(0), prop.multiProcessorCount);
        Log::newline("SMs per processor:  " ESC(1) "%d" ESC(0), sm_per_multiproc);
        Log::newline("Clock rate:         " ESC(1) "%d" ESC(0), prop.clockRate);
        Log::newline();

        uint64_t score = (uint64_t)prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;
        if (score > maxScore) {
            maxId = id;
            maxScore = score;
        }
    }

    assert(maxId >= 0);

    gpuErrChk(cudaSetDevice(maxId));

    gpuErrChk(cudaGetDeviceProperties(&prop, maxId));

    Log::info(__func__, ESC(32; 1) "Selected GPU %d: \"%s\" with compute capability %d.%d",
              maxId, prop.name, prop.major, prop.minor);

    if (prop.integrated == 1) {
        Log::info(__func__, ESC(32; 1) "Integrated GPU detected");
    }
}
