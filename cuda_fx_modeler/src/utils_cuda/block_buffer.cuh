#pragma once
#include "gpu.cuh"

class IBlockBuffer {
   public:
    static IBlockBuffer* createInstance(size_t n_samples, size_t n_channels, size_t block_count, size_t init_fill_count);

    virtual ~IBlockBuffer() {}

    virtual float* getMemoryPointer() = 0;
    virtual size_t getProcOffset() = 0;
    virtual bool tryReadBlock(float** buffers, bool deinterleave = false) = 0;
    virtual bool tryWriteBlock(const float* const* buffers, bool interleave = false) = 0;
    virtual float* getReadPointer() = 0;
    virtual float* getWritePointer() = 0;
    virtual bool tryGetProcPointer(float** ptr) = 0;
    virtual float* getReadPointerBlocking() = 0;
    virtual float* getWritePointerBlocking() = 0;
    virtual float* getProcPointerBlocking() = 0;
    virtual bool tryAdvanceReadPointer() = 0;
    virtual bool tryAdvanceWritePointer() = 0;
    virtual bool tryAdvanceProcPointer() = 0;
};