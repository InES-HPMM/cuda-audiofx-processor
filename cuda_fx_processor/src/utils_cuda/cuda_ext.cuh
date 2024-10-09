#pragma once
#include <cuda_runtime.h>

#include <enums.hpp>
#include <stdexcept>
#include <vector>

#include "gpu.cuh"
#include "spdlog/spdlog.h"

bool isDevicePointer(const void* ptr);

bool copyToDevicePtr(const void* dest, const void* src, size_t size, cudaStream_t stream = nullptr);
void logCudaGraphNodes(cudaGraph_t graph, spdlog::level::level_enum log_level, std::string parent_node_id = "");

enum class MultiMemcpyType { Segmented2Interleaved,
                             Interleaved2Segmented };

class IGraphNode {
   protected:
    cudaGraphNode_t _node;

   public:
    virtual ~IGraphNode() {}

    virtual void addToGraph(cudaStream_t stream) = 0;
    virtual void addToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) = 0;
    virtual void addToGraph(cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) = 0;

    virtual void update(cudaGraphExec_t procGraphExec) = 0;

    virtual cudaGraphNode_t getNode() { return _node; }
    virtual cudaGraphNode_t* getNodePtr() { return &_node; }
};

class IKernelNode : public IGraphNode {
   public:
    static void launchOrRecord(dim3 n_blocks, dim3 n_thread, size_t sharedMem, void* kernel, void** args, cudaStream_t stream = nullptr, IKernelNode* instance = nullptr, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone);
    static IKernelNode* create(dim3 n_blocks, dim3 n_thread, size_t sharedMem, void* kernel, void** args, cudaGraph_t graph = nullptr, const cudaGraphNode_t* dependencies = nullptr, size_t n_dependencies = 0);

    virtual void updateKernelParamAt(int index, void* param, cudaGraphExec_t procGraphExec = nullptr) = 0;
};

class IMemCpyNode : public IGraphNode {
   public:
    static void launchOrRecord1D(void* dst, const void* src, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaStream_t stream = nullptr, IMemCpyNode* instance = nullptr, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone);
    static IMemCpyNode* create1D(void* dst, const void* src, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaGraph_t graph = nullptr, const cudaGraphNode_t* dependencies = nullptr, size_t n_dependencies = 0);

    static void launchOrRecord2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaStream_t stream = nullptr, IMemCpyNode* instance = nullptr, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone);
    static IMemCpyNode* create2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaGraph_t graph = nullptr, const cudaGraphNode_t* dependencies = nullptr, size_t n_dependencies = 0);

    static void launchOrRecordMulti(MultiMemcpyType type, void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind, cudaStream_t stream = nullptr, IMemCpyNode* instance = nullptr, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone);
    static IMemCpyNode* createMulti(MultiMemcpyType type, void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind, cudaGraph_t graph = nullptr, const cudaGraphNode_t* dependencies = nullptr, size_t n_dependencies = 0);

    virtual void updateSrcPtr(const void* src, cudaGraphExec_t procGraphExec = nullptr) = 0;
    virtual void updateDstPtr(void* dst, cudaGraphExec_t procGraphExec = nullptr) = 0;
    virtual const void* getSrcPtr() = 0;
    virtual void* getDstPtr() = 0;
};
