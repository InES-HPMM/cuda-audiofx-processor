#include <numeric>

#include "cuda_ext.cuh"

bool isDevicePointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    return attributes.devicePointer;
}

bool copyToDevicePtr(const void* dest, const void* src, size_t size, cudaStream_t stream) {
    if (isDevicePointer(src)) {
        dest = src;
        return false;
    } else {
        bool destroy_stream = false;
        if (!stream) {
            gpuErrChk(cudaStreamCreate(&stream));
            destroy_stream = true;
        }
        gpuErrChk(cudaMalloc(&dest, sizeof(src)));
        gpuErrChk(cudaMemcpyAsync((void*)dest, src, sizeof(dest), cudaMemcpyHostToDevice, stream));

        if (destroy_stream) {
            gpuErrChk(cudaStreamSynchronize(stream));
            gpuErrChk(cudaStreamDestroy(stream));
        }
        return true;
    }
}

void logCudaGraphNodes(cudaGraph_t graph, spdlog::level::level_enum log_level, std::string parent_node_id) {
        cudaGraphNode_t* nodes = nullptr;
        size_t node_count = 0;
        gpuErrChk(cudaGraphGetNodes(graph, nodes, &node_count));  // get number of Nodes
        if (node_count > 0) {
            nodes = new cudaGraphNode_t[node_count];
            gpuErrChk(cudaGraphGetNodes(graph, nodes, &node_count));  // get Nodes
            for (size_t i = 0; i < node_count; i++) {
                cudaGraphNodeType type;
                gpuErrChk(cudaGraphNodeGetType(nodes[i], &type));
                std::string node_id = parent_node_id + std::to_string(i);
                switch (type) {
                    case cudaGraphNodeTypeKernel:
                        spdlog::log(log_level, "Node {} Type: Kernel", node_id);
                        break;
                    case cudaGraphNodeTypeMemcpy:
                        spdlog::log(log_level, "Node {} Type: Memcpy", node_id);
                        break;
                    case cudaGraphNodeTypeMemset:
                        spdlog::log(log_level, "Node {} Type: Memset", node_id);
                        break;
                    case cudaGraphNodeTypeHost:
                        spdlog::log(log_level, "Node {} Type: Host", node_id);
                        break;
                    case cudaGraphNodeTypeGraph:
                        cudaGraph_t child_graph;
                        cudaGraphChildGraphNodeGetGraph(nodes[i], &child_graph);
                        logCudaGraphNodes(child_graph, log_level, node_id + "-");
                        break;
                    case cudaGraphNodeTypeEmpty:
                        spdlog::log(log_level, "Node {} Type: Empty", node_id);
                        break;
                    case cudaGraphNodeTypeEventRecord:
                        spdlog::log(log_level, "Node {} Type: EventRecord", node_id);
                        break;
                    default:
                        spdlog::log(log_level, "Node {} Type: Unknown", node_id);
                        break;
                }
            }
        } else {
            spdlog::warn("Graph {} No nodes have been recorded to graph", parent_node_id);
        }
    }

std::vector<size_t> getOptionalDefaultChannelMapping(std::vector<size_t> channel_mapping, size_t n_channels) {
    if (channel_mapping.empty()) {
        channel_mapping.resize(n_channels);
        std::iota(channel_mapping.begin(), channel_mapping.end(), 0);
    }
    return channel_mapping;
}

class KernelNode : public IKernelNode {
   protected:
    cudaKernelNodeParams _params;

   public:
    KernelNode(dim3 n_blocks, dim3 n_thread, size_t sharedMem, void* kernel, void** args) {
        _params.func = kernel;
        _params.gridDim = n_blocks;
        _params.blockDim = n_thread;
        _params.sharedMemBytes = sharedMem;
        _params.kernelParams = args;
        _params.extra = nullptr;  // Yes you need to set this to null. Took me 3h to debug the segfault the cudaGraphAddKernelNode call causes if it isn't set to null.
    }

    void
    addToGraph(cudaStream_t stream) override {
        cudaGraph_t graph;
        cudaStreamCaptureStatus capture_status;
        const cudaGraphNode_t* dependencies;
        size_t n_dependencies;
        gpuErrChk(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph, &dependencies, &n_dependencies));
        addToGraph(stream, graph, dependencies, n_dependencies);
    }

    void addToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        gpuErrChk(cudaGraphAddKernelNode(&_node, graph, dependencies, n_dependencies, &_params));
        gpuErrChk(cudaGraphKernelNodeGetParams(_node, &_params));  // adding the node to the graph seems to enrich the params with some additional info which we need to retreive, to enable subsequent updates
        gpuErrChk(cudaStreamUpdateCaptureDependencies(stream, &_node, 1, 1));
    }

    void addToGraph(cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        gpuErrChk(cudaGraphAddKernelNode(&_node, graph, dependencies, n_dependencies, &_params));
    }

    void update(cudaGraphExec_t graph_exec) override {
        gpuErrChk(cudaGraphExecKernelNodeSetParams(graph_exec, _node, &_params));
    }
    void updateKernelParamAt(int index, void* param, cudaGraphExec_t graph_exec) override {
        _params.kernelParams[index] = param;
        if (graph_exec) update(graph_exec);
    }
};

IKernelNode* IKernelNode::create(dim3 n_blocks, dim3 n_thread, size_t sharedMem, void* kernel, void** args, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) {
    auto instance = new KernelNode(n_blocks, n_thread, sharedMem, kernel, args);
    if (graph) instance->addToGraph(graph, dependencies, n_dependencies);
    return instance;
}
void IKernelNode::launchOrRecord(dim3 n_blocks, dim3 n_thread, size_t sharedMem, void* kernel, void** args, cudaStream_t stream, IKernelNode* instance, cudaStreamCaptureStatus capture_status) {
    if (stream && capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        gpuErrChk(cudaLaunchKernel(kernel, n_blocks, n_thread, args, sharedMem, stream));
    } else {
        instance = new KernelNode(n_blocks, n_thread, sharedMem, kernel, args);
        instance->addToGraph(stream);
    }
}

class MemCpyNode : public IMemCpyNode {
   protected:
    cudaMemcpy3DParms _params;
    const void* _src;
    void* _dst;

   public:
    MemCpyNode(void* dst, const void* src) : _src(src), _dst(dst) {}
    virtual ~MemCpyNode() {}
    virtual void addToGraph(cudaStream_t stream) override {
        if (stream == nullptr) return;
        cudaGraph_t graph;
        cudaStreamCaptureStatus capture_status;
        const cudaGraphNode_t* dependencies;
        size_t n_dependencies;
        gpuErrChk(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph, &dependencies, &n_dependencies));
        addToGraph(stream, graph, dependencies, n_dependencies);
    }

    virtual void addToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        addToGraph(graph, dependencies, n_dependencies);
        gpuErrChk(cudaStreamUpdateCaptureDependencies(stream, &_node, 1, 1));
    }
    virtual void addToGraph(cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        gpuErrChk(cudaGraphAddMemcpyNode(&_node, graph, dependencies, n_dependencies, &_params));
        gpuErrChk(cudaGraphMemcpyNodeGetParams(_node, &_params));  // adding the node to the graph seems to enrich the params with some additional info which we need to retreive, to enable subsequent updates
    }

    void update(cudaGraphExec_t graph_exec) override {
        gpuErrChk(cudaGraphExecMemcpyNodeSetParams(graph_exec, _node, &_params));
    }

    virtual void updateSrcPtr(const void* src, cudaGraphExec_t graph_exec = nullptr) override {
        _src = src;
        _params.srcPtr.ptr = (void*)src;
        cudaGraphMemcpyNodeSetParams(_node, &_params);
        if (graph_exec) update(graph_exec);
    }

    virtual void updateDstPtr(void* dst, cudaGraphExec_t graph_exec = nullptr) override {
        _dst = dst;
        _params.dstPtr.ptr = dst;
        cudaGraphMemcpyNodeSetParams(_node, &_params);
        if (graph_exec) update(graph_exec);
    }
    virtual const void* getSrcPtr() { return _src; }
    virtual void* getDstPtr() { return _dst; }
};

class MemCpy1DNode : public MemCpyNode {
   public:
    MemCpy1DNode(void* dst, const void* src, size_t element_width, size_t n_elements, cudaMemcpyKind kind) : MemCpyNode(dst, src) {
        _params.srcArray = NULL;
        _params.srcPos = make_cudaPos(0, 0, 0);
        _params.srcPtr = make_cudaPitchedPtr((void*)src, n_elements * element_width, n_elements, 1);
        _params.dstArray = NULL;
        _params.dstPos = make_cudaPos(0, 0, 0);
        _params.dstPtr = make_cudaPitchedPtr(dst, n_elements * element_width, n_elements, 1);
        _params.extent = make_cudaExtent(n_elements * element_width, 1, 1);
        _params.kind = kind;
    }
};

IMemCpyNode* IMemCpyNode::create1D(void* dst, const void* src, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) {
    auto instance = new MemCpy1DNode(dst, src, element_width, n_elements, kind);
    if (graph) instance->addToGraph(graph, dependencies, n_dependencies);
    return instance;
}
void IMemCpyNode::launchOrRecord1D(void* dst, const void* src, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaStream_t stream, IMemCpyNode* instance, cudaStreamCaptureStatus capture_status) {
    if (stream && capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        gpuErrChk(cudaMemcpyAsync(dst, src, n_elements * element_width, kind, stream));
    } else {
        instance = new MemCpy1DNode(dst, src, element_width, n_elements, kind);
        instance->addToGraph(stream);
    }
}

class MemCpy2DNode : public MemCpyNode {
   public:
    MemCpy2DNode(void* dst, size_t dpitch, const void* src, size_t spitch, size_t element_width, size_t n_elements, cudaMemcpyKind kind) : MemCpyNode(dst, src) {
        _params.srcArray = NULL;
        _params.srcPos = make_cudaPos(0, 0, 0);
        _params.srcPtr = make_cudaPitchedPtr((void*)src, spitch, element_width, n_elements);
        _params.dstArray = NULL;
        _params.dstPos = make_cudaPos(0, 0, 0);
        _params.dstPtr = make_cudaPitchedPtr(dst, dpitch, element_width, n_elements);
        _params.extent = make_cudaExtent(element_width, n_elements, 1);
        _params.kind = kind;
    }
};
IMemCpyNode* IMemCpyNode::create2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) {
    auto instance = new MemCpy2DNode(dst, dpitch, src, spitch, element_width, n_elements, kind);
    if (graph) instance->addToGraph(graph, dependencies, n_dependencies);
    return instance;
}
void IMemCpyNode::launchOrRecord2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t element_width, size_t n_elements, cudaMemcpyKind kind, cudaStream_t stream, IMemCpyNode* instance, cudaStreamCaptureStatus capture_status) {
    if (stream && capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        gpuErrChk(cudaMemcpy2DAsync(dst, dpitch, src, spitch, element_width, n_elements, kind, stream));
    } else {
        instance = new MemCpy2DNode(dst, dpitch, src, spitch, element_width, n_elements, kind);
        instance->addToGraph(stream);
    }
}

class IMultiMemCpyNodeWrapper : public MemCpyNode {
   protected:
    std::vector<MemCpy2DNode*> _nodes;
    size_t _element_width;
    size_t _n_frames;
    size_t _n_channels;
    std::vector<size_t> _channel_mapping;
    cudaMemcpyKind _kind;
    cudaGraph_t _child_graph;
    cudaGraphNode_t _child_graph_node;

   public:
    IMultiMemCpyNodeWrapper(void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind) : MemCpyNode(dst, src), _element_width(element_width), _n_frames(n_frames), _n_channels(n_channels), _channel_mapping(channel_mapping), _kind(kind) {}

    void* get_interleaved_ptr(const void* ptr, size_t channel) { return ((char*)ptr) + channel * _element_width; }
    void* get_planar_ptr(const void* ptr, size_t channel) { return ((char*)ptr) + channel * _n_frames * _element_width; }
    void* get_segmented_ptr(const void* ptr, size_t channel) { return ((float**)ptr)[channel]; }
    size_t get_interleaved_pitch() { return _n_channels * _element_width; }
    size_t get_continuous_pitch() { return _element_width; }

    virtual void* get_dst_ptr(const void* dst, size_t channel) = 0;
    virtual void* get_src_ptr(const void* src, size_t channel) = 0;
    virtual size_t get_dst_pitch() = 0;
    virtual size_t get_src_pitch() = 0;
    virtual cudaGraphNode_t getNode() { return _child_graph_node; }
    virtual cudaGraphNode_t* getNodePtr() { return &_child_graph_node; }

    // had to move this from constructor into member function to avoid the "pure virtual function call" error
    void setup() {
        for (size_t i = 0; i < _channel_mapping.size(); i++) {
            _nodes.push_back(new MemCpy2DNode(get_dst_ptr(_dst, i), get_dst_pitch(), get_src_ptr(_src, i), get_src_pitch(), _element_width, _n_frames, _kind));
        }
    }

    void addToGraph(cudaStream_t stream) override {
        if (stream == nullptr) return;
        cudaGraph_t graph;
        cudaStreamCaptureStatus capture_status;
        const cudaGraphNode_t* dependencies;
        size_t n_dependencies;
        gpuErrChk(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph, &dependencies, &n_dependencies));
        addToGraph(stream, graph, dependencies, n_dependencies);
    }
    void addToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->addToGraph(stream, graph, dependencies, n_dependencies);
        }
    }
    void addToGraph(cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        gpuErrChk(cudaGraphCreate(&_child_graph, 0));
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->addToGraph(_child_graph, nullptr, 0);
        }
        gpuErrChk(cudaGraphAddChildGraphNode(&_child_graph_node, graph, dependencies, n_dependencies, _child_graph));
    }

    void update(cudaGraphExec_t graph_exec) override {
        throw std::runtime_error("Updating MemCpy2d nodes of an instantiated graph doesn't seem to work, even if I don't see which exec update limitation I'm violating");
        gpuErrChk(cudaGraphExecChildGraphNodeSetParams(graph_exec, _child_graph_node, _child_graph));
    }

    virtual void updateSrcPtr(const void* src, cudaGraphExec_t graph_exec = nullptr) override {
        _src = src;
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->updateSrcPtr(get_src_ptr(src, i));
        }
        if (graph_exec) update(graph_exec);
    }

    virtual void updateDstPtr(void* dst, cudaGraphExec_t graph_exec = nullptr) override {
        _dst = dst;
        for (size_t i = 0; i < _nodes.size(); i++) {
            auto prt = get_dst_ptr(dst, i);
            _nodes[i]->updateDstPtr(prt);
        }
        if (graph_exec) update(graph_exec);
    }
};

class MemCpyInterleaved2SegmentedNode : public IMultiMemCpyNodeWrapper {
   public:
    MemCpyInterleaved2SegmentedNode(void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind) : IMultiMemCpyNodeWrapper(dst, src, element_width, n_frames, n_channels, channel_mapping, kind) {
        setup();
    }
    void* get_src_ptr(const void* src, size_t channel) override { return get_interleaved_ptr(src, _channel_mapping[channel]); }
    void* get_dst_ptr(const void* dst, size_t channel) override { return get_segmented_ptr(dst, channel); }
    size_t get_src_pitch() override { return get_interleaved_pitch(); }
    size_t get_dst_pitch() override { return get_continuous_pitch(); }
};

class MemCpySegmented2InterleavedNode : public IMultiMemCpyNodeWrapper {
   public:
    MemCpySegmented2InterleavedNode(void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind) : IMultiMemCpyNodeWrapper(dst, src, element_width, n_frames, n_channels, channel_mapping, kind) {
        setup();
    }

    void* get_src_ptr(const void* src, size_t channel) override { return get_segmented_ptr(src, _channel_mapping[channel]); }
    void* get_dst_ptr(const void* dst, size_t channel) override { return get_interleaved_ptr(dst, channel); }
    size_t get_src_pitch() override { return get_continuous_pitch(); }
    size_t get_dst_pitch() override { return get_interleaved_pitch(); }
};

IMemCpyNode* IMemCpyNode::createMulti(MultiMemcpyType type, void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t n_dependencies) {
    MemCpyNode* instance;
    switch (type) {
        case MultiMemcpyType::Interleaved2Segmented:
            instance = new MemCpyInterleaved2SegmentedNode(dst, src, element_width, n_frames, n_channels, getOptionalDefaultChannelMapping(channel_mapping, n_channels), kind);
            break;
        case MultiMemcpyType::Segmented2Interleaved:
            instance = new MemCpySegmented2InterleavedNode(dst, src, element_width, n_frames, n_channels, getOptionalDefaultChannelMapping(channel_mapping, n_channels), kind);
            break;
        default:
            throw std::runtime_error("Invalid MultiMemcpyType");
    }
    if (graph) instance->addToGraph(graph, dependencies, n_dependencies);
    return instance;
}

void IMemCpyNode::launchOrRecordMulti(MultiMemcpyType type, void* dst, const void* src, size_t element_width, size_t n_frames, size_t n_channels, std::vector<size_t> channel_mapping, cudaMemcpyKind kind, cudaStream_t stream, IMemCpyNode* instance, cudaStreamCaptureStatus capture_status) {
    channel_mapping = getOptionalDefaultChannelMapping(channel_mapping, n_channels);
    IMultiMemCpyNodeWrapper* node = static_cast<IMultiMemCpyNodeWrapper*>(IMemCpyNode::createMulti(type, dst, src, element_width, n_frames, n_channels, channel_mapping, kind));
    if (stream && capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        gpuErrChk(cudaStreamSynchronize(stream));
        std::vector<cudaStream_t> streams(n_channels);
        for (size_t i = 0; i < channel_mapping.size(); i++) {
            gpuErrChk(cudaStreamCreate(&streams[i]));
            gpuErrChk(cudaMemcpy2DAsync(node->get_dst_ptr(dst, i), node->get_dst_pitch(), node->get_src_ptr(src, i), node->get_src_pitch(), element_width, n_frames, kind, streams[i]));
        }
        std::for_each(streams.begin(), streams.end(), cudaStreamSynchronize);
        std::for_each(streams.begin(), streams.end(), cudaStreamDestroy);
        delete node;
    } else {
        instance = node;
        instance->addToGraph(stream);
    }
}