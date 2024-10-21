
#include <errno.h>
#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <cuda_ext.cuh>
#include <future>
#include <kernels.cuh>
#include <log.hpp>
#include <ringbuffer.hpp>
#include <stdexcept>
#include <thread>

#include "block_buffer.cuh"
#include "gpu.cuh"
#include "gpu_signal_graph.cuh"
#include "spdlog/spdlog.h"

class GpuSignalVertex : public IGpuSignalVertex {
   protected:
    bool _owns_dest_buffer;
    BufferRack _src_ptr;
    BufferRack _dest_ptr;

    std::vector<IGpuSignalVertex*> _parents;
    std::vector<IGpuSignalVertex*> _children;

    void addToGraph(cudaGraph_t setup_graph, cudaGraph_t process_graph) {
        std::vector<cudaGraphNode_t> dependencies;
        for (IGpuSignalVertex* parent : _parents) {
            dependencies.push_back(parent->getProcessNode());
        }
        addToGraph(setup_graph, process_graph, dependencies.data(), dependencies.size());
    }

    virtual void addToGraph(cudaGraph_t setup_graph, cudaGraph_t process_graph, cudaGraphNode_t* dependencies, size_t n_dependencies) = 0;

   public:
    GpuSignalVertex(IGpuSignalVertex* parent, std::vector<Buffer*> src_ptr = {}, std::vector<Buffer*> dst_ptr = {}) : _parents(parent ? std::vector<IGpuSignalVertex*>{parent} : std::vector<IGpuSignalVertex*>{}), _src_ptr(src_ptr), _dest_ptr(dst_ptr) {
    }
    GpuSignalVertex(std::vector<IGpuSignalVertex*> parents, std::vector<Buffer*> src_ptr = {}, std::vector<Buffer*> dst_ptr = {}) : _parents(parents), _src_ptr(src_ptr), _dest_ptr(dst_ptr) {
    }

    ~GpuSignalVertex() {
    }

    virtual const BufferRackSpecs& getInputSpecs() { return _src_ptr.getSpecs(); }
    virtual const BufferRackSpecs& getOutputSpecs() { return _dest_ptr.getSpecs(); }
    virtual void setSrcPtr(std::vector<Buffer*> src_ptr, cudaGraphExec_t graph_exec) = 0;
    virtual void setDestPtr(std::vector<Buffer*> dest_ptr, cudaGraphExec_t graph_exec) = 0;
    std::vector<IGpuSignalVertex*>& getParents() override { return _parents; }
    std::vector<IGpuSignalVertex*>& getChildren() override { return _children; }
    virtual void setup(cudaGraph_t setup_graph, cudaGraph_t process_graph, size_t n_proc_frames, std::vector<Buffer*>& buffer_ptrs) = 0;
};

class GpuSignalCopyVertex : public GpuSignalVertex {
   private:
    bool _owns_dest_buffer;
    size_t _n_channels;
    IMemCpyNode* _node;

    void addToGraph(cudaGraph_t setup_graph, cudaGraph_t process_graph) {
        std::vector<cudaGraphNode_t> dependencies;
        for (IGpuSignalVertex* parent : _parents) {
            dependencies.push_back(parent->getProcessNode());
        }
        addToGraph(setup_graph, process_graph, dependencies.data(), dependencies.size());
    }

    void addToGraph(cudaGraph_t setup_graph, cudaGraph_t process_graph, cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        throw std::runtime_error("Not implemented");
    }

   public:
    GpuSignalCopyVertex(IMemCpyNode* node, std::vector<Buffer*> src_ptr, std::vector<Buffer*> dst_ptr, size_t n_channels, std::vector<IGpuSignalVertex*> parents = {}, std::vector<IGpuSignalVertex*> children = {})
        : GpuSignalVertex(parents, src_ptr, dst_ptr), _node(node), _n_channels(n_channels) {
    }
    ~GpuSignalCopyVertex() {
    }

    const std::vector<Buffer*>& getSrcPtr() override {
        return _src_ptr.getBuffers();
    }
    const std::vector<Buffer*>& getDestPtr() override {
        return _dest_ptr.getBuffers();
    }
    void setSrcPtr(const std::vector<Buffer*> src_ptr, cudaGraphExec_t graph_exec) override {
        _src_ptr.set(src_ptr);
        _node->updateExecSrcPtr(_src_ptr.getDataMod(), graph_exec);
    }
    void setDestPtr(std::vector<Buffer*> dest_ptr, cudaGraphExec_t graph_exec) override {
        _dest_ptr.set(dest_ptr);
        _node->updateExecDstPtr(_dest_ptr.getDataMod(), graph_exec);
    }
    cudaGraphNode_t getProcessNode() override { return _node->getNode(); }
    cudaGraphNode_t* getProcessNodePtr() override { return _node->getNodePtr(); }
    size_t getIncomingChannelCount() override { return _n_channels; }
    size_t getOutgoingChannelCount() override { return _n_channels; }

    void setup(cudaGraph_t setup_graph, cudaGraph_t process_graph, size_t n_proc_frames, std::vector<Buffer*>& buffer_ptrs) override {
        throw std::runtime_error("Not implemented");
    }
};

class GpuSignalFxVertex : public GpuSignalVertex {
   private:
    IGpuFx* _fx;
    cudaGraphNode_t _setup_node;
    cudaGraphNode_t _process_node;
    cudaGraphNode_t _post_process_node;

    void addToGraph(cudaGraph_t setup_graph, cudaGraph_t process_graph, cudaGraphNode_t* dependencies, size_t n_dependencies) override {
        cudaGraphAddChildGraphNode(&_setup_node, setup_graph, nullptr, 0, _fx->recordSetupGraph());
        cudaGraphAddChildGraphNode(&_process_node, process_graph, dependencies, n_dependencies, _fx->recordProcessGraph(&_dest_ptr, &_src_ptr));
        cudaGraphAddChildGraphNode(&_post_process_node, process_graph, &_process_node, 1, _fx->recordPostProcessGraph());
    }

   public:
    GpuSignalFxVertex(IGpuFx* fx, IGpuSignalVertex* parent) : GpuSignalVertex(parent), _fx(fx) {
    }
    GpuSignalFxVertex(IGpuFx* fx, std::vector<IGpuSignalVertex*> parents) : GpuSignalVertex(parents), _fx(fx) {
    }

    ~GpuSignalFxVertex() {
    }

    IGpuFx* getFx() { return _fx; }
    const std::vector<Buffer*>& getSrcPtr() override { return _src_ptr.getBuffers(); }
    const std::vector<Buffer*>& getDestPtr() override { return _dest_ptr.getBuffers(); }
    void setSrcPtr(const std::vector<Buffer*> src_ptr, cudaGraphExec_t graph_exec) override {
        _src_ptr.set(src_ptr);
        _fx->updateBufferPtrs(nullptr, &_dest_ptr, &_src_ptr);
    }
    void setDestPtr(std::vector<Buffer*> dest_ptr, cudaGraphExec_t graph_exec) override {
        _dest_ptr.set(dest_ptr);
        _fx->updateBufferPtrs(nullptr, &_dest_ptr, &_src_ptr);
    }
    cudaGraphNode_t getProcessNode() override { return _process_node; }
    cudaGraphNode_t* getProcessNodePtr() override { return &_process_node; }

    size_t getIncomingChannelCount() override {
        return std::accumulate(_parents.begin(), _parents.end(), 0, [](size_t a, IGpuSignalVertex* b) { return a + b->getOutgoingChannelCount(); });
    }

    size_t getOutgoingChannelCount() override {
        return _fx->getOutChannelCount();
    }

    void setup(cudaGraph_t setup_graph, cudaGraph_t process_graph, size_t n_proc_frames, std::vector<Buffer*>& buffer_ptrs) {
        _fx->configure(n_proc_frames, getIncomingChannelCount());

        bool has_multiple_parents = _parents.size() > 1;
        bool has_parent_with_multiple_children = std::any_of(_parents.begin(), _parents.end(), [](IGpuSignalVertex* parent) { return parent->getChildren().size() > 1; });
        bool output_channel_counts_mismatch = _parents.front()->getOutgoingChannelCount() != getOutgoingChannelCount();

        if (has_multiple_parents) {
            BufferRackSpecs specs;
            std::vector<Buffer*> buffers;
            for (size_t i = 0; i < _parents.size(); i++) {
                buffers.insert(buffers.end(), _parents[i]->getDestPtr().begin(), _parents[i]->getDestPtr().end());
            }
            _src_ptr.set(buffers);
        } else {
            _src_ptr = _parents.front()->getDestPtr();
        }

        if (_children.empty() || has_multiple_parents || has_parent_with_multiple_children || output_channel_counts_mismatch) {
            // can not use in place processing -> allocate new destination buffer
            _dest_ptr.set(_fx->getOutputSpecs());
            buffer_ptrs.insert(buffer_ptrs.end(), _dest_ptr.getBuffers().begin(), _dest_ptr.getBuffers().end());
        } else {
            // use in place processing
            _dest_ptr.set(_src_ptr.getBuffers());
        }

        GpuSignalVertex::addToGraph(setup_graph, process_graph);
    }
};

class GpuSignalGraph : public IGpuSignalGraph {
   private:
    std::vector<GpuSignalVertex*> _roots;
    std::vector<GpuSignalVertex*> _vertices;
    std::vector<GpuSignalFxVertex*> _fx_vertices;
    std::vector<GpuSignalVertex*> _leaves;
    std::vector<Buffer*> _buffer_ptrs;
    std::vector<GpuSignalVertex*> _input_vertices;
    std::vector<GpuSignalVertex*> _output_vertices;

    std::vector<RingBuffer*> _input_ringbuffers;
    std::vector<RingBuffer*> _output_ringbuffers;
    std::vector<float*> _input_buffers;
    std::vector<float*> _output_buffers;
    float* _spinning_buffer;

    size_t _n_proc_frames;
    size_t _n_in_channels;
    size_t _n_out_channels;
    bool _async;

    cudaGraph_t _setup_graph;
    cudaGraph_t _process_graph;
    cudaGraph_t _spin_graph;
    cudaGraphExec_t _setup_graph_exec;
    cudaGraphExec_t _process_graph_exec;
    cudaGraphExec_t _spin_graph_exec;
    cudaStream_t _stream;
    cudaEvent_t spin_start, proc_start, proc_stop;

    void updateLeaves() {
        _leaves.clear();
        for (GpuSignalVertex* vertices : _vertices) {
            if (vertices->getChildren().empty()) {
                _leaves.push_back(vertices);
            }
        }
    }

    void _processSync(const std::vector<float*>& dst_bufs, const std::vector<float*>& src_bufs) {
        for (auto vertex : _fx_vertices) {
            vertex->getFx()->updateSoftParams(_process_graph_exec, vertex->getProcessNode());
        }
        for (size_t i = 0; i < _input_vertices.size(); i++) {
            _input_vertices[i]->setSrcPtr({Buffer::create(src_bufs[i], BufferSpecs(MemoryContext::HOST, _n_proc_frames))}, _process_graph_exec);
        }
        for (size_t i = 0; i < _output_vertices.size(); i++) {
            _output_vertices[i]->setDestPtr({Buffer::create(dst_bufs[i], BufferSpecs(MemoryContext::HOST, _n_proc_frames))}, _process_graph_exec);
        }
        _process();
    }

    void _processAsync(const std::vector<float*>& dst_bufs, const std::vector<float*>& src_bufs) {
        for (size_t i = 0; i < _n_in_channels; i++) {
            while (_input_ringbuffers[i]->write(src_bufs[i], _n_proc_frames) <= 0) {
            }
        }
        for (size_t i = 0; i < _n_out_channels; i++) {
            while (_output_ringbuffers[i]->read(dst_bufs[i], _n_proc_frames) <= 0) {
            }
        }
    }

    void asyncLoop() {
        for (auto vertex : _fx_vertices) {
            vertex->getFx()->updateSoftParams(_process_graph_exec, vertex->getProcessNode());
        }

        // using a spin kernel to keep the CUDA scheduler awake while waiting for new data
        // at a higher CUDA version this should be replaced with a WHILE node at the start of the process graph
        for (size_t i = 0; i < _n_in_channels; i++) {
            float* input_ptr;
            while (_input_ringbuffers[i]->getReadPtr(&input_ptr, _n_proc_frames) <= 0) {
                gpuErrChk(cudaStreamSynchronize(_stream));
                gpuErrChk(cudaGraphLaunch(_spin_graph_exec, _stream));
            }
            _input_vertices[i]->setSrcPtr({Buffer::create(input_ptr, BufferSpecs(MemoryContext::HOST, _n_proc_frames))}, _process_graph_exec);
        }
        for (size_t i = 0; i < _n_out_channels; i++) {
            float* output_ptr;
            while (_output_ringbuffers[i]->getWritePtr(&output_ptr, _n_proc_frames) <= 0) {
                gpuErrChk(cudaStreamSynchronize(_stream));
                gpuErrChk(cudaGraphLaunch(_spin_graph_exec, _stream));
            }
            _output_vertices[i]->setDestPtr({Buffer::create(output_ptr, BufferSpecs(MemoryContext::HOST, _n_proc_frames))}, _process_graph_exec);
        }
        _process();
        for (size_t i = 0; i < _n_out_channels; i++) {
            while (_output_ringbuffers[i]->advanceWriteIndex(_n_proc_frames) <= 0) {
            }
        }
        for (size_t i = 0; i < _n_in_channels; i++) {
            while (_input_ringbuffers[i]->advanceReadIndex(_n_proc_frames) <= 0) {
            }
        }
    }

    void _process() {
        gpuErrChk(cudaEventRecord(proc_start, _stream));
        gpuErrChk(cudaGraphLaunch(_process_graph_exec, _stream));
        gpuErrChk(cudaEventRecord(proc_stop, _stream));
        gpuErrChk(cudaEventSynchronize(proc_stop));
        float proc_time = 0;
        gpuErrChk(cudaEventElapsedTime(&proc_time, proc_start, proc_stop));
        if (proc_time > _n_proc_frames / 48.0f) {
            spdlog::warn("Processing time: {} ms", std::to_string(proc_time));
        }
    }

   public:
    GpuSignalGraph() {
    }

    ~GpuSignalGraph() {
        for (GpuSignalVertex* vertex : _vertices) {
            delete vertex;
        }
        for (GpuSignalVertex* vertex : _input_vertices) {
            delete vertex;
        }
        for (GpuSignalVertex* vertex : _output_vertices) {
            delete vertex;
        }
        for (RingBuffer* buffer : _input_ringbuffers) {
            delete buffer;
        }
        for (RingBuffer* buffer : _output_ringbuffers) {
            delete buffer;
        }
        for (float* buffer : _input_buffers) {
            delete[] buffer;
        }
        for (float* buffer : _output_buffers) {
            delete[] buffer;
        }
    }

    size_t getInputChannelCount() {
        return _n_in_channels;
    }

    size_t getOutputChannelCount() {
        return _n_out_channels;
    }

    void setup(size_t n_proc_frames, size_t n_in_channels, size_t n_out_channels, bool async) override {
        _n_proc_frames = n_proc_frames;
        _n_in_channels = n_in_channels;
        _async = async;

        gpuErrChk(cudaEventCreate(&spin_start));
        gpuErrChk(cudaEventCreate(&proc_start));
        gpuErrChk(cudaEventCreate(&proc_stop));
        gpuErrChk(cudaGraphCreate(&_setup_graph, 0));
        gpuErrChk(cudaGraphCreate(&_process_graph, 0));

        // the spin kernel copies splin_buffer_size samples in place
        // too small buffer size will overload the scheduler leading to bad performance of the process graph
        // too large buffer size can delay the next process graph execution too much (unfortunately CUDA 11.4 does not have a good option to terminate a kernel from host)
        // 2 * n_proc_frames has been found to be a good compromise
        size_t spin_buffer_size = std::min(768, 2 * static_cast<int>(n_proc_frames));
        gpuErrChk(cudaGraphCreate(&_spin_graph, 0));
        gpuErrChk(cudaMalloc(&_spinning_buffer, spin_buffer_size * sizeof(float)));
        IKernelNode::create(1, spin_buffer_size, 0, (void*)spin_kernel, new void*[3]{&_spinning_buffer, &_spinning_buffer, &_n_proc_frames}, _spin_graph);
        gpuErrChk(cudaGraphInstantiate(&_spin_graph_exec, _spin_graph, NULL, NULL, 0));

        std::vector<GpuSignalVertex*> queue;
        std::vector<GpuSignalVertex*> orphans;
        std::vector<IGpuSignalVertex*> orphans_i;
        std::copy_if(_vertices.begin(), _vertices.end(), std::back_inserter(orphans), [](GpuSignalVertex* vertex) { return vertex->getParents().empty(); });
        std::transform(orphans.begin(), orphans.end(), std::back_inserter(orphans_i), [](GpuSignalVertex* vertex) { return static_cast<GpuSignalFxVertex*>(vertex); });
        std::copy(_vertices.begin(), _vertices.end(), std::back_inserter(queue));
        std::transform(_vertices.begin(), _vertices.end(), std::back_inserter(_fx_vertices), [](GpuSignalVertex* vertex) { return static_cast<GpuSignalFxVertex*>(vertex); });

        for (size_t i = 0; i < _n_in_channels; i++) {
            BufferRack src_ptr(BufferSpecs(MemoryContext::HOST, n_proc_frames));
            BufferRack dest_ptr(BufferSpecs(MemoryContext::DEVICE, n_proc_frames));
            auto input_vertex = new GpuSignalCopyVertex(
                IMemCpyNode::create1D(dest_ptr.getDataMod(), src_ptr.getDataMod(), sizeof(float), _n_proc_frames, cudaMemcpyHostToDevice, _process_graph),
                src_ptr.getBuffers(), dest_ptr.getBuffers(), 1, {}, orphans_i);
            src_ptr.deallocateBuffers();
            for (Buffer* buffer : dest_ptr.getBuffers()) {
                _buffer_ptrs.push_back(buffer);
            }
            _input_vertices.push_back(input_vertex);
            _roots.push_back(input_vertex);
        }

        for (GpuSignalVertex* orphan : orphans) {
            orphan->getParents().insert(orphan->getParents().end(), _input_vertices.begin(), _input_vertices.end());
        }

        std::vector<GpuSignalVertex*> visited = std::vector<GpuSignalVertex*>{_roots};
        size_t queue_index = 0;
        while (!queue.empty()) {
            auto vertex = queue.at(queue_index);
            if (std::all_of(vertex->getParents().begin(), vertex->getParents().end(), [&visited](IGpuSignalVertex* parent) { return std::find(visited.begin(), visited.end(), parent) != visited.end(); })) {
                vertex->setup(_setup_graph, _process_graph, _n_proc_frames, _buffer_ptrs);
                queue.erase(queue.begin() + queue_index);
                visited.push_back(vertex);
                queue_index = 0;
            } else {
                queue_index++;
            }
        }

        if (_leaves.empty()) {
            std::copy(_input_vertices.begin(), _input_vertices.end(), std::back_inserter(_leaves));
        }
        size_t n_leave_outputs = std::accumulate(_leaves.begin(), _leaves.end(), 0, [](size_t sum, GpuSignalVertex* v) { return sum + v->getOutgoingChannelCount(); });
        _n_out_channels = n_out_channels == 0 ? n_leave_outputs : n_out_channels;

        if (n_leave_outputs != _n_out_channels) {
            throw std::runtime_error("Leave nodes produce " + std::to_string(n_leave_outputs) + " output channels, but " + std::to_string(_n_out_channels) + " are expected");
        } else {
            size_t i = 0;
            for (GpuSignalVertex* leave : _leaves) {
                for (size_t c = 0; c < leave->getOutgoingChannelCount(); c++) {
                    BufferRack dest_ptr(BufferSpecs(MemoryContext::HOST, n_proc_frames));
                    _output_vertices.push_back(
                        new GpuSignalCopyVertex(
                            IMemCpyNode::create1D(dest_ptr.getDataMod(), leave->getDestPtr()[c]->getDataMod(), sizeof(float), _n_proc_frames, cudaMemcpyDeviceToHost, _process_graph, leave->getProcessNodePtr(), 1),
                            {leave->getDestPtr()[c]}, dest_ptr.getBuffers(), 1, {leave}, {}));
                    dest_ptr.deallocateBuffers();
                    i++;
                }
            }
        }

        gpuErrChk(cudaStreamCreate(&_stream));
        gpuErrChk(cudaGraphInstantiate(&_setup_graph_exec, _setup_graph, NULL, NULL, 0));
        gpuErrChk(cudaGraphLaunch(_setup_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));

        logCudaGraphNodes(_process_graph, spdlog::level::debug);
        gpuErrChk(cudaGraphInstantiate(&_process_graph_exec, _process_graph, NULL, NULL, 0));

        if (_async) {
            for (size_t i = 0; i < _n_in_channels; i++) {
                _input_ringbuffers.push_back(RingBuffer::create(n_proc_frames, 16));
                _input_buffers.push_back(new float[n_proc_frames]);
            }
            for (size_t i = 0; i < _n_out_channels; i++) {
                _output_ringbuffers.push_back(RingBuffer::create(n_proc_frames, 16, 0));
                _output_buffers.push_back(new float[n_proc_frames]);
            }
            std::thread([this]() {
                while (true) {
                    asyncLoop();
                }
            }).detach();
        }
    }

    void process(const std::vector<float*>& dst_bufs, const std::vector<float*>& src_bufs) override {
        if (_async) {
            _processAsync(dst_bufs, src_bufs);
        } else {
            _processSync(dst_bufs, src_bufs);
        }
    }

    void teardown() override {
        for (Buffer* buffer : _buffer_ptrs) {
            gpuErrChk(cudaFree(buffer->getDataMod()));
        }
        gpuErrChk(cudaStreamDestroy(_stream));

        gpuErrChk(cudaEventDestroy(spin_start));
        gpuErrChk(cudaEventDestroy(proc_start));
        gpuErrChk(cudaEventDestroy(proc_stop));

        gpuErrChk(cudaGraphDestroy(_process_graph));
        gpuErrChk(cudaGraphExecDestroy(_process_graph_exec));
        gpuErrChk(cudaGraphDestroy(_setup_graph));
        gpuErrChk(cudaGraphExecDestroy(_setup_graph_exec));
        gpuErrChk(cudaGraphDestroy(_spin_graph));
        gpuErrChk(cudaGraphExecDestroy(_spin_graph_exec));
    }

    IGpuSignalVertex* add(IGpuFx* fx, IGpuSignalVertex* parent = nullptr) override {
        if (parent == nullptr && !_leaves.empty()) {
            parent = _leaves.front();
        }

        GpuSignalFxVertex* vertex = new GpuSignalFxVertex(fx, parent);
        if (parent != nullptr) {
            vertex->getChildren().insert(vertex->getChildren().end(), parent->getChildren().begin(), parent->getChildren().end());
            parent->getChildren().clear();
            parent->getChildren().push_back(vertex);
        }
        _vertices.push_back(vertex);
        updateLeaves();

        return vertex;
    }

    IGpuSignalVertex* addRoot(IGpuFx* fx) override {
        GpuSignalFxVertex* vertex = new GpuSignalFxVertex(fx, nullptr);
        _vertices.push_back(vertex);
        updateLeaves();

        return vertex;
    }

    std::vector<IGpuSignalVertex*> split(std::vector<IGpuFx*> fxs, IGpuSignalVertex* parent = nullptr) override {
        if (parent == nullptr && !_leaves.empty()) {
            parent = _leaves.front();
        }

        std::vector<IGpuSignalVertex*> vertices;
        for (auto fx : fxs) {
            GpuSignalFxVertex* vertex = new GpuSignalFxVertex(fx, parent);
            if (parent != nullptr) parent->getChildren().push_back(vertex);
            vertices.push_back(vertex);
            _vertices.push_back(vertex);
        }
        updateLeaves();
        return vertices;
    }

    IGpuSignalVertex* merge(IGpuFx* fx, std::vector<IGpuSignalVertex*> parents) override {
        GpuSignalFxVertex* vertex = new GpuSignalFxVertex(fx, parents);

        // std::vector<IGpuSignalVertex*> parents_with_children;
        // std::copy_if(parents.begin(), parents.end(), std::back_inserter(parents_with_children), [](IGpuSignalVertex* parent) { return !parent->getChildren().empty(); });
        // IGpuSignalVertex* parent_with_children = nullptr;
        // if (parents_with_children.size() > 1) {
        //     throw std::runtime_error("More than one parent has children. Cannot merge.");
        // } else if (parents_with_children.size() == 1) {
        //     parent_with_children = parents_with_children.front();
        //     vertex->getChildren().insert(vertex->getChildren().end(), parent_with_children->getChildren().begin(), parent_with_children->getChildren().end());
        // }

        for (auto parent : parents) {
            parent->getChildren().clear();
            parent->getChildren().push_back(vertex);
        }
        _vertices.push_back(vertex);

        updateLeaves();
        return vertex;
    }
};

IGpuSignalGraph* IGpuSignalGraph::createGpuSignalGraph() {
    return new GpuSignalGraph();
}
