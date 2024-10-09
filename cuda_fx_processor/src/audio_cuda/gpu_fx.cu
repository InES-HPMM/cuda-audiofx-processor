#include <cuda_ext.cuh>

#include "gpu_fx.cuh"
#include "spdlog/spdlog.h"

class GpuFx : public IGpuFx {
   private:
    cudaGraph_t _setup_graph = nullptr;
    cudaGraph_t _process_graph = nullptr;
    cudaGraph_t _post_process_graph = nullptr;
    bool _destroy_setup_graph = false;
    bool _destroy_process_graph = false;
    bool _destroy_post_process_graph = false;
    bool _has_post_processing;

    cudaStream_t getRecordingStream() {
        cudaStream_t stream;
        gpuErrChk(cudaStreamCreate(&stream));
        gpuErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureMode::cudaStreamCaptureModeGlobal));
        return stream;
    }
    cudaGraph_t recordStreamToGraph(cudaStream_t stream, cudaGraph_t* graph) {
        gpuErrChk(cudaGraphCreate(graph, 0));
        gpuErrChk(cudaStreamEndCapture(stream, graph));
        gpuErrChk(cudaStreamDestroy(stream));
        return *graph;
    }
    cudaGraph_t createGraphWithEmpyNode(cudaGraph_t* graph) {
        gpuErrChk(cudaGraphCreate(graph, 0));
        cudaGraphNode_t emptyNode;
        gpuErrChk(cudaGraphAddEmptyNode(&emptyNode, *graph, nullptr, 0));
        return *graph;
    }

   protected:
    std::string _name;
    size_t _n_in_channels;
    size_t _n_out_channels;
    size_t _n_proc_channels;
    size_t _n_proc_frames;
    size_t _n_proc_samples;

    BufferRackSpecs _input_specs;
    BufferRackSpecs _output_specs;

    virtual void allocateBuffers() = 0;
    virtual void deallocateBuffers() = 0;

   public:
    GpuFx(std::string name, bool has_post_processing = true) : _name(name), _has_post_processing(has_post_processing) {}
    virtual ~GpuFx() {};

    std::string getName() {
        return _name;
    }
    size_t getInChannelCount() { return _n_in_channels; }
    size_t getOutChannelCount() { return _n_out_channels; }
    size_t getProcSampleCount() { return _n_proc_samples; }
    size_t getOutSampleCount() { return _n_proc_frames * _n_out_channels; }

    BufferRackSpecs getInputSpecs() { return _input_specs; }
    BufferRackSpecs getOutputSpecs() { return _output_specs; }

    virtual void configure(size_t n_proc_frames, size_t n_in_channels = 0, size_t n_out_channels = 0) {
        if (n_out_channels == 0 || n_in_channels == 0) {
            throw std::runtime_error("n_in_channels " + std::to_string(n_in_channels) + " and n_out_channels " + std::to_string(n_out_channels) + " must be greater than 0");
        }
        _n_proc_frames = n_proc_frames;
        _n_in_channels = n_in_channels;
        _n_out_channels = n_out_channels;
        _n_proc_channels = n_in_channels;
        _n_proc_samples = _n_proc_frames * _n_proc_channels;
        _input_specs = BufferRackSpecs(BufferSpecs(MemoryContext::DEVICE, _n_proc_frames, _n_in_channels, ChannelOrder::INTERLEAVED));
        _output_specs = BufferRackSpecs(BufferSpecs(MemoryContext::DEVICE, _n_proc_frames, _n_out_channels, ChannelOrder::INTERLEAVED));
    }
    virtual cudaGraph_t recordSetupGraph() {
        allocateBuffers();
        _destroy_setup_graph = true;
        return recordStreamToGraph(setup(getRecordingStream(), cudaStreamCaptureStatus::cudaStreamCaptureStatusActive), &_setup_graph);
    }

    virtual cudaGraph_t recordProcessGraph(const BufferRack* dest, const BufferRack* src) {
        _destroy_process_graph = true;
        return recordStreamToGraph(process(getRecordingStream(), dest, src, cudaStreamCaptureStatus::cudaStreamCaptureStatusActive), &_process_graph);
    }

    virtual cudaGraph_t recordPostProcessGraph() {
        _destroy_post_process_graph = true;
        if (_has_post_processing) {
            return recordStreamToGraph(postProcess(getRecordingStream(), cudaStreamCaptureStatus::cudaStreamCaptureStatusActive), &_post_process_graph);
        } else {
            return createGraphWithEmpyNode(&_post_process_graph);
        }
    }

    virtual cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        if (capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
            allocateBuffers();
        }
        return stream;
    }
    virtual cudaStream_t process(cudaStream_t stream, const BufferRack* dest, const BufferRack* src, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) = 0;
    virtual cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) { return stream; }
    virtual void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) { throw std::runtime_error("Not implemented"); };
    virtual void teardown() {
        deallocateBuffers();
        if (_destroy_setup_graph) gpuErrChk(cudaGraphDestroy(_setup_graph));
        if (_destroy_process_graph) gpuErrChk(cudaGraphDestroy(_process_graph));
        if (_destroy_post_process_graph) gpuErrChk(cudaGraphDestroy(_post_process_graph));
    }
    virtual GpuFx* clone() {
        throw std::runtime_error("Not implemented");
    }
};