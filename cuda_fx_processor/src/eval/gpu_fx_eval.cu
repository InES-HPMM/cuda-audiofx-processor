
#include <cuda_runtime.h>

#include <buffer.cuh>
#include <cuda_ext.cuh>
#include <gpu.cuh>
#include <gpu_fx.cuh>
#include <kernels.cuh>
#include <operators.cuh>
#include <path.hpp>
#include <rmsd.cuh>

#include "gpu_fx_eval.cuh"

class GpuFxStreamEvaluator : public IGpuFxEvaluator {
   protected:
    bool _process_in_place;
    IGpuFx* _gpuFx = nullptr;
    cudaStream_t _stream;
    BufferRack _input_buffer;
    BufferRack _output_buffer;
    BufferRack _actual_output;

    size_t _n_chunks;
    size_t _n_frames;

    virtual void setupIOBuffers() {
        _actual_output.set(_gpuFx->getOutputSpecs().setFrameCount(_n_frames).setContext(MemoryContext::HOST));
        _actual_output.clearBuffers();

        _input_buffer.set(_gpuFx->getInputSpecs().setFrameCount(_n_frames));
        if (_process_in_place) {
            _output_buffer = _input_buffer;
        } else {
            _output_buffer.set(_gpuFx->getOutputSpecs().setFrameCount(_n_frames));
        }
    }

    virtual void setup() override {
        _gpuFx->configure(_n_proc_frames);
        setupIOBuffers();
        gpuErrChk(cudaStreamCreate(&_stream));
        _gpuFx->setup(_stream);
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    virtual void process() override {
        _gpuFx->process(_stream, &_output_buffer, &_input_buffer);
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    virtual void postProcess() override {
        _gpuFx->postProcess(_stream);
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    virtual void spin() override {
        spin_kernel<<<1, _n_proc_frames, 0, _stream>>>(_output_buffer.getDataMod(), _input_buffer.getDataMod(), _n_proc_frames);
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    virtual void teardown() override {
        _gpuFx->teardown();
        if (_stream) gpuErrChk(cudaStreamDestroy(_stream));
        _input_buffer.deallocateBuffers();
        // _output_buffer.deallocateBuffers(); // TODO: fix free segfault
        _actual_output.deallocateBuffers();
    }

    virtual IFPSignal* test(const IFPSignal* input) {
        size_t _n_proc_samples_in = _n_proc_frames * _gpuFx->getInChannelCount();
        size_t _n_proc_samples_out = _n_proc_frames * _gpuFx->getOutChannelCount();

        for (size_t i = 0; i < _n_chunks; i++) {
            gpuErrChk(cudaMemcpyAsync(_input_buffer.getDataMod(), (float*)input->getDataPtrConst() + i * _n_proc_samples_in, sizeof(float) * _n_proc_samples_in, cudaMemcpyHostToDevice, _stream));
            process();
            postProcess();
            gpuErrChk(cudaMemcpyAsync(_actual_output.getDataMod() + i * _n_proc_samples_out, _output_buffer.getDataMod(), sizeof(float) * _n_proc_samples_out, cudaMemcpyDeviceToHost, _stream));
        }
        gpuErrChk(cudaStreamSynchronize(_stream));
        return IFPSignal::fromBuffer(_actual_output.getDataMod(), _actual_output.getFrameCount(), _actual_output.getChannelCount(), SampleRate::SR_48000, ChannelOrder::INTERLEAVED);
    }

    virtual std::string getName() override {
        return _gpuFx->getName() + "-stream-ip" + std::to_string(_process_in_place);
    }

   public:
    GpuFxStreamEvaluator(IGpuFx* gpuFx) : IGpuFxEvaluator() {
        _gpuFx = gpuFx;
    }

    ~GpuFxStreamEvaluator() {
        delete _gpuFx;
    }

    bool testAccuracy(const IFPSignal* input, const IFPSignal* expected_output, const size_t n_proc_frames, const bool process_in_place, const float max_rmsd, const int expected_min_rmsd_offset = 0, const bool write_output = false) override {
        _n_proc_frames = n_proc_frames;
        _process_in_place = process_in_place;
        _n_chunks = input->getFrameCount() / _n_proc_frames;
        _n_frames = _n_chunks * _n_proc_frames;

        return Evaluator::_testAccuracy(input, expected_output, n_proc_frames, max_rmsd, expected_min_rmsd_offset, write_output);
    }

    PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, size_t n_proc_frames, bool simulate_buffer_intervals, bool process_in_place) override {
        _process_in_place = process_in_place;
        _n_frames = n_proc_frames;
        return Evaluator::_measurePerformance(n_warmup, n_measure, n_proc_frames, simulate_buffer_intervals);
    }
};

IGpuFxEvaluator* IGpuFxEvaluator::createStreamEval(IGpuFx* gpuFx) {
    return new GpuFxStreamEvaluator(gpuFx);
}

class GpuFxGraphEvaluator : public GpuFxStreamEvaluator {
   private:
    cudaGraphExec_t _setup_graph_exec;
    cudaGraphExec_t _process_graph_exec;
    cudaGraphExec_t _post_process_graph_exec;
    cudaGraphExec_t _spin_graph_exec;
    cudaGraph_t _spin_graph;

    void setup() override {
        _gpuFx->configure(_n_proc_frames);
        gpuErrChk(cudaGraphInstantiate(&_setup_graph_exec, _gpuFx->recordSetupGraph(), NULL, NULL, 0));
        gpuErrChk(cudaStreamCreate(&_stream));
        gpuErrChk(cudaGraphLaunch(_setup_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
        gpuErrChk(cudaStreamDestroy(_stream));

        GpuFxStreamEvaluator::setupIOBuffers();

        gpuErrChk(cudaGraphCreate(&_spin_graph, 0));
        float* dst = _output_buffer.getDataMod();
        float* src = _input_buffer.getDataMod();
        IKernelNode::create(1, _n_proc_frames, 0, (void*)spin_kernel, new void*[3]{&dst, &src, &_n_proc_frames}, _spin_graph);
        gpuErrChk(cudaGraphInstantiate(&_spin_graph_exec, _spin_graph, NULL, NULL, 0));

        gpuErrChk(cudaGraphInstantiate(&_process_graph_exec, _gpuFx->recordProcessGraph(&_output_buffer, &_input_buffer), NULL, NULL, 0));
        gpuErrChk(cudaGraphInstantiate(&_post_process_graph_exec, _gpuFx->recordPostProcessGraph(), NULL, NULL, 0));
        gpuErrChk(cudaStreamCreate(&_stream));
    }

    void process() override {
        gpuErrChk(cudaGraphLaunch(_process_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    void postProcess() override {
        gpuErrChk(cudaGraphLaunch(_post_process_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    void spin() {
        gpuErrChk(cudaGraphLaunch(_spin_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    void teardown() override {
        if (_setup_graph_exec) gpuErrChk(cudaGraphExecDestroy(_setup_graph_exec));
        if (_process_graph_exec) gpuErrChk(cudaGraphExecDestroy(_process_graph_exec));
        if (_post_process_graph_exec) gpuErrChk(cudaGraphExecDestroy(_post_process_graph_exec));
        GpuFxStreamEvaluator::teardown();
    }

   public:
    GpuFxGraphEvaluator(IGpuFx* gpuFx) : GpuFxStreamEvaluator(gpuFx) {
    }

    virtual std::string getName() override {
        return _gpuFx->getName() + "-graph";
    }
};

IGpuFxEvaluator* IGpuFxEvaluator::createGraphEval(IGpuFx* gpuFx) {
    return new GpuFxGraphEvaluator(gpuFx);
}