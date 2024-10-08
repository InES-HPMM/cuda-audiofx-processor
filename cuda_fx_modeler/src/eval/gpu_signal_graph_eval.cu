#include <buffer.cuh>
#include <cuda_ext.cuh>
#include <kernels.cuh>

#include "gpu_signal_graph_eval.cuh"

class GpuSignalGraphEvaluator : public IGpuSignalGraphEvaluator {
   private:
    std::string _name;
    IGpuSignalGraph* _signal_graph = nullptr;
    std::vector<float*> _src_buffers_slices;
    std::vector<float*> _dest_buffers_slices;
    size_t _n_in_channels;

    cudaGraphExec_t _spin_graph_exec;
    cudaGraph_t _spin_graph;
    BufferRack _spinning_buffer;

    virtual void setupIOBuffers() {
    }

    virtual void setup() override {
        _signal_graph->setup(_n_proc_frames, _n_in_channels);
        for (size_t i = 0; i < _signal_graph->getInputChannelCount(); i++) {
            _src_buffers_slices.push_back(new float[_n_proc_frames]);
        }
        for (size_t i = 0; i < _signal_graph->getOutputChannelCount(); i++) {
            _dest_buffers_slices.push_back(new float[_n_proc_frames]);
        }

        gpuErrChk(cudaGraphCreate(&_spin_graph, 0));
        _spinning_buffer.set(BufferSpecs(MemoryContext::DEVICE, _n_proc_frames));
        float* buffer = _spinning_buffer.getDataMod();
        IKernelNode::create(1, _n_proc_frames, 0, (void*)spin_kernel, new void*[3]{&buffer, &buffer, &_n_proc_frames}, _spin_graph);
        gpuErrChk(cudaGraphInstantiate(&_spin_graph_exec, _spin_graph, NULL, NULL, 0));
    }

    virtual void process() override {
        _signal_graph->process(_dest_buffers_slices, _src_buffers_slices);
    }

    virtual void postProcess() override {}

    virtual void teardown() override {
        _signal_graph->teardown();
    }

    virtual void spin() override {
        gpuErrChk(cudaGraphLaunch(_spin_graph_exec, 0));
        gpuErrChk(cudaStreamSynchronize(0));
    }

    virtual IFPSignal* test(const IFPSignal* input) {
        size_t n_chunks = input->getFrameCount() / _n_proc_frames;
        size_t n_frames = n_chunks * _n_proc_frames;

        IFPSignal* output = IFPSignal::create(n_frames, _signal_graph->getOutputChannelCount(), input->getSampleRate(), input->getChannelOrder());

        for (size_t i = 0; i < n_chunks; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t c = 0; c < input->getChannelCount(); c++) {
                _src_buffers_slices[c] = ((float**)input->getDataPtrConst())[c] + i * _n_proc_frames;
            }
            for (size_t c = 0; c < output->getChannelCount(); c++) {
                _dest_buffers_slices[c] = ((float**)output->getDataPtrConst())[c] + i * _n_proc_frames;
            }
            process();
        }
        return output;
    }

    virtual std::string getName() override {
        return _name;
    }

   public:
    GpuSignalGraphEvaluator(IGpuSignalGraph* signal_graph, std::string name) : IGpuSignalGraphEvaluator(), _name(name), _signal_graph(signal_graph) {
    }

    ~GpuSignalGraphEvaluator() {
        delete _signal_graph;
    }

    bool testAccuracy(const IFPSignal* input, const IFPSignal* expected_output, const size_t n_proc_frames, const float max_rmsd, const int expected_min_rmsd_offset = 0, const bool write_output = false) override {
        _n_in_channels = input->getChannelCount();
        return Evaluator::_testAccuracy(input, expected_output, n_proc_frames, max_rmsd, expected_min_rmsd_offset, write_output);
    }

    PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, size_t n_proc_frames, size_t n_in_channels, bool simulate_buffer_intervals) override {
        _n_in_channels = n_in_channels;
        return Evaluator::_measurePerformance(n_warmup, n_measure, n_proc_frames, simulate_buffer_intervals);
    }
};

IGpuSignalGraphEvaluator* IGpuSignalGraphEvaluator::create(IGpuSignalGraph* signal_graph, std::string name) {
    return new GpuSignalGraphEvaluator(signal_graph, name);
}