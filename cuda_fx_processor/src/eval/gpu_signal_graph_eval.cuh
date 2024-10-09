#include <evaluator.hpp>
#include <gpu_fx.cuh>
#include <gpu_signal_graph.cuh>

class IGpuSignalGraphEvaluator : public Evaluator {
   public:
    static IGpuSignalGraphEvaluator* create(IGpuSignalGraph* signal_graph, std::string name);

    ~IGpuSignalGraphEvaluator() {}

    virtual bool testAccuracy(const IFPSignal* input, const IFPSignal* expected_output, const size_t n_proc_frames, const float max_rmsd, const int expected_min_rmsd_offset = 0, const bool write_output = false) = 0;

    virtual PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, size_t n_proc_frames, size_t n_in_channels, bool simulate_buffer_intervals) = 0;
};