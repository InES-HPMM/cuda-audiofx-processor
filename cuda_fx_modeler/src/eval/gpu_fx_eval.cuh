
#include <evaluator.hpp>
#include <gpu_fx.cuh>

class IGpuFxEvaluator : public Evaluator {
   public:
    static IGpuFxEvaluator* createStreamEval(IGpuFx* gpu_fx);
    static IGpuFxEvaluator* createGraphEval(IGpuFx* gpu_fx);

    virtual ~IGpuFxEvaluator() {}

    virtual bool testAccuracy(const IFPSignal* input, const IFPSignal* expected_output, const size_t n_proc_frames, const bool process_in_place, const float max_rmsd, const int expected_min_rmsd_offset = 0, const bool write_output = false) = 0;

    virtual PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, size_t n_proc_frames, bool simulate_buffer_intervals, bool process_in_place) = 0;
};
