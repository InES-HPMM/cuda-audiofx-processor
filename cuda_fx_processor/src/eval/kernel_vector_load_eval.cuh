#include <evaluator.hpp>
#include <gpu.cuh>
#include <kernel_eval.cuh>
#include <operators.cuh>
#include <path.hpp>

__global__ void dynamic_kernel(float* dest, float* src1, int n, int c) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;
    for (auto s = offset.x; s < n * c; s += stride.x + c) {
        for (auto i = 0; i < c; i++) {
            dest[s + i] = src1[s + i];
        }
    }
}
__global__ void fixed_kernel(float* dest, float* src1, int n, int c) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;
    for (auto s = offset.x; s < n * c; s += stride.x) {
        dest[s] = src1[s];
    }
}
__global__ void fixed2_kernel(float2* dest, float2* src1, int n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;
    for (auto s = offset.x; s < n; s += stride.x) {
        dest[s] = src1[s];
    }
}
__global__ void fixed3_kernel(float3* dest, float3* src1, int n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;
    for (auto s = offset.x; s < n; s += stride.x) {
        dest[s] = src1[s];
    }
}

__global__ void fixed4_kernel(float4* dest, float4* src1, int n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;
    for (auto s = offset.x; s < n; s += stride.x) {
        dest[s] = src1[s];
    }
}

class VectorLoadEvaluator : public KernelEvaluator {
   protected:
    float* _dest;
    float* _src1;
    float* _dest_offset_ptr;
    float* _src1_offset_ptr;
    int _n_proc_frames;
    int _n_channels;
    int _n_executions;
    int _exec_index;
    int _stride = 100;

    void setKernelArgs() {
        _dest_offset_ptr = _dest + _n_proc_frames * _n_channels * ((_exec_index % _stride) * (_n_executions / _stride) + (_exec_index / _stride));
        _src1_offset_ptr = _src1 + _n_proc_frames * _n_channels * ((_exec_index % _stride) * (_n_executions / _stride) + (_exec_index / _stride));
        _kernel_args = new void*[4];
        _kernel_args[0] = &_dest_offset_ptr;
        _kernel_args[1] = &_src1_offset_ptr;
        _kernel_args[2] = &_n_proc_frames;
        _kernel_args[3] = &_n_channels;
    }

    virtual void setup() override {
        KernelEvaluator::setup();
        gpuErrChk(cudaMalloc(&_dest, _n_proc_frames * _n_channels * (_n_executions + 1) * sizeof(float)));
        gpuErrChk(cudaMalloc(&_src1, _n_proc_frames * _n_channels * (_n_executions + 1) * sizeof(float)));
    }

    virtual void process() override {
        setKernelArgs();
        KernelEvaluator::process();
        _exec_index++;
    }

    virtual void teardown() override {
        if (_dest) gpuErrChk(cudaFree(_dest));
        if (_src1) gpuErrChk(cudaFree(_src1));
    }

    virtual std::string getName() override {
        return _name + "-" + std::to_string(_n_proc_frames) + "frames";
    }

   public:
    VectorLoadEvaluator(std::string name, size_t n_channels) : KernelEvaluator(name), _n_channels(n_channels) {
    }

    PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, void* kernel, const int n_blocks_max = 0, const int n_threads_max = 0, const size_t n_shared_mem = 0) {
        _n_executions = n_warmup + n_measure;
        _exec_index = 0;
        return KernelEvaluator::measurePerformanceOptimalArgs(n_warmup, n_measure, kernel, _kernel_args, n_blocks_max, n_threads_max);
    }

    void gridSearch(size_t n_warmup, size_t n_measure, void* kernel, size_t n_frames_min, size_t n_frames_max, size_t n_frames_step, const int n_blocks_max = 0, const int n_threads_max = 0) {
        std::vector<PerformanceMeasurement*> measurements;
        for (size_t n_frames = n_frames_min; n_frames <= n_frames_max; n_frames += n_frames_step) {
            _n_proc_frames = n_frames;

            measurements.push_back(measurePerformance(n_warmup, n_measure, kernel, n_blocks_max, n_threads_max));
        }
        PerformanceMeasurement::writeStatisticsToCsv(path::out(_name + "-" + std::to_string(_n_channels) + "ch-" + std::to_string(_n_blocks) + "b-" + std::to_string(_n_threads) + "n-grid-search.csv"), measurements);
    }
};

void testVectorLoadPerformance() {
    size_t n_frames_min = 16;
    size_t n_frames_max = 256;
    size_t n_frames_step = 16;
    size_t n_warmup = 1000;
    size_t n_measure = 10000;
    size_t n_blocks_max = 1;
    size_t n_threads_max = 256;
    size_t n_channels = 4;
    // VectorLoadEvaluator("vector-load-fix", 4).gridSearch(n_warmup, n_measure, (void*)fixed_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, 32);

    VectorLoadEvaluator("vector-load-dyn", n_channels).gridSearch(n_warmup, n_measure, (void*)dynamic_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max);
    VectorLoadEvaluator("vector-load-elem", n_channels).gridSearch(n_warmup, n_measure, (void*)fixed_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max);
    VectorLoadEvaluator("vector-load-elem", n_channels).gridSearch(n_warmup, n_measure, (void*)fixed_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max * n_channels);
    switch (n_channels) {
        case 2:
            VectorLoadEvaluator("vector-load-vec2", n_channels).gridSearch(n_warmup, n_measure, (void*)fixed2_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max);
            break;
        case 3:
            VectorLoadEvaluator("vector-load-vec3", n_channels).gridSearch(n_warmup, n_measure, (void*)fixed3_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max);
            break;
        case 4:
            VectorLoadEvaluator("vector-load-vec4", n_channels).gridSearch(n_warmup, n_measure, (void*)fixed4_kernel, n_frames_min, n_frames_max, n_frames_step, n_blocks_max, n_threads_max);
            break;
    }
}