#pragma once

#include <evaluator.hpp>
#include <gpu.cuh>
#include <operators.cuh>
#include <path.hpp>

class KernelEvaluator : public Evaluator {
   protected:
    std::string _name;
    cudaStream_t _stream = nullptr;
    int _n_blocks;
    int _n_threads;
    size_t _n_shared_mem;
    void* _kernel;
    void** _kernel_args;

    virtual void setup() override {
        gpuErrChk(cudaStreamCreate(&_stream));
    }

    virtual void process() override {
        gpuErrChk(cudaLaunchKernel(_kernel, _n_blocks, _n_threads, _kernel_args, _n_shared_mem, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    virtual void postProcess() override {
    }

    virtual void teardown() override {
        if (_stream) gpuErrChk(cudaStreamDestroy(_stream));
    }

    virtual std::string getName() override {
        return _name + "-stream-" + std::to_string(_n_blocks) + "x" + std::to_string(_n_threads);
    }

   public:
    KernelEvaluator(std::string name) : _name(name) {
    }

    bool testAccuracy() {
        throw std::runtime_error("Not implemented");
    }
    PerformanceMeasurement* measurePerformanceOptimalArgs(size_t n_warmup, size_t n_measure, void* kernel, void** kernel_args, const int n_blocks_max = 0, const int n_threads_max = 0, const size_t n_shared_mem = 0) {
        float occupancy = 0;
        getOptimalLaunchArgs(kernel, n_shared_mem, n_blocks_max, n_threads_max, _n_blocks, _n_threads, occupancy);
        return measurePerformance(n_warmup, n_measure, kernel, kernel_args, _n_blocks, _n_threads, n_shared_mem);
    }

    PerformanceMeasurement* measurePerformance(size_t n_warmup, size_t n_measure, void* kernel, void** kernel_args, int n_blocks, int n_threads, size_t n_shared_mem = 0) {
        _n_blocks = n_blocks;
        _n_threads = n_threads;
        _n_shared_mem = n_shared_mem;
        _kernel = kernel;
        _kernel_args = kernel_args;
        return Evaluator::_measurePerformance(n_warmup, n_measure, 0, false);
    }

    void gridSearchLaunchArgs(size_t n_warmup, size_t n_measure, void* kernel, void** kernel_args, size_t n_blocks_min, size_t n_blocks_max, size_t n_threads_min, size_t n_threads_max, size_t n_threads_step, size_t n_shared_mem = 0) {
        std::vector<PerformanceMeasurement*> measurements;
        for (size_t n_blocks = n_blocks_min; n_blocks <= n_blocks_max; n_blocks++) {
            for (size_t n_threads = n_threads_min; n_threads <= n_threads_max; n_threads += n_threads_step) {
                measurements.push_back(measurePerformance(n_warmup, n_measure, kernel, kernel_args, n_blocks, n_threads, n_shared_mem));
            }
        }
        PerformanceMeasurement::writeStatisticsToCsv(path::out(_name + "-grid-search-launch-args.csv"), measurements);
    }

    void gridSearchBufferSize(size_t n_warmup, size_t n_measure, void* kernel, void** kernel_args, void (*kernel_args_update_func)(void**, size_t, size_t), size_t n_frames_min, size_t n_frames_max, size_t n_frames_step, size_t n_channels, size_t n_blocks, size_t n_threads, size_t n_shared_mem = 0) {
        std::vector<size_t> frame_counts;
        for (size_t n_frames = n_frames_min; n_frames <= n_frames_max; n_frames += n_frames_step) {
            frame_counts.push_back(n_frames);
        }
        gridSearchBufferSize(n_warmup, n_measure, kernel, kernel_args, kernel_args_update_func, frame_counts, n_channels, n_blocks, n_threads, n_shared_mem);
    }

    void gridSearchBufferSize(size_t n_warmup, size_t n_measure, void* kernel, void** kernel_args, void (*kernel_args_update_func)(void**, size_t, size_t), std::vector<size_t> frame_counts, size_t n_channels, size_t n_blocks, size_t n_threads, size_t n_shared_mem = 0) {
        std::vector<PerformanceMeasurement*> measurements;
        for (size_t n_frames : frame_counts) {
            size_t n_samples = n_frames * n_channels;

            // spdlog::info("n_frames={}, n_channels={}, n_samples={}", n_frames, n_channels, n_samples);
            kernel_args_update_func(kernel_args, n_frames, n_channels);

            measurements.push_back(measurePerformance(n_warmup, n_measure, kernel, kernel_args, n_blocks, n_threads == 0 ? n_samples : n_threads, n_shared_mem));
        }
        std::string name = getName() + "-" + std::to_string(frame_counts.front()) + "f-to-" + std::to_string(frame_counts.back()) + "f-" + std::to_string(n_channels) + "c-grid-search-buffer-size";
        PerformanceMeasurement::writeDataToCsv(path::out(name + "-data.csv"), measurements);
        PerformanceMeasurement::writeStatisticsToCsv(path::out(name + "-stats.csv"), measurements);
    }

    static void getOptimalLaunchArgs(void* kernel, const size_t n_shared_mem, const int n_blocks_max, const int n_threads_max, int& n_blocks, int& n_threads, float& occupancy) {
        cudaOccupancyMaxPotentialBlockSize(&n_blocks, &n_threads, kernel, n_shared_mem, n_threads_max);
        if (n_blocks_max > 0) {
            n_blocks = std::min(n_blocks, n_blocks_max);
        }

        int maxActiveBlocksPerMultiprocessor;
        gpuErrChk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerMultiprocessor, kernel, n_threads, n_shared_mem));
        int device;
        cudaDeviceProp props;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        occupancy = (maxActiveBlocksPerMultiprocessor * n_threads / props.warpSize) /
                    (float)(props.maxThreadsPerMultiProcessor /
                            props.warpSize);
    }

    static void logOptimalLaunchArgs(void* kernel, const size_t n_shared_mem = 0, const int n_blocks_max = 0, const int n_threads_max = 0, const spdlog::level::level_enum level = spdlog::level::info) {
        int n_blocks;
        int n_threads;
        float occupancy;
        getOptimalLaunchArgs(kernel, n_shared_mem, n_blocks_max, n_threads_max, n_blocks, n_threads, occupancy);
        spdlog::log(level, "Optimal launch args: blocks={}, threads={}, occupancy={}", n_blocks, n_threads, occupancy);
    };
};

class KernelGraphEvaluator : public KernelEvaluator {
   private:
    cudaGraph_t _process_graph;
    cudaGraphExec_t _process_graph_exec;

    void setup() override {
        gpuErrChk(cudaGraphCreate(&_process_graph, 0));
        IKernelNode::create(_n_blocks, _n_threads, 0, _kernel, _kernel_args, _process_graph);
        gpuErrChk(cudaGraphInstantiate(&_process_graph_exec, _process_graph, NULL, NULL, 0));

        gpuErrChk(cudaStreamCreate(&_stream));
    }

    void process() override {
        gpuErrChk(cudaGraphLaunch(_process_graph_exec, _stream));
        gpuErrChk(cudaStreamSynchronize(_stream));
    }

    void teardown() override {
        if (_process_graph_exec) gpuErrChk(cudaGraphExecDestroy(_process_graph_exec));
        KernelEvaluator::teardown();
    }

   public:
    KernelGraphEvaluator(std::string name) : KernelEvaluator(name) {
    }

    virtual std::string getName() override {
        return _name + "-graph-" + std::to_string(_n_blocks) + "x" + std::to_string(_n_threads);
    }
};
