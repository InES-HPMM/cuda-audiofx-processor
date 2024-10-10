

#include <cmath>  // pow, sin
#include <cuda_ext.cuh>
#include <operators.cuh>

#include "gpu_fx.cu"

__global__ static void ff_mix_interleaved(float* dst, const float* src, const size_t n_in_lanes, const size_t n_out_channels, const size_t n_frames) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_frames; s += stride.x) {
        for (size_t channel = 0; channel < n_out_channels; channel++) {
            auto channel_sum = 0.0f;
            for (size_t lane = 0; lane < n_in_lanes; lane++) {
                channel_sum += src[s * n_in_lanes * n_out_channels + lane * n_out_channels + channel];
            }
            dst[s * n_out_channels + channel] = channel_sum / n_in_lanes;
        }
    }
}

__global__ static void ff_mix_segment(float* dst, const float* const* src, const size_t n_in_lanes, const size_t n_out_channels, const size_t n_frames) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_frames; s += stride.x) {
        for (size_t channel = 0; channel < n_out_channels; channel++) {
            auto channel_sum = 0.0f;
            for (size_t lane = 0; lane < n_in_lanes; lane++) {
                channel_sum += src[lane][s * n_out_channels + channel];
            }
            dst[s * n_out_channels + channel] = channel_sum / n_in_lanes;
        }
    }
}

class FxMixTool : public GpuFx {
   protected:
    IGraphNode* _node = nullptr;
    size_t _n_lanes;

    size_t _n_in_channels_default;
    size_t _n_out_channels_default;

    void allocateBuffers() override {
    }

    void deallocateBuffers() override {
        if (_node) delete _node;
    }

   public:
    FxMixTool(std::string name, size_t n_in_channels, size_t n_out_channels) : GpuFx(name, false), _n_in_channels_default(n_in_channels), _n_out_channels_default(n_out_channels) {
    }

    ~FxMixTool() {
    }

    void configure(size_t process_buffer_size, size_t n_in_channels, size_t n_out_channels) override {
        if (n_in_channels == 0)
            n_in_channels = _n_in_channels_default;
        else
            n_in_channels = n_in_channels;
        if (n_out_channels == 0)
            n_out_channels = _n_out_channels_default;
        else
            n_out_channels = n_out_channels;

        if (n_in_channels <= n_out_channels) {
            spdlog::error("{}: n_in_channels {} must be larger than _n_out_channels {}", _name, n_in_channels, n_out_channels);
            throw std::runtime_error(_name + ": received invalid channel configuration");
        } else if (n_in_channels % n_out_channels != 0) {
            spdlog::error("{}: n_in_channels {} must be a multiple of n_out_channels {}", _name, n_in_channels, n_out_channels);
            throw std::runtime_error(_name + ": received invalid channel configuration");
        }
        GpuFx::configure(process_buffer_size, n_in_channels, n_out_channels);
        _n_lanes = _n_in_channels / _n_out_channels;
    }
};

class FxMixSegment : public FxMixTool {
   private:
    float* _src_data;
    float** _src_data_host;

    void allocateBuffers() override {
        _src_data_host = new float*[_n_lanes];
        for (size_t i = 0; i < _n_lanes; i++) {
            gpuErrChk(cudaMalloc(_src_data_host + i, _n_proc_frames * _n_out_channels * sizeof(float)));
        }
        gpuErrChk(cudaMalloc(&_src_data, _n_lanes * sizeof(float*)));
        gpuErrChk(cudaMemcpy(_src_data, _src_data_host, _n_lanes * sizeof(float*), cudaMemcpyHostToDevice));
    }

    void deallocateBuffers() override {
        gpuErrChk(cudaFree(_src_data));
    }

   public:
    FxMixSegment(size_t n_in_channels, size_t n_out_channels) : FxMixTool("FxMixSegment", n_in_channels, n_out_channels) {}

    void configure(size_t process_buffer_size, size_t n_in_channels, size_t n_out_channels) override {
        FxMixTool::configure(process_buffer_size, n_in_channels, n_out_channels);
        _input_specs = BufferRackSpecs(BufferSpecs(MemoryContext::DEVICE, _n_proc_frames, _n_out_channels), _n_in_channels);
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        throw std::runtime_error("FxMixSegment: updateBufferPtrs not implemented");
        // static_cast<IKernelNode*>(_node)->updateKernelParamAt(1, (void*)src->getDataListConst());
        // static_cast<IKernelNode*>(_node)->updateKernelParamAt(0, dst->getDataMod(), procGraphExec);
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        auto src_buffers = src->getDataListMod();
        auto dst_buffer = dst->getDataMod();

        // TODO: Pointers to segmented buffers need to be on the device as well for a kernel to work with. Remove buffer params from process streams and use the updateBufferPtrs method at the start and whenever subsequently necessary.
        gpuErrChk(cudaMemcpyAsync(_src_data, src_buffers, sizeof(float*) * _n_lanes, cudaMemcpyHostToDevice, stream));
        IKernelNode::launchOrRecord(dim3(1), dim3(_n_proc_frames * _n_out_channels), 0, (void*)ff_mix_segment, new void*[5]{&dst_buffer, &_src_data, &_n_lanes, &_n_out_channels, &_n_proc_frames}, stream, static_cast<IKernelNode*>(_node), capture_status);
        return stream;
    }
};

IGpuFx* IGpuFx::createMixSegment(size_t n_in_channels, size_t n_out_channels) {
    return new FxMixSegment(n_in_channels, n_out_channels);
}

class FxMixInterleaved : public FxMixTool {
   public:
    FxMixInterleaved(size_t n_in_channels, size_t n_out_channels) : FxMixTool("FxMixInterleaved", n_in_channels, n_out_channels) {}

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        static_cast<IKernelNode*>(_node)->updateKernelParamAt(1, (void*)src->getDataListConst());
        static_cast<IKernelNode*>(_node)->updateKernelParamAt(0, dst->getDataMod(), procGraphExec);
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        auto src_buffers = src->getDataMod();
        auto dst_buffer = dst->getDataMod();

        IKernelNode::launchOrRecord(dim3(1), dim3(2 * _n_proc_samples), 0, (void*)ff_mix_interleaved, new void*[5]{&dst_buffer, &src_buffers, &_n_lanes, &_n_out_channels, &_n_proc_frames}, stream, static_cast<IKernelNode*>(_node), capture_status);

        return stream;
    }
};

IGpuFx* IGpuFx::createMixInterleaved(size_t n_in_channels, size_t n_out_channels) {
    return new FxMixInterleaved(n_in_channels, n_out_channels);
}
