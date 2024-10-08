#include <cuda_ext.cuh>
#include <operators.cuh>

#include "gpu_fx.cu"

enum class NoiseGateState {
    CLOSED = 0,
    OPEN = 1,
    HOLD = 2,
    ATTACK = 3,
    RELEASE = 4
};

__global__ static void ff_gate(float* dst, const float* src, NoiseGateState* state, size_t* hold_time_counter, float* gain, float* threshold, float* attack_coeff, float* release_coeff, size_t* hold_time, size_t n_samples) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_samples; s += stride.x) {
        auto src_abs = abs(src[s]);
        switch (*state) {
            case NoiseGateState::CLOSED:
                *gain = 0;
                if (src_abs > *threshold) {
                    *state = NoiseGateState::RELEASE;
                }
                break;
            case NoiseGateState::RELEASE:
                *gain = *gain * *release_coeff + (1 - *release_coeff);
                if (src_abs < *threshold) {
                    *state = NoiseGateState::HOLD;
                } else if (*gain > 0.99999) {
                    *state = NoiseGateState::OPEN;
                }
                break;
            case NoiseGateState::OPEN:
                *gain = 1;
                if (src_abs < *threshold) {
                    *state = NoiseGateState::HOLD;
                }
                break;
            case NoiseGateState::HOLD:
                // *gain = 1;
                (*hold_time_counter)++;
                if (*hold_time_counter >= *hold_time) {
                    *state = NoiseGateState::ATTACK;
                    *hold_time_counter = 0;
                }
                if (src_abs > *threshold) {
                    *state = NoiseGateState::RELEASE;
                }
                break;
            case NoiseGateState::ATTACK:
                *gain = *gain * *attack_coeff;

                if (src_abs > *threshold) {
                    *state = NoiseGateState::RELEASE;
                } else if (*gain <= 0.00001) {
                    *state = NoiseGateState::CLOSED;
                }
                break;
        }
        dst[s] = src[s] * *gain;
    }
}

class FxGate : public GpuFx {
   protected:
    IGraphNode* _node = nullptr;

    size_t _n_channels_default;

    float _threshold;
    float _attack_coeff;
    float _release_coeff;
    size_t _hold_time;

    NoiseGateState* _state_ptr = nullptr;
    float* _gain_ptr = nullptr;
    float* _threshold_ptr = nullptr;
    float* _attack_coeff_ptr = nullptr;
    float* _release_coeff_ptr = nullptr;
    size_t* _hold_time_counter_ptr = nullptr;
    size_t* _hold_time_ptr = nullptr;

    void allocateBuffers() override {
        cudaMallocManaged(&_state_ptr, sizeof(NoiseGateState));
        cudaMallocManaged(&_gain_ptr, sizeof(float));
        cudaMallocManaged(&_threshold_ptr, sizeof(float));
        cudaMallocManaged(&_attack_coeff_ptr, sizeof(float));
        cudaMallocManaged(&_release_coeff_ptr, sizeof(float));
        cudaMallocManaged(&_hold_time_counter_ptr, sizeof(size_t));
        cudaMallocManaged(&_hold_time_ptr, sizeof(size_t));
    }

    void deallocateBuffers() override {
        if (_node) delete _node;
        cudaFree(_state_ptr);
        cudaFree(_hold_time_counter_ptr);
        cudaFree(_gain_ptr);
        cudaFree(_threshold_ptr);
        cudaFree(_attack_coeff_ptr);
        cudaFree(_release_coeff_ptr);
        cudaFree(_hold_time_ptr);
    }

   public:
    FxGate(float threshold, float attack_time_ms, float release_time_ms, float hold_time_ms, SampleRate sample_rate) : GpuFx("FxGate", false) {
        _threshold = threshold;
        _attack_coeff = exp(-2197.22457734f / (attack_time_ms * as_int(sample_rate)));
        _release_coeff = exp(-2197.22457734f / (release_time_ms * as_int(sample_rate)));
        _hold_time = hold_time_ms * 0.001 * as_int(sample_rate);
    }
    FxGate(float threshold, float attack_coeff, float release_coeff, float hold_time) : GpuFx("FxGate", false), _threshold(threshold), _attack_coeff(attack_coeff), _release_coeff(release_coeff), _hold_time(hold_time) {}
    ~FxGate() {
    }
    GpuFx* clone() override {
        return new FxGate(_threshold, _attack_coeff, _release_coeff, _hold_time);
    }

    void configure(size_t process_buffer_size, size_t n_in_channels, size_t n_out_channels) override {
        if (n_in_channels != 0 && n_in_channels > 1 || n_out_channels  != 0 && n_out_channels > 1) {
            spdlog::warn("{} is a single channel effect, set n_in_channels {} and n_in_channels {} are ignored", _name, n_in_channels, n_out_channels);
        }
        GpuFx::configure(process_buffer_size, 1, 1);
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        gpuErrChk(cudaMemsetAsync(_state_ptr, 0, sizeof(NoiseGateState), stream));
        gpuErrChk(cudaMemsetAsync(_hold_time_counter_ptr, 0, sizeof(size_t), stream));
        gpuErrChk(cudaMemsetAsync(_gain_ptr, 0, sizeof(float), stream));
        gpuErrChk(cudaMemcpyAsync(_threshold_ptr, &_threshold, sizeof(float), cudaMemcpyHostToDevice, stream));
        gpuErrChk(cudaMemcpyAsync(_attack_coeff_ptr, &_attack_coeff, sizeof(float), cudaMemcpyHostToDevice, stream));
        gpuErrChk(cudaMemcpyAsync(_release_coeff_ptr, &_release_coeff, sizeof(float), cudaMemcpyHostToDevice, stream));
        gpuErrChk(cudaMemcpyAsync(_hold_time_ptr, &_hold_time, sizeof(size_t), cudaMemcpyHostToDevice, stream));
        return stream;
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        ff_gate<<<1, 1, 0, stream>>>(dst->getDataMod(), src->getDataMod(), _state_ptr, _hold_time_counter_ptr, _gain_ptr, _threshold_ptr, _attack_coeff_ptr, _release_coeff_ptr, _hold_time_ptr, _n_proc_frames);
        return stream;
    }
};

IGpuFx* IGpuFx::createGate(float threshold, float attack_time_ms, float release_time_ms, float hold_time_ms, SampleRate sample_rate) {
    return new FxGate(threshold, attack_time_ms, release_time_ms, hold_time_ms, sample_rate);
}