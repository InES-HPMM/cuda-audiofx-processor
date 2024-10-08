#include <convert.hpp>
#include <iomanip>

#include "buffer.cuh"
#include "gpu.cuh"
#include "spdlog/spdlog.h"

class BufferImpl : public Buffer {
   private:
    BufferSpecs _specs;
    float* _data;
    bool _owns_memory;

   public:
    BufferImpl(float* data, BufferSpecs specs) : _data(data), _specs(specs), _owns_memory(false) {}
    BufferImpl(BufferSpecs specs) : _specs(specs), _owns_memory(true) {
        switch (_specs.context) {
            case MemoryContext::HOST:
                _data = new float[_specs.n_samples];
                break;
            case MemoryContext::DEVICE:
                gpuErrChk(cudaMalloc(&_data, _specs.n_samples * sizeof(float)));
                break;
            default:
                throw std::invalid_argument("Invalid MemoryContext");
        }
    }

    ~BufferImpl() {
        if (!_owns_memory) return;
        switch (_specs.context) {
            case MemoryContext::HOST:
                delete[] _data;
                break;
            case MemoryContext::DEVICE:
                gpuErrChk(cudaFree(_data));
                break;
            default:
                spdlog::warn("Invalid MemoryContext");
        }
        _owns_memory = false;
    }

    BufferSpecs getSpecs() const override {
        return _specs;
    }
    std::string toString() const {
        return "Buffer{" + _specs.toString() + ", address=" + int_to_hex((size_t)(_data)) + "}";
    }

    size_t getChannelCount() const override {
        return _specs.n_channels;
    }
    size_t getFrameCount() const override {
        return _specs.n_frames;
    }
    size_t getSampleCount() const override {
        return _specs.n_samples;
    }
    ChannelOrder getChannelOrder() const override {
        return _specs.channel_order;
    }
    MemoryContext getMemoryContext() const override {
        return _specs.context;
    }
    const float* getDataConst() const override {
        return _data;
    }
    float* getDataMod() override {
        return _data;
    }
    void clear(cudaStream_t stream) override {
        switch (_specs.context) {
            case MemoryContext::HOST:
                memset(_data, 0, _specs.n_samples * sizeof(float));
                break;
            case MemoryContext::DEVICE:
                if (stream) {
                    gpuErrChk(cudaMemsetAsync(_data, 0, _specs.n_samples * sizeof(float), stream));
                } else {
                    gpuErrChk(cudaMemset(_data, 0, _specs.n_samples * sizeof(float)));
                }
                break;
            default:
                throw std::invalid_argument("Invalid MemoryContext");
        }
    }

    // void setData(float* data) override{
    //     delete[] _data;
    //     _data = data;
    // }
    // void setData(float* data, size_t n_channels, size_t n_frames, ChannelOrder channel_order, MemoryContext context) override{
    //     _n_channels = n_channels;
    //     _n_frames = n_frames;
    //     _channel_order = channel_order;
    //     _context = context;
    //     setData(data);
    // }
};

Buffer* Buffer::create(float* data, BufferSpecs specs) {
    return new BufferImpl(data, specs);
}

Buffer* Buffer::create(BufferSpecs specs) {
    return new BufferImpl(specs);
}