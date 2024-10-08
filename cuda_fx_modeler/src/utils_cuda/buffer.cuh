#pragma once
#include <convert.hpp>
#include <enums.hpp>
#include <numeric>
#include <vector>

#include "spdlog/spdlog.h"

class BufferSpecs {
   public:
    size_t n_frames;
    size_t n_channels;
    size_t n_samples;
    ChannelOrder channel_order;
    MemoryContext context;

    BufferSpecs() : n_frames(0), n_channels(0), n_samples(0), channel_order(ChannelOrder::INTERLEAVED), context(MemoryContext::HOST) {}

    BufferSpecs(MemoryContext context, size_t n_frames, size_t n_channels = 1, ChannelOrder channel_order = ChannelOrder::INTERLEAVED) : n_frames(n_frames), n_channels(n_channels), n_samples(n_frames * n_channels), channel_order(channel_order), context(context) {}

    std::string toString() const {
        std::string context_str;
        switch (context) {
            case MemoryContext::HOST:
                context_str = "HOST";
                break;
            case MemoryContext::DEVICE:
                context_str = "DEVICE";
                break;
            default:
                context_str = "INVALID";
        }
        std::string channel_order_str;
        switch (channel_order) {
            case ChannelOrder::INTERLEAVED:
                channel_order_str = "INTERLEAVED";
                break;
            case ChannelOrder::PLANAR:
                channel_order_str = "PLANAR";
                break;
            default:
                channel_order_str = "INVALID";
        }

        return "BufferSpecs{context=" + context_str + ", n_frames=" + std::to_string(n_frames) + ", n_channels=" + std::to_string(n_channels) + ", channel_order=" + channel_order_str + "}";
    }
};

class Buffer {
   public:
    static Buffer* create(float* data, BufferSpecs specs);
    static Buffer* create(BufferSpecs specs);

    virtual ~Buffer() {}

    virtual BufferSpecs getSpecs() const = 0;
    virtual std::string toString() const = 0;
    virtual size_t getChannelCount() const = 0;
    virtual size_t getFrameCount() const = 0;
    virtual size_t getSampleCount() const = 0;
    virtual ChannelOrder getChannelOrder() const = 0;
    virtual MemoryContext getMemoryContext() const = 0;
    virtual const float* getDataConst() const = 0;
    virtual float* getDataMod() = 0;
    virtual void clear(cudaStream_t stream = nullptr) = 0;
    // virtual void setData(float* data) = 0;
    // virtual void setData(float* data, size_t n_channels, size_t n_frames, ChannelOrder channel_order, MemoryContext context) = 0;
};

class BufferRackSpecs {
   public:
    std::vector<BufferSpecs> v;
    BufferRackSpecs() : v() {}
    BufferRackSpecs(BufferSpecs specs, size_t n = 1) : v(n, specs) {}
    BufferRackSpecs(const std::vector<BufferSpecs>& specs) : v(specs) {}
    BufferRackSpecs(const std::vector<Buffer*>& buffers) : v(buffers.size()) {
        for (size_t i = 0; i < buffers.size(); i++) {
            v[i] = buffers[i]->getSpecs();
        }
    }

    size_t getChannelCount() const {
        return std::accumulate(v.begin(), v.end(), 0, [](size_t sum, BufferSpecs specs) { return sum + specs.n_channels; });
    }
    size_t getFrameCount() const {
        return v.at(0).n_frames;
    }
    size_t getSampleCount() const {
        return getChannelCount() * getFrameCount();
    }

    void append(const std::vector<BufferSpecs>& specs) {
        v.resize(v.size() + specs.size());
        v.insert(v.end(), specs.begin(), specs.end());
    }

    BufferRackSpecs setContext(MemoryContext context) {
        for (auto& specs : v) {
            specs.context = context;
        }
        return *this;
    }
    BufferRackSpecs setChannelOrder(ChannelOrder channel_order) {
        for (auto& specs : v) {
            specs.channel_order = channel_order;
        }
        return *this;
    }
    BufferRackSpecs setChannelCount(size_t n_channels) {
        for (auto& specs : v) {
            specs.n_channels = n_channels;
            specs.n_samples = specs.n_frames * n_channels;
        }
        return *this;
    }
    BufferRackSpecs setFrameCount(size_t n_frames) {
        for (auto& specs : v) {
            specs.n_frames = n_frames;
            specs.n_samples = n_frames * specs.n_channels;
        }
        return *this;
    }
    std::string toString() const {
        std::string str = "BufferRackSpecs{";
        for (auto specs : v) {
            str += specs.toString() + ", ";
        }
        str.pop_back();
        str.pop_back();
        str += "}";
        return str;
    }
};

class BufferRack {
   private:
    std::vector<Buffer*> _buffers;
    std::vector<float*> _data;
    BufferRackSpecs _specs;

   public:
    BufferRack() {}
    BufferRack(Buffer* buffer) {
        set(buffer);
    }
    BufferRack(BufferSpecs specs) {
        set(specs);
    }
    BufferRack(const std::vector<Buffer*>& buffers) {
        set(buffers);
    }
    BufferRack(BufferRackSpecs specs) {
        set(specs);
    }
    ~BufferRack() {
    }

    // assign by reference operator
    BufferRack& operator=(const BufferRack& other) {
        if (this != &other) {
            for (auto buffer : _buffers) {
                delete buffer;
            }
            _buffers.clear();
            _data.clear();
            for (auto buffer : other._buffers) {
                _buffers.push_back(buffer);
                _data.push_back(buffer->getDataMod());
            }
        }
        return *this;
    }

    void set(Buffer* buffer) {
        set(std::vector<Buffer*>{buffer});
    }
    void set(BufferSpecs specs) {
        set(std::vector<Buffer*>{Buffer::create(specs)});
    }
    void set(BufferRackSpecs specs) {
        std::vector<Buffer*> buffers(specs.v.size());
        for (size_t i = 0; i < specs.v.size(); i++) {
            buffers[i] = Buffer::create(specs.v[i]);
        }
        set(buffers);
    }
    void set(const std::vector<Buffer*>& buffers) {
        _buffers.resize(buffers.size());
        _data.resize(buffers.size());
        _specs = BufferRackSpecs(buffers);
        for (size_t i = 0; i < buffers.size(); i++) {
            _buffers[i] = buffers[i];
            _data[i] = buffers[i]->getDataMod();
        }
    }

    void deallocateBuffers() {
        for (auto buffer : _buffers) {
            delete buffer;
        }
        _buffers.clear();
        _data.clear();
    }

    const BufferRackSpecs& getSpecs() const {
        return _specs;
    }
    std::string getSpecsString() const {
        std::string str = "BufferRack{";
        for (auto buffer : _buffers) {
            str += buffer->toString() + ", ";
        }
        str.pop_back();
        str.pop_back();
        str += ", datalist address=" + int_to_hex((size_t)_data.data()) + "}";
        return str;
    }
    void logSpecs(std::string prefix, spdlog::level::level_enum level = spdlog::level::debug) const {
        spdlog::log(level, "{}: {}", prefix, getSpecsString());
    }

    size_t getChannelCount() const {
        return std::accumulate(_buffers.begin(), _buffers.end(), 0, [](size_t sum, Buffer* buffer) { return sum + buffer->getChannelCount(); });
    }
    size_t getFrameCount() const {
        return _buffers.front()->getFrameCount();
    }
    size_t getSampleCount() const {
        return getChannelCount() * getFrameCount();
    }

    const std::vector<Buffer*>& getBuffers() const {
        return _buffers;
    }

    const Buffer* getBufferConst(size_t index = 0) const {
        return _buffers.at(index);
    }

    Buffer* getBufferMod(size_t index = 0) const {
        return _buffers.at(index);
    }

    const float* getDataConst(size_t index = 0) const {
        return _buffers.at(index)->getDataConst();
    }

    float* getDataMod(size_t index = 0) const {
        return _buffers.at(index)->getDataMod();
    }

    const float* const* getDataListConst() const {
        return static_cast<const float* const*>(_data.data());
    }

    float* const* getDataListMod() const {
        return _data.data();
    }

    void clearBuffers(cudaStream_t stream = nullptr) {
        for (auto buffer : _buffers) {
            buffer->clear(stream);
        }
    }

    size_t getBufferCount() const {
        return _buffers.size();
    }
};
