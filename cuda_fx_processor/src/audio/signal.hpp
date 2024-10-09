#pragma once

#include <enums.hpp>
#include <string>

class IPCMSignal;
class IFPSignal;

class IPCMSignal {
   public:
    static IPCMSignal* readFromFile(const std::string path);
    static IPCMSignal* copyFromBuffer(const char* data, const size_t n_samples, const size_t n_channels, const SampleRate sample_rate, const BitDepth bit_depth);
    static IPCMSignal* fromBuffer(char* data, const size_t n_samples, const size_t n_channels, const SampleRate sample_rate, const BitDepth bit_depth);

    virtual size_t getChannelCount() const = 0;
    virtual size_t getFrameCount() const = 0;
    virtual size_t getSampleCount() const { return getFrameCount() * getChannelCount(); }
    virtual SampleRate getSampleRate() const = 0;
    virtual size_t getSampleRateValue() const { return as_int(getSampleRate()); }
    virtual BitDepth getBitDepth() const = 0;
    virtual size_t getBitDepthValue() const { return as_int(getBitDepth()); }
    virtual size_t getByteDepth() const = 0;
    virtual size_t getByteCount() const = 0;

    virtual const char* getDataPtrConst() const = 0;
    virtual char* getDataPtrMod() const = 0;
    virtual void setData(char* data) = 0;

    virtual IFPSignal* toFPSignal(const ChannelOrder channel_order) const = 0;
    virtual IPCMSignal* clone() const = 0;
    virtual void writeToWav(const std::string path) const = 0;
};

class IFPSignal {
   public:
    static IFPSignal* create(const IFPSignal* signal);
    static IFPSignal* create(const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order);
    static IFPSignal* readFromFile(const std::string path, const ChannelOrder _channel_order);
    static IFPSignal* copyFromBuffer(const void* data, const size_t n_samples, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order);
    static IFPSignal* fromBuffer(void* data, const size_t n_samples, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order);

    virtual ~IFPSignal() {}

    virtual size_t getChannelCount() const = 0;
    virtual size_t getFrameCount() const = 0;
    virtual size_t getSampleCount() const { return getFrameCount() * getChannelCount(); }
    virtual SampleRate getSampleRate() const = 0;
    virtual size_t getSampleRateValue() const { return as_int(getSampleRate()); }
    virtual ChannelOrder getChannelOrder() const = 0;

    virtual const void* getDataPtrConst() const = 0;
    virtual void* getDataPtrMod() const = 0;
    virtual void* getDataPtrMod(const ChannelOrder channel_order) = 0;
    virtual void setData(void* data, const ChannelOrder channel_order) = 0;
    virtual IPCMSignal* toPCMSignal(const BitDepth bit_depth) const = 0;
    virtual void writeToWav(const std::string path, const BitDepth bit_depth) const = 0;
};
