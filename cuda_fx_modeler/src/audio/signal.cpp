#include "signal.hpp"

#include <cassert>
#include <coding.cuh>
#include <convert.hpp>
#include <filesystem>
#include <fstream>

#include "spdlog/spdlog.h"
namespace fs = std::filesystem;

// Wav File Format: http://soundfile.sapp.org/doc/WaveFormat/

struct riff_chunk_header_t {
    uint32_t chunkId;
    uint32_t chunkSize;
};

static riff_chunk_header_t WAV_HDR_MAIN = {0x46464952, 0};     // LSB first hex representation of "RIFF"
static riff_chunk_header_t WAV_HDR_FROMAT = {0x20746D66, 16};  // LSB first hex representation of "fmt "
static riff_chunk_header_t WAV_HDR_DATA = {0x61746164, 0};     // LSB first hex representation of "data"

struct wav_format_t {
    uint16_t audioFormat;
    uint16_t n_channels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

static size_t getByteCount(size_t n_channels, size_t n_frames, size_t bit_depth) {
    return n_channels * n_frames * bit_depth / 8;
}
IPCMSignal* readFromWaveFile(const std::string path);
IFPSignal* readFromWaveFile(const std::string path, ChannelOrder channel_order);

class PCMSignal : public IPCMSignal {
   private:
    size_t _n_channels;
    size_t _n_frames;
    SampleRate _sample_rate;
    BitDepth _bit_depth;
    char* _data;

   public:
    PCMSignal(const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const BitDepth bit_depth) : _n_channels(n_channels), _n_frames(n_frames), _sample_rate(sample_rate), _bit_depth(bit_depth) {
        _data = new char[getByteCount()];
    }

    ~PCMSignal() { delete[] _data; }

    size_t getChannelCount() const override { return _n_channels; }
    size_t getFrameCount() const override { return _n_frames; }
    SampleRate getSampleRate() const override { return _sample_rate; }
    BitDepth getBitDepth() const override { return _bit_depth; }
    size_t getByteDepth() const override { return getBitDepthValue() / 8; }
    size_t getByteCount() const override { return _n_channels * _n_frames * as_int(_bit_depth) / 8; }

    const char* getDataPtrConst() const { return _data; }
    char* getDataPtrMod() const { return _data; }
    void setData(char* data) { memcpy(_data, data, getByteCount()); }

    IFPSignal* toFPSignal(const ChannelOrder channel_order) const override {
        auto signal = IFPSignal::create(_n_frames, _n_channels, _sample_rate, channel_order);
        pcm_to_float(signal->getDataPtrMod(), _data, _n_frames, _n_channels, getByteCount(), as_int(_bit_depth), channel_order);
        return signal;
    }

    IPCMSignal* clone() const override {
        IPCMSignal* signal = new PCMSignal(_n_frames, _n_channels, _sample_rate, _bit_depth);
        memcpy(signal->getDataPtrMod(), _data, getByteCount());
        return signal;
    }

    void writeToWav(const std::string path) const override {
        wav_format_t _wav_format = {
            1,
            (uint16_t)getChannelCount(),
            (uint32_t)as_int(getSampleRate()),
            (uint32_t)getByteCount(),
            (uint16_t)(getChannelCount() * getBitDepthValue() / 8),
            (uint16_t)as_int(getBitDepth())};

        WAV_HDR_DATA.chunkSize = getByteCount();
        WAV_HDR_MAIN.chunkSize = 36 + WAV_HDR_DATA.chunkSize;

        spdlog::info("Write to file: {}", path);
        std::ofstream os = std::ofstream(fs::path(path), (std::ofstream::openmode)(std::ofstream::trunc));

        os.write((char*)&WAV_HDR_MAIN, 8);
        os.write("WAVE", 4);
        os.write((char*)&WAV_HDR_FROMAT, 8);
        os.write((char*)&_wav_format, WAV_HDR_FROMAT.chunkSize);
        os.write((char*)&WAV_HDR_DATA, 8);
        os.write(getDataPtrConst(), WAV_HDR_DATA.chunkSize);
    }
};

IPCMSignal* IPCMSignal::readFromFile(const std::string path) {
    auto path_fs = fs::path(path);
    if (path_fs.extension() == GetAudioFileExt(AudioFileFormat::WAV))
        return readFromWaveFile(path);
    else {
        throw std::invalid_argument("Invalid audio file format");
    }
}

IPCMSignal* IPCMSignal::copyFromBuffer(const char* data, const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const BitDepth bit_depth) {
    IPCMSignal* signal = new PCMSignal(n_frames, n_channels, sample_rate, bit_depth);
    memcpy(signal->getDataPtrMod(), data, sizeof(float) * n_frames * n_channels);
    return signal;
}

IPCMSignal* IPCMSignal::fromBuffer(char* data, const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const BitDepth bit_depth) {
    IPCMSignal* signal = new PCMSignal(n_frames, n_channels, sample_rate, bit_depth);
    signal->setData(data);
    return signal;
}

class FPSignal : public IFPSignal {
   private:
    size_t _n_channels;
    size_t _n_frames;
    SampleRate _sample_rate;
    ChannelOrder _channel_order;
    float** _data_planar;
    float* _data_interleaved;

    void deletePlanarData() {
        if (_data_planar == nullptr) return;
        for (size_t c = 0; c < _n_channels; c++) {
            if (_data_planar[c] == nullptr) continue;
            delete[] _data_planar[c];
        }
        delete[] _data_planar;
    }

   public:
    FPSignal(const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order) : _n_channels(n_channels), _n_frames(n_frames), _sample_rate(sample_rate), _channel_order(channel_order) {
        switch (channel_order) {
            case ChannelOrder::PLANAR:
                _data_planar = new float*[n_channels];
                for (size_t c = 0; c < n_channels; c++) {
                    _data_planar[c] = new float[n_frames];
                }
                break;
            case ChannelOrder::INTERLEAVED:
                _data_interleaved = new float[n_channels * n_frames];
                break;
            default:
                throw std::invalid_argument("Invalid channel order");
        }
    }

    ~FPSignal() {
        if (_channel_order == ChannelOrder::PLANAR) {
            deletePlanarData();
        } else {
            delete[] _data_interleaved;
        }
    }

    size_t getChannelCount() const override { return _n_channels; }
    size_t getFrameCount() const override { return _n_frames; }
    SampleRate getSampleRate() const override { return _sample_rate; }
    ChannelOrder getChannelOrder() const override { return _channel_order; }

    const void* getDataPtrConst() const override {
        switch (_channel_order) {
            case ChannelOrder::PLANAR:
                return (void*)_data_planar;
            case ChannelOrder::INTERLEAVED:
                return (void*)_data_interleaved;
            default:
                throw std::invalid_argument("Invalid channel order");
        }
    }

    void* getDataPtrMod() const override {
        switch (_channel_order) {
            case ChannelOrder::PLANAR:
                return (void*)_data_planar;
            case ChannelOrder::INTERLEAVED:
                return (void*)_data_interleaved;
            default:
                throw std::invalid_argument("Invalid channel order");
        }
    }

    void* getDataPtrMod(ChannelOrder channel_order) override {
        if (channel_order == ChannelOrder::PLANAR) {
            if (_channel_order == ChannelOrder::INTERLEAVED) {
                spdlog::warn("Data is interleaved. Converting to planar.");
                _data_planar = new float*[_n_channels];
                for (size_t c = 0; c < _n_channels; c++) {
                    _data_planar[c] = new float[_n_frames];
                    for (size_t s = 0; s < _n_frames; s++) {
                        _data_planar[c][s] = _data_interleaved[c * _n_frames + s];
                    }
                }
            }
            return (void*)_data_planar;
        } else {
            if (_channel_order == ChannelOrder::PLANAR) {
                spdlog::warn("FPSignal Data is planar. Converting to interleaved.");
                _data_interleaved = new float[_n_channels * _n_frames];
                for (size_t c = 0; c < _n_channels; c++) {
                    for (size_t s = 0; s < _n_frames; s++) {
                        _data_interleaved[c * _n_frames + s] = _data_planar[c][s];
                    }
                }
            }
            return (void*)_data_interleaved;
        }
    }

    void setData(void* data, ChannelOrder channel_order) override {
        switch (channel_order) {
            case ChannelOrder::PLANAR:
                deletePlanarData();
                _data_planar = ((float**)data);
                if (_channel_order == ChannelOrder::INTERLEAVED) delete[] _data_interleaved;
                _channel_order = ChannelOrder::PLANAR;
                break;
            case ChannelOrder::INTERLEAVED:
                delete[] _data_interleaved;
                _data_interleaved = ((float*)data);
                if (_channel_order == ChannelOrder::PLANAR) deletePlanarData();
                _channel_order = ChannelOrder::INTERLEAVED;
                break;
            default:
                throw std::invalid_argument("Invalid channel order");
        }
    }
    IPCMSignal* toPCMSignal(const BitDepth bit_depth) const override {
        IPCMSignal* signal = new PCMSignal(_n_frames, _n_channels, _sample_rate, bit_depth);
        float_to_pcm(signal->getDataPtrMod(), getDataPtrConst(), _n_frames, _n_channels, signal->getByteCount(), signal->getBitDepthValue(), _channel_order);
        return signal;
    }
    void writeToWav(const std::string path, BitDepth bit_depth) const override {
        IPCMSignal* pcm_signal = toPCMSignal(bit_depth);
        pcm_signal->writeToWav(path);
        delete pcm_signal;
    }
};

IFPSignal* IFPSignal::create(const IFPSignal* signal) {
    return new FPSignal(signal->getFrameCount(), signal->getChannelCount(), signal->getSampleRate(), signal->getChannelOrder());
}

IFPSignal* IFPSignal::create(const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order) {
    return new FPSignal(n_frames, n_channels, sample_rate, channel_order);
}

IFPSignal* IFPSignal::readFromFile(const std::string path, const ChannelOrder channel_order) {
    auto path_fs = fs::path(path);
    if (path_fs.extension() == GetAudioFileExt(AudioFileFormat::WAV)) {
        IPCMSignal* pcm_signal = readFromWaveFile(path);
        IFPSignal* signal = pcm_signal->toFPSignal(channel_order);
        delete pcm_signal;
        return signal;
    } else {
        throw std::invalid_argument("Invalid audio file format");
    }
}

IFPSignal* IFPSignal::copyFromBuffer(const void* data, const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order) {
    IFPSignal* signal = new FPSignal(n_frames, n_channels, sample_rate, channel_order);
    memcpy(signal->getDataPtrMod(channel_order), data, sizeof(float) * n_frames * n_channels);
    return signal;
}

IFPSignal* IFPSignal::fromBuffer(void* data, const size_t n_frames, const size_t n_channels, const SampleRate sample_rate, const ChannelOrder channel_order) {
    IFPSignal* signal = new FPSignal(n_frames, n_channels, sample_rate, channel_order);
    signal->setData(data, channel_order);
    return signal;
}

IPCMSignal* readFromWaveFile(const std::string path) {
    std::ifstream is = std::ifstream(fs::path(path), std::ifstream::binary);
    char* readFileFormat = (char*)alloca(4);
    char* char4 = (char*)alloca(4);

    riff_chunk_header_t hdr_main;
    is.read((char*)&hdr_main, 8);
    spdlog::debug("Filename: {}", path.c_str());
    spdlog::debug("Chunk ID: {}", int_to_char_ptr(char4, &hdr_main.chunkId, 4));
    spdlog::debug("Chunk ID: {}", hdr_main.chunkId);
    spdlog::debug("Chunk Size: {}", hdr_main.chunkSize);

    is.read(readFileFormat, 4);
    spdlog::debug("Format: {}", readFileFormat);

    riff_chunk_header_t hdr_format;
    is.read((char*)&hdr_format, 8);
    spdlog::debug("Format Chunk ID: {}", int_to_char_ptr(char4, &hdr_format.chunkId, 3));
    spdlog::debug("Format Chunk ID: {}", hdr_format.chunkId);
    spdlog::debug("Format Chunk Size: {}", hdr_format.chunkSize);

    wav_format_t wav_format;
    is.read((char*)&wav_format, hdr_format.chunkSize);
    spdlog::debug("Format: {}", wav_format.audioFormat);
    spdlog::debug("Num Channels: {}", wav_format.n_channels);
    spdlog::debug("Sample Rate: {}", wav_format.sampleRate);
    spdlog::debug("Byte Rate: {}", wav_format.byteRate);
    spdlog::debug("Block Align: {}", wav_format.blockAlign);
    spdlog::debug("Bits per Sample: {}", wav_format.bitsPerSample);

    riff_chunk_header_t hdr_data;
    is.read((char*)&hdr_data, 8);
    spdlog::debug("Data Header Chunk Id: {}", int_to_char_ptr(char4, &hdr_data.chunkId, 4));
    spdlog::debug("Data Header Chunk ID: {}", hdr_data.chunkId);
    spdlog::debug("Data Header Chunk Size: {}", hdr_data.chunkSize);

    assert(!memcmp(int_to_char_ptr(char4, &hdr_main.chunkId, 4), "RIFF", 4));
    assert(!memcmp(readFileFormat, "WAVE", 4));
    assert(!memcmp(int_to_char_ptr(char4, &hdr_format.chunkId, 4), "fmt ", 4));

    size_t n_frames = hdr_data.chunkSize / (wav_format.bitsPerSample >> 3) / wav_format.n_channels;
    size_t n_bytes = getByteCount(wav_format.n_channels, n_frames, wav_format.bitsPerSample);

    spdlog::debug("Number of Frames: {}", n_frames);
    spdlog::debug("Duration: {:.2}s", n_frames / wav_format.sampleRate);

    IPCMSignal* signal = new PCMSignal(n_frames, wav_format.n_channels, GetSampleRate(wav_format.sampleRate), GetBitDepth(wav_format.bitsPerSample));
    is.read(signal->getDataPtrMod(), hdr_data.chunkSize);

    return signal;
}
