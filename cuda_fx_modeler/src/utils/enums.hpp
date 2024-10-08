#pragma once
#include <iostream>

template <typename Enumeration>
constexpr auto as_int(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

enum class SampleRate {
    SR_48000 = 48000,
};
enum class BitDepth {
    BD_16 = 16,
    BD_24 = 24,
};
enum class ChannelOrder {
    INTERLEAVED,
    PLANAR,
};

enum class MemoryContext {
    HOST,
    DEVICE,
};
enum class AudioFileFormat {
    WAV,
};

enum class BiquadType {
    LOWPASS,
    HIGHPASS,
    BANDPASS,
    NOTCH,
    PEAK,
    LOWSHELF,
    HIGHSHELF,
};

enum class TrtEnginePrecision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

const std::string GetAudioFileExt(AudioFileFormat format);
SampleRate GetSampleRate(size_t sample_rate);
BitDepth GetBitDepth(size_t bit_depth);