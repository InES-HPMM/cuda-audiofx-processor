#include "enums.hpp"

#include <stdexcept>
#include <string>

const std::string GetAudioFileExt(AudioFileFormat format) {
    switch (format) {
        case AudioFileFormat::WAV:
            return ".wav";
        default:
            throw std::invalid_argument("Invalid audio file format");
    }
}

SampleRate GetSampleRate(size_t sample_rate) {
    switch (sample_rate) {
        case as_int(SampleRate::SR_48000):
            return SampleRate::SR_48000;
        default:
            throw std::invalid_argument("Invalid sample rate");
    }
}

BitDepth GetBitDepth(size_t bit_depth) {
    switch (bit_depth) {
        case as_int(BitDepth::BD_16):
            return BitDepth::BD_16;
        case as_int(BitDepth::BD_24):
            return BitDepth::BD_24;
        default:
            throw std::invalid_argument("Invalid bit depth");
    }
}