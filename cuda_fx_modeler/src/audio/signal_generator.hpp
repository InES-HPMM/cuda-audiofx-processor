#pragma once
#include <cstddef>
class SignalGenerator {
   protected:
    size_t _sample_rate;

   public:
    static SignalGenerator* createSineGenerator(size_t sample_rate, size_t n_sine_waves, float* frequencies, float* amplitudes, float* phase_offsets);

    virtual float* get_samples(float* out, size_t n_samples) = 0;
    virtual float* get_duration_s(float* out, float duration_s) = 0;
};