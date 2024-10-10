#include "signal_generator.hpp"

#include <math.h>

struct SineConfig {
    float frequency;
    float amplitude;
    float phase_offset;
};

class SineGenerator : public SignalGenerator {
   private:
    size_t _sample_rate;
    SineConfig* _sine_waves;
    size_t n_sine_waves;

   public:
    SineGenerator(size_t sample_rate, size_t n_sine_waves, float* frequencies, float* amplitudes, float* phase_offsets)
        : _sample_rate(sample_rate), n_sine_waves(n_sine_waves), _sine_waves(new SineConfig[n_sine_waves]) {
        for (size_t i = 0; i < n_sine_waves; i++) {
            _sine_waves[i].frequency = frequencies[i];
            _sine_waves[i].amplitude = amplitudes[i];
            _sine_waves[i].phase_offset = phase_offsets[i];
        }
    }

    float* get_samples(float* out, size_t n_samples) {
        for (size_t i = 0; i < n_samples; i++) {
            out[i] = 0;
        }
        for (size_t i = 0; i < n_sine_waves; i++) {
            float phase = _sine_waves[i].phase_offset;
            float radians_per_sample = 2 * M_PI * _sine_waves[i].frequency / _sample_rate;
            for (size_t j = 0; j < n_samples; j++) {
                out[j] += _sine_waves[i].amplitude * sin(radians_per_sample * j + phase);
            }
            _sine_waves[i].phase_offset = fmod(_sine_waves[i].phase_offset + radians_per_sample * n_samples, 2 * M_PI);
        }
        return out;
    }

    float* get_duration_s(float* out, float duration_s) {
        return get_samples(out, static_cast<size_t>(duration_s * _sample_rate));
    }
};

SignalGenerator* SignalGenerator::createSineGenerator(size_t sample_rate, size_t n_sine_waves, float* frequencies, float* amplitudes, float* phase_offsets) {
    return new SineGenerator(sample_rate, n_sine_waves, frequencies, amplitudes, phase_offsets);
}
