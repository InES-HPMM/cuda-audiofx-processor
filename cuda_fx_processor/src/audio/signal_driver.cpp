#include <assert.h>

#include <chrono>
#include <coding.cuh>
#include <cstring>
#include <filesystem>

#include "driver.hpp"
#include "signal.hpp"
#include "signal_generator.hpp"

using namespace std;
namespace chrono = std::chrono;
namespace fs = std::filesystem;

class SignalDriver : public Driver {
   private:
    std::string _output_path;
    ISignalGraph* _signal_chain;

   public:
    SignalDriver() {
        _buffer_size = 1024;
        _is_running = false;
    }
    ~SignalDriver() {
        delete _signal_chain;
    }

    void start(bool async) override {
        _signal_chain->setup(_buffer_size, 2, 2);
        IFPSignal* output = IFPSignal::create(_buffer_size, _signal_chain->getOutputChannelCount(), SampleRate::SR_48000, ChannelOrder::PLANAR);
        _sample_rate = output->getSampleRateValue();
        float phase = 0;

        std::vector<float*> dst_slices(_signal_chain->getOutputChannelCount());

        size_t frame_count = output->getFrameCount() / _buffer_size;
        long max_processing_time = 0;

        float* sine_buffer = new float[_buffer_size];
        SignalGenerator* generator = SignalGenerator::createSineGenerator(_sample_rate, 2, new float[2]{1000, 8000}, new float[2]{0.5, 0.5}, new float[2]{0, 0});
        // SignalGenerator* generator = SignalGenerator::createSineGenerator(_sample_rate, 1, new float[1]{440}, new float[1]{0.5}, new float[1]{0});
        for (size_t i = 0; i < frame_count; i++) {
            generator->get_samples(sine_buffer, _buffer_size);

            std::vector<float*> src_slice(_signal_chain->getInputChannelCount(), sine_buffer);
            for (size_t c = 0; c < _signal_chain->getOutputChannelCount(); c++) {
                dst_slices[c] = ((float**)output->getDataPtrMod())[c] + i * _buffer_size;
            }
            _signal_chain->process(dst_slices, src_slice);
        }

        output->writeToWav(_output_path, BitDepth::BD_24);

        delete output;
    }
    void stop() override {
    }

    void addSignalChain(ISignalGraph* chain, std::vector<std::string> inputs, std::vector<std::string> outputs) override {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        _signal_chain = chain;
        _output_path = outputs[0];
    }

    void setBufferSize(size_t buffer_size) override {
        _buffer_size = buffer_size;
    }
};

Driver* Driver::createSignalDriver() {
    return new SignalDriver();
}