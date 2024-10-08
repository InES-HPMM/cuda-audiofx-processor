#include <assert.h>

#include <chrono>
#include <coding.cuh>
#include <cstring>
#include <filesystem>

#include "driver.hpp"
#include "signal.hpp"

using namespace std;
namespace chrono = std::chrono;
namespace fs = std::filesystem;

class FileDriver : public Driver {
   private:
    std::string _output_path;
    ISignalGraph* _signal_chain;

    IFPSignal* _input;
    IFPSignal* _output;

   public:
    FileDriver() {
        _buffer_size = 96;
        _is_running = false;
    }
    ~FileDriver() {
        delete _input;
        delete _output;
    }

    void start() override {
        _signal_chain->setup(_buffer_size, _input->getChannelCount());
        _output = IFPSignal::create(_input->getFrameCount(), _signal_chain->getOutputChannelCount(), _input->getSampleRate(), _input->getChannelOrder());
        _sample_rate = as_int(_input->getSampleRate());

        std::vector<float*> src_buffers_slice(_output->getChannelCount());
        std::vector<float*> dest_buffers_slice(_output->getChannelCount());
        size_t frame_count = _input->getFrameCount() / _buffer_size;
        long max_processing_time = 0;

        for (size_t i = 0; i < frame_count; i++) {
            auto start = chrono::high_resolution_clock::now();
            for (size_t c = 0; c < _input->getChannelCount(); c++) {
                src_buffers_slice[c] = ((float**)_input->getDataPtrConst())[c] + i * _buffer_size;
            }
            for (size_t c = 0; c < _output->getChannelCount(); c++) {
                dest_buffers_slice[c] = ((float**)_output->getDataPtrConst())[c] + i * _buffer_size;
            }

            _signal_chain->process(dest_buffers_slice, src_buffers_slice);
            auto end = chrono::high_resolution_clock::now();
            auto processing_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
            if (processing_time > max_processing_time) {
                max_processing_time = processing_time;
            }
            // Log::info("File Driver", "Processed frame %d in %d us. Max process time %d", i, processing_time, max_processing_time);
        }

        _output->writeToWav(_output_path, BitDepth::BD_24);
    }
    void stop() override {
    }

    void addSignalChain(ISignalGraph* chain, std::vector<std::string> inputs, std::vector<std::string> outputs) override {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        _signal_chain = chain;

        _input = IFPSignal::readFromFile(inputs[0], ChannelOrder::PLANAR);
        _output_path = outputs[0];
    }

    void setBufferSize(size_t buffer_size) override {
        _buffer_size = buffer_size;
    }
};

Driver* Driver::createFileDriver() {
    return new FileDriver();
}