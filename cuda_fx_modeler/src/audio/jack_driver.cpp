#include <assert.h>
#include <jack/jack.h>
#include <jack/midiport.h>
#include <unistd.h>

#include <chrono>
#include <convert.hpp>
#include <cstring>
#include <enums.hpp>
#include <global.hpp>
#include <map>
#include <set>

#include "driver.hpp"
#include "spdlog/spdlog.h"

using namespace std;
namespace chrono = std::chrono;

class SignalChainMapping {
   public:
    SignalChainMapping(ISignalGraph* signal_chain, std::vector<std::string> input_port_names, std::vector<std::string> output_port_names)
        : signal_chain(signal_chain), input_port_names(input_port_names), output_port_names(output_port_names) {}
    ISignalGraph* signal_chain;
    std::vector<std::string> input_port_names;
    std::vector<std::string> output_port_names;
    size_t getInputPortCount() { return input_port_names.size(); }
    size_t getOutputPortCount() { return output_port_names.size(); }
};

class JackDriver : public Driver {
   private:
    std::set<std::string> _input_port_names;
    std::set<std::string> _output_port_names;
    std::map<std::string, jack_port_t*> _ports;
    std::vector<SignalChainMapping> _signal_chain_mappings;
    jack_client_t* _handle;
    std::chrono::_V2::system_clock::time_point _last_process_call;

    static int processCallback(jack_nframes_t n_samples, void* arg) {
        assert(arg);
        auto jc = static_cast<JackDriver*>(arg);
        jc->process(n_samples);

        return 0;
    }

    static void shutdownCallback(void* arg) {
        auto jc = static_cast<JackDriver*>(arg);
        spdlog::warn("Jack shutting down...");
        jc->shutdown();
    }

    static int xrunCallback(void* arg) {
        auto jc = static_cast<JackDriver*>(arg);
        return 0;
    }

    jack_port_t* addInput(const std::string& name, const std::string& type = JACK_DEFAULT_AUDIO_TYPE, size_t bufferSize = 0) {
        spdlog::info("Registering {} input port: {} (buffer: {})", type, name, bufferSize);
        jack_port_t* p = jack_port_register(_handle, name.c_str(), type.c_str(), JackPortIsInput, bufferSize);
        assert(p);
        _ports[name] = p;
        return p;
    }

    jack_port_t* addOutput(const std::string& name, const std::string& type = JACK_DEFAULT_AUDIO_TYPE, size_t bufferSize = 0) {
        spdlog::info(name, "Registering {} output port: {} (buffer: {})", type, name, bufferSize);
        jack_port_t* p = jack_port_register(_handle, name.c_str(), type.c_str(), JackPortIsOutput, bufferSize);
        assert(p);
        _ports[name] = p;
        return p;
    }

    void process(size_t n_samples) {
        // auto start = chrono::high_resolution_clock::now();
        // spdlog::info("Time since last process call: {} us", chrono::duration_cast<chrono::microseconds>(start - _last_process_call).count());
        // _last_process_call = start;
        assert(n_samples == _buffer_size);
        std::map<std::string, float*> buffers;
        for (std::map<std::string, jack_port_t*>::iterator it = _ports.begin(); it != _ports.end(); ++it) {
            buffers[it->first] = (float*)jack_port_get_buffer(it->second, _buffer_size);
        }
        // spdlog::info("Got buffers from jack after {} us", chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count());

        for (SignalChainMapping chain_map : _signal_chain_mappings) {
            std::vector<float*> src_bufs(chain_map.getInputPortCount());
            for (size_t i = 0; i < chain_map.getInputPortCount(); i++) {
                src_bufs[i] = buffers[chain_map.input_port_names[i]];
            }
            std::vector<float*> dst_bufs(chain_map.getOutputPortCount());
            for (size_t i = 0; i < chain_map.getOutputPortCount(); i++) {
                dst_bufs[i] = buffers[chain_map.output_port_names[i]];
            }
            // spdlog::info("Call signal chain processing after {} us", chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count());
            // auto start = chrono::high_resolution_clock::now();
            chain_map.signal_chain->process(dst_bufs, src_bufs);
            // spdlog::info("Processed frame in {} us", chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count());
        }
        // sleep(800000 - (chrono::high_resolution_clock::now() - start).count());
        // while (chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() < 100) {
        // }
    }

    void shutdown() {
    }

   public:
    JackDriver() {}
    ~JackDriver() {}

    void start() {
        jack_status_t status;

        spdlog::info("Starting JACK client");

        _handle = jack_client_open("cuda_mixing_console", JackNoStartServer, &status, NULL);
        if (!_handle) {
            throw std::runtime_error("Failed to open JACK client");
        }
        if (status & JackNameNotUnique) {
            throw std::runtime_error("JACK client name not unique");
        }

        spdlog::info("Jack Client Open returned status: {}", as_int(status));

        errChk(jack_set_buffer_size(_handle, _buffer_size));
        errChk(jack_set_process_callback(_handle, processCallback, this));
        jack_set_xrun_callback(_handle, xrunCallback, this);
        jack_on_shutdown(_handle, shutdownCallback, this);

        _sample_rate = jack_get_sample_rate(_handle);
        spdlog::info("Samplerate: {}", _sample_rate);
        spdlog::info("Buffer size: {}", _buffer_size);

        _is_running = true;
        for (std::string port_name : _input_port_names) {
            addInput(port_name);
        }
        for (std::string port_name : _output_port_names) {
            addOutput(port_name);
        }

        for (SignalChainMapping chain_map : _signal_chain_mappings) {
            chain_map.signal_chain->setup(_buffer_size, chain_map.getInputPortCount(), chain_map.getOutputPortCount());
        }

        errChk(jack_activate(_handle));

        for (std::string dest_port_name : _input_port_names) {
            std::string source_port_name = "system:";
            jack_connect(_handle, source_port_name.append(dest_port_name).c_str(), jack_port_name(_ports[dest_port_name]));
        }
        for (std::string dest_port_name : _output_port_names) {
            std::string source_port_name = "system:";
            jack_connect(_handle, jack_port_name(_ports[dest_port_name]), source_port_name.append(dest_port_name).c_str());
        }

        _last_process_call = chrono::high_resolution_clock::now();
    }

    void stop() {
        assert(_handle);
        assert(_is_running);

        jack_client_close(_handle);
        _is_running = false;
        usleep(500000);
    }

    void addSignalChain(ISignalGraph* chain, std::vector<std::string> inputs, std::vector<std::string> outputs) {
        _signal_chain_mappings.push_back(SignalChainMapping(chain, inputs, outputs));
        for (std::string input : inputs) {
            _input_port_names.insert(input);
        }
        for (std::string output : outputs) {
            _output_port_names.insert(output);
        }
    }

    void setBufferSize(size_t buffer_size) {
        _buffer_size = buffer_size;

        if (_is_running) {
            stop();
            errChk(jack_set_buffer_size(_handle, _buffer_size));
            for (SignalChainMapping chain_map : _signal_chain_mappings) {
                chain_map.signal_chain->setup(_buffer_size, chain_map.getInputPortCount(), chain_map.getOutputPortCount());
            }
            start();
        } else {
            spdlog::warn("Jack handle not initialized. Buffer size will be set when client is started.");
        }
    }

    size_t getBufferSize() {
        return _buffer_size;
    }

    size_t getBitDepth() {
        return _bit_depth;
    }

    size_t getSampleRate() {
        return _sample_rate;
    }
};

Driver* Driver::createJackDriver() {
    return new JackDriver();
}