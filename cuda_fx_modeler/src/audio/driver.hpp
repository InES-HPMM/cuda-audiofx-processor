#pragma once

#include <string>

#include "signal_graph.hpp"

class Driver {
   protected:
    size_t _buffer_size;
    size_t _bit_depth;
    size_t _sample_rate;
    bool _is_running;

   public:
    static Driver* createJackDriver();
    static Driver* createJackDriverAsync();
    static Driver* createFileDriver();
    static Driver* createSignalDriver();

    virtual ~Driver() {}
    virtual void start() = 0;
    virtual void stop() = 0;

    virtual void addSignalChain(ISignalGraph* chain, std::vector<std::string> inputs, std::vector<std::string> outputs) = 0;

    virtual void setBufferSize(size_t buffer_size) = 0;

    size_t getBufferSize() { return _buffer_size; }
    size_t getBitDepth() { return _bit_depth; }
    size_t getSampleRate() { return _sample_rate; }
};