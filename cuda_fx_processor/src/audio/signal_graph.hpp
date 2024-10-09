#pragma once

#include <block_buffer.cuh>
#include <thread>
#include <vector>

#include "fx.hpp"
class ISignalGraph : public std::thread {
   public:
    static ISignalGraph *createPassThroughSignalChain();

    virtual ~ISignalGraph() {}

    virtual void setup(const size_t n_proc_frames, const size_t n_in_channels, const size_t n_out_channels = 0) = 0;
    virtual void process(const std::vector<float *> &dst_bufs, const std::vector<float *> &src_bufs) = 0;
    virtual void processAsync(const std::vector<float *> &dst_bufs, const std::vector<float *> &src_bufs) { throw std::runtime_error("Not implemented"); };
    virtual void teardown() { throw std::runtime_error("Not implemented"); };
    virtual void startProcessThread() { throw std::runtime_error("Not implemented"); };
    virtual void stopProcessThread() { throw std::runtime_error("Not implemented"); };

    virtual size_t getInputChannelCount() = 0;
    virtual size_t getOutputChannelCount() = 0;
};