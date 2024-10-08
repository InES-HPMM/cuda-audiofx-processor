#include "signal_graph.hpp"

#include <string.h>

#include <log.hpp>

class PassThroughSignalGraph : public ISignalGraph {
   private:
    size_t _n_proc_frames;
    size_t _n_channels;

   public:
    PassThroughSignalGraph() : _n_proc_frames(0), _n_channels(0) {}
    ~PassThroughSignalGraph() {}

    void setup(const size_t n_proc_frames, const size_t n_in_channels, const size_t n_out_channels) override {
        if (n_in_channels != n_out_channels) {
            throw std::runtime_error("Input and output channel counts must match");
        }
        _n_proc_frames = n_proc_frames;
        _n_channels = n_in_channels;
    }
    void process(const std::vector<float*>& dst_bufs, const std::vector<float*>& src_bufs) override {
        if (dst_bufs.size() != _n_channels || src_bufs.size() != _n_channels) {
            throw std::runtime_error("Input and output channel counts must match");
        }
        for (size_t i = 0; i < _n_channels; i++) {
            memcpy(dst_bufs[i], src_bufs[i], _n_proc_frames * sizeof(float));
        }
    }
    void processAsync(const std::vector<float*>& dst_bufs, const std::vector<float*>& src_bufs) override {
        throw std::runtime_error("Not implemented");
    }

    size_t getInputChannelCount() override { return _n_channels; }
    size_t getOutputChannelCount() override { return _n_channels; }
};

ISignalGraph* ISignalGraph::createPassThroughSignalChain() {
    return new PassThroughSignalGraph();
}