#include <cuda_ext.cuh>

#include "gpu_fx.cu"

class FxIOMap : public GpuFx {
   protected:
    IMemCpyNode* _node;
    std::vector<size_t> _channel_mapping;
    cudaMemcpyKind _kind;

    virtual void allocateBuffers() override {
    }
    virtual void deallocateBuffers() override {
        delete _node;
    }

   public:
    FxIOMap(std::string name, size_t n_out_channels, std::vector<size_t> channel_mapping) : GpuFx(name, false), _channel_mapping(channel_mapping) {
        _n_out_channels = n_out_channels;
    }

    void configure(size_t process_buffer_size, size_t n_in_channels, size_t n_out_channels) override {
        if (n_out_channels != 0 && n_out_channels != _n_out_channels) {
            spdlog::warn("You are not allowed to change the number of output channels for {}", _name);
        }
        GpuFx::configure(process_buffer_size, n_in_channels, _n_out_channels);
    }
};

class FxInputMap : public FxIOMap {
   public:
    FxInputMap(std::vector<size_t> input_mapping) : FxIOMap("FxInputMap", input_mapping.size(), {input_mapping}) {}

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Segmented2Interleaved, dst->getDataMod(), src->getDataListMod(), sizeof(float), _n_proc_frames, _n_out_channels, _channel_mapping, cudaMemcpyDeviceToDevice, stream, &_node, capture_status);
        return stream;
    }
};

IGpuFx* IGpuFx::createInputMap(std::vector<size_t> input_mapping) {
    return new FxInputMap(input_mapping);
}

class FxOutputMap : public FxIOMap {
   public:
    FxOutputMap(std::vector<size_t> output_mapping) : FxIOMap("FxOutputMap", output_mapping.size(), {output_mapping}) {
    }

    void configure(size_t process_buffer_size, size_t n_in_channels, size_t n_out_channels) override {
        FxIOMap::configure(process_buffer_size, n_in_channels, n_out_channels);
        _output_specs = BufferRackSpecs(BufferSpecs(MemoryContext::DEVICE, _n_proc_frames), _n_out_channels);
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Interleaved2Segmented, (void*)dst->getDataListMod(), src->getDataMod(), sizeof(float), _n_proc_frames, _n_in_channels, _channel_mapping, cudaMemcpyDeviceToDevice, stream, &_node, capture_status);
        return stream;
    }
};
IGpuFx* IGpuFx::createOutputMap(std::vector<size_t> output_mapping) {
    return new FxOutputMap(output_mapping);
}