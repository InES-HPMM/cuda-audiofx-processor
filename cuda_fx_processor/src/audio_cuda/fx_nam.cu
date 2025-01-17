
#include <cuda_ext.cuh>
#include <trt_engine.cuh>

#include "gpu.cuh"
#include "gpu_fx.cu"
#include "kernels.cuh"
#include "log.hpp"
#include "spdlog/spdlog.h"

class FxNam : public GpuFx {
   private:
    IMemCpyNode* _src_node = nullptr;
    TrtEngine* _trt_engine = nullptr;
    size_t _receptive_field_size;
    size_t _buf_size_in;
    size_t _buf_size_out;

    void allocateBuffers() override {
        _trt_engine->allocate();
        _receptive_field_size = _trt_engine->getInputBuffer(0)->getFrameCount() - _trt_engine->getOutputBuffer(0)->getFrameCount();
        // a dilated convnet reduces the input by the size of the receptive field, to provide each output sample with temporal context
        // since the output needs to be the same size as the drivers buffer size, the input buffer size is increased by the receptive field
        _buf_size_in = _n_proc_frames + _receptive_field_size;
        _buf_size_out = _n_proc_frames;
    }

    void deallocateBuffers() override {
        _trt_engine->deallocate();
        if (_src_node) delete _src_node;
    }

   public:
    FxNam(TrtEngine* trt_engine) : GpuFx("FxNam"), _trt_engine(trt_engine) {}
    FxNam(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision) : FxNam(TrtEngine::create(onnx_model_path, trt_model_dir, precision)) {}

    ~FxNam() {
        delete _trt_engine;
    }

    GpuFx* clone() override {
        return new FxNam(_trt_engine->clone());
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        if (n_input_channels > 1 || n_output_channels > 1) {
            spdlog::warn("{} can only process a single channel. n_input_channels and n_output_channels are overwritten by 1", _name);
        }
        GpuFx::configure(process_buffer_size, 1, 1);
        _trt_engine->configure(process_buffer_size, 8190, n_input_channels, n_output_channels);
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        _src_node->updateExecSrcPtr(src->getDataMod(), procGraphExec);
        _mix_node->updateExecKernelParamAt(0, dst->getDataMod(), procGraphExec);
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        _trt_engine->setup(stream);

        return stream;
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        // extract the first audio channel, since the model expects mono audio
        IMemCpyNode::launchOrRecord1D(_trt_engine->getInputBuffer(0)->getDataMod() + _receptive_field_size, src->getDataMod(), sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, &_src_node, capture_status);

        _trt_engine->inference(stream);

        // mix the processed audio with the original audio
        auto dst_data = dst->getDataMod();
        auto src_data = src->getDataMod();
        auto trt_output = _trt_engine->getOutputBuffer(0)->getDataMod();
        IKernelNode::launchOrRecord(1, _n_proc_samples, 0, (void*)fff_mix, new void*[5]{&dst_data, &src_data, &trt_output, &_n_proc_samples, &_mix_ratio}, stream, &_mix_node, capture_status);
        return stream;
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // rewind the input buffer by _buf_size_out to move the current audio chunk into the receptive field
        gpuErrChk(cudaMemcpyAsync(_trt_engine->getInputBuffer(0)->getDataMod(), _trt_engine->getInputBuffer(0)->getDataMod() + _buf_size_out, (_buf_size_in - _buf_size_out) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return stream;
    }
};

IGpuFx* IGpuFx::createNam(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision, size_t buf_size) {
    return new FxNam(onnx_model_path, trt_model_dir, precision);
}