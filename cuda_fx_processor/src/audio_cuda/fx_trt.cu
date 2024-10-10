
#include <cuda_ext.cuh>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <trt_engine.cuh>

#include "NvOnnxParser.h"
#include "gpu.cuh"
#include "gpu_fx.cu"
#include "log.hpp"
#include "spdlog/spdlog.h"

using namespace nvinfer1;

class NvInverLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept {
        spdlog::level::level_enum level = spdlog::level::level_enum::critical;
        // suppress info-level messages
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                level = spdlog::level::level_enum::err;
                break;
            case Severity::kWARNING:
                level = spdlog::level::level_enum::warn;
                break;
            case Severity::kINFO:
                level = spdlog::level::level_enum::debug;
                break;
            case Severity::kVERBOSE:
                level = spdlog::level::level_enum::trace;
                break;
        }
        spdlog::log(level, "NvInfer: {}", msg);
    }
};

class FxTrtEngine : public GpuFx {
   private:
    IMemCpyNode* _src_node = nullptr;
    IMemCpyNode* _dest_node = nullptr;
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
        if (_dest_node) delete _dest_node;
    }

   public:
    FxTrtEngine(TrtEngine* trt_engine) : GpuFx("FxNam"), _trt_engine(trt_engine) {}
    FxTrtEngine(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision) : FxTrtEngine(TrtEngine::create(onnx_model_path, trt_model_dir, precision)) {}

    ~FxTrtEngine() {
        delete _trt_engine;
    }

    GpuFx* clone() override {
        return new FxTrtEngine(_trt_engine->clone());
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        if (n_input_channels > 1 || n_output_channels > 1) {
            spdlog::warn("{} can only process a single channel. n_input_channels and n_output_channels are overwritten by 1", _name);
        }
        GpuFx::configure(process_buffer_size, 1, 1);
        _trt_engine->configure(process_buffer_size, n_input_channels, n_output_channels);
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        _src_node->updateSrcPtr(src->getDataMod(), procGraphExec);
        _dest_node->updateSrcPtr(dst->getDataMod(), procGraphExec);
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        _trt_engine->setup(stream);

        return stream;
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        // extract the first audio channel, since the model expects mono audio
        IMemCpyNode::launchOrRecord1D(_trt_engine->getInputBuffer(0)->getDataMod() + _receptive_field_size, src->getDataMod(), sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _src_node, capture_status);

        _trt_engine->inference(stream);

        // copy the mono output of the model to both channels of the output buffer
        IMemCpyNode::launchOrRecord1D(dst->getDataMod(), _trt_engine->getOutputBuffer(0)->getDataMod(), sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _dest_node, capture_status);
        return stream;
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // rewind the input buffer by _buf_size_out to move the current audio chunk into the receptive field
        gpuErrChk(cudaMemcpyAsync(_trt_engine->getInputBuffer(0)->getDataMod(), _trt_engine->getInputBuffer(0)->getDataMod() + _buf_size_out, (_buf_size_in - _buf_size_out) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return stream;
    }
};

IGpuFx* IGpuFx::createTrtEngine(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision, size_t buf_size) {
    return new FxTrtEngine(onnx_model_path, trt_model_dir, precision);
}