
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
    TrtEnginePrecision _precision;
    std::filesystem::path _onnx_model_path;
    std::filesystem::path _trt_model_dir;
    std::filesystem::path _trt_model_path;
    size_t _receptive_field_size;
    size_t _buf_size_in;
    size_t _buf_size_out;
    float* _input_buffer = nullptr;
    float* _output_buffer = nullptr;
    const char* _model_input_name = nullptr;
    const char* _model_output_name = nullptr;
    NvInverLogger _nv_infer_logger;
    std::unique_ptr<nvinfer1::IRuntime> _runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> _context = nullptr;

    bool loadTRTModel() {
        spdlog::debug("FxTrtEngine.loadTRTModel Attempting to load trt engine {}", _trt_model_path.filename().string());
        if (!std::filesystem::exists(_trt_model_path) && !buildTRTModel()) {
            spdlog::error("FxTrtEngine.loadTRTModel Couldn't find engine file at path: {}", _trt_model_path.string());
            return false;
        }

        std::ifstream file(_trt_model_path.string(), std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            spdlog::error("FxTrtEngine.loadTRTModel Unable to read engine file");
            return false;
        }
        file.close();

        // Create a runtime to deserialize the engine file.
        _runtime = std::unique_ptr<IRuntime>{createInferRuntime(_nv_infer_logger)};
        if (!_runtime) {
            spdlog::error("FxTrtEngine.loadTRTModel Failed to create infer runtime");
            return false;
        }

        // Create an engine, a representation of the optimized model.
        _engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!_engine) {
            spdlog::error("FxTrtEngine.loadTRTModel Failed to create engine");
            return false;
        }

        size_t n_bindings = _engine->getNbBindings();
        if (n_bindings != 2) {
            spdlog::error("FxTrtEngine.loadTRTModel Expected 2 bindings but got {}", n_bindings);
        }
        _model_input_name = _engine->getBindingName(0);
        _model_output_name = _engine->getBindingName(1);

        // The execution context contains all of the state associated with a particular invocation
        _context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
        if (!_context) {
            return false;
        }

        spdlog::debug("FxTrtEngine.loadTRTModel Successfully loaded network {}", _trt_model_path.filename().string());
        return true;
    }

    bool buildTRTModel() {
        if (!std::filesystem::exists(_onnx_model_path)) {
            spdlog::error("FxTrtEngine.buildTRTModel Could not find model at path: {}", _trt_model_path.string());
            return false;
        }
        spdlog::info("FxTrtEngine.buildTRTModel No engine found at path: {}. Regenerating ... this could take a while.", _trt_model_path.c_str());

        // Create our engine builder.
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(_nv_infer_logger));
        if (!builder) {
            spdlog::error("FxTrtEngine.buildTRTModel Create engine builder failed");
            return false;
        }

        // Define an explicit batch size and then create the network (implicit batch size is deprecated).
        // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
        auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            spdlog::error("FxTrtEngine.buildTRTModel Network creation failed");
            return false;
        }

        // Create a parser for reading the onnx file.
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, _nv_infer_logger));
        if (!parser) {
            spdlog::error("FxTrtEngine.buildTRTModel Parser creation failed");
            return false;
        }

        // Parse the buffer we read into memory.
        auto parsed = parser->parseFromFile(_onnx_model_path.c_str(), static_cast<int32_t>(ILogger::Severity::kVERBOSE));
        if (!parsed) {
            spdlog::error("FxTrtEngine.buildTRTModel Parsing failed");
            return false;
        }

        // Ensure that all the inputs have the same batch size
        const auto numInputs = network->getNbInputs();
        const auto numOutputs = network->getNbOutputs();
        if (numInputs != 1 || numOutputs != 1) {
            spdlog::error("FxTrtEngine.buildTRTModel Model should have exactly 1 input and 1 output but has {} inputs and {} outputs!", numInputs, numOutputs);
            return false;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        IOptimizationProfile* optProfile = builder->createOptimizationProfile();
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims2(1, _buf_size_in));
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims2(1, _buf_size_in));
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims2(1, _buf_size_in));
        config->addOptimizationProfile(optProfile);

        // Ensure the GPU supports TF36 inference
        if (!builder->platformHasTf32()) {
            spdlog::warn("FxTrtEngine.buildTRTModel GPU doesn't support Tf32 precision");
        }
        config->setFlag(BuilderFlag::kTF32);

        // CUDA stream used for profiling by the builder.
        cudaStream_t profileStream;
        gpuErrChk(cudaStreamCreate(&profileStream));
        config->setProfileStream(profileStream);

        spdlog::info("Building {}", _trt_model_path.filename().string());
        // Build the engine
        // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
        // Doing so will provide you with more information on why exactly it is failing.
        std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            spdlog::error("FxTrtEngine.buildTRTModel Failed to build TRT engine");
            return false;
        }

        // Write the engine to disk
        std::ofstream outfile(_trt_model_path.string(), std::ofstream::binary);
        outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

        spdlog::info("Success, saved engine to {}", _trt_model_path.string());

        gpuErrChk(cudaStreamDestroy(profileStream));
        return true;
    }

    std::string setTRTModelPath() {
        _trt_model_path = _trt_model_dir / _onnx_model_path.stem();

        // Serialize the specified options into the filename
        if (_precision == TrtEnginePrecision::FP16) {
            _trt_model_path += ".fp16";
        } else if (_precision == TrtEnginePrecision::FP32) {
            _trt_model_path += ".fp32";
        } else {
            _trt_model_path += ".int8";
        }

        _trt_model_path += "." + std::to_string(_buf_size_out) + ".trt";

        return _trt_model_path;
    }

    void allocateBuffers() override {
        gpuErrChk(cudaMalloc(&_input_buffer, _buf_size_in * sizeof(float)));
        gpuErrChk(cudaMalloc(&_output_buffer, _buf_size_out * sizeof(float)));

        setTRTModelPath();
        if (!loadTRTModel()) {
            throw std::runtime_error("Failed to load or build TRT model");
        }
        // Ensure all dynamic bindings have been defined.
        if (!_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }
        _context->setTensorAddress(_model_input_name, _input_buffer);
        _context->setTensorAddress(_model_output_name, _output_buffer);
    }

    void deallocateBuffers() override {
        if (_input_buffer) gpuErrChk(cudaFree(_input_buffer));
        if (_output_buffer) gpuErrChk(cudaFree(_output_buffer));
        if (_src_node) delete _src_node;
        if (_dest_node) delete _dest_node;
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
        _context.release();
        _engine.release();
        _runtime.release();
    }

    cudaStream_t _process(cudaStream_t stream, float* dst, const float* src, cudaStreamCaptureStatus capture_status) {
        // extract the first audio channel, since the model expects mono audio
        IMemCpyNode::launchOrRecord1D(_input_buffer + _receptive_field_size, src, sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _src_node, capture_status);

        if (!_context->enqueueV3(stream)) {
            throw std::runtime_error("Failed to inference trt enging" + _trt_model_path.string());
        }

        // copy the mono output of the model to both channels of the output buffer
        IMemCpyNode::launchOrRecord1D(dst, _output_buffer, sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _dest_node, capture_status);

        return stream;
    }

   public:
    FxTrtEngine(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision) : GpuFx("FxTrtEngine") {
        _onnx_model_path = std::filesystem::path(onnx_model_path);
        _trt_model_dir = std::filesystem::path(trt_model_dir);
        _precision = precision;
        _receptive_field_size = 8190;  // for dilated convnet: sum of all dilations in the model, TODO: read from difference in input and output tensor sizes
    }

    ~FxTrtEngine() {
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        if (n_input_channels > 1 || n_output_channels > 1) {
            spdlog::warn("{} can only process a single channel. n_input_channels and n_output_channels are overwritten by 1", _name);
        }
        GpuFx::configure(process_buffer_size, 1, 1);
        // a dilated convnet reduces the input by the size of the receptive field, to provide each output sample with temporal context
        // since the output needs to be the same size as the drivers buffer size, the input buffer size is increased by the receptive field
        _buf_size_in = process_buffer_size + _receptive_field_size;
        _buf_size_out = process_buffer_size;
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        _src_node->updateSrcPtr(src->getDataMod(), procGraphExec);
        _dest_node->updateSrcPtr(dst->getDataMod(), procGraphExec);
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        gpuErrChk(cudaMemsetAsync(_input_buffer, 0, _buf_size_in * sizeof(float), stream));
        gpuErrChk(cudaMemsetAsync(_output_buffer, 0, _buf_size_out * sizeof(float), stream));

        return stream;
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        return _process(stream, dst->getDataMod(), src->getDataMod(), capture_status);
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // rewind the input buffer by _buf_size_out to move the current audio chunk into the receptive field
        gpuErrChk(cudaMemcpyAsync(_input_buffer, _input_buffer + _buf_size_out, (_buf_size_in - _buf_size_out) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return stream;
    }

    GpuFx* clone() override {
        return new FxTrtEngine(_onnx_model_path.string(), _trt_model_dir.string(), _precision);
    }
};

IGpuFx* IGpuFx::createTrtEngine(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision, size_t buf_size) {
    return new FxTrtEngine(onnx_model_path, trt_model_dir, precision);
}