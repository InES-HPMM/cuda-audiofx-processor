
#include <filesystem>
#include <fstream>
#include <gpu.cuh>
#include <iostream>
#include <map>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <stdexcept>

#include "NvOnnxParser.h"
#include "spdlog/spdlog.h"
#include "trt_engine.cuh"
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

class TrtEngineImpl : public TrtEngine {
   private:
    TrtEnginePrecision _precision;
    std::filesystem::path _onnx_model_path;
    std::filesystem::path _trt_model_dir;
    std::filesystem::path _engine_name;
    std::filesystem::path _engine_path;
    const char* _model_input_name = nullptr;
    const char* _model_output_name = nullptr;
    NvInverLogger _nv_infer_logger;
    std::unique_ptr<nvinfer1::IRuntime> _runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> _context = nullptr;

    std::map<std::string, BufferSpecs> _buffer_specs;
    std::map<std::string, Buffer*> _buffers;
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;

    std::filesystem::path getEngineName(std::filesystem::path model_name, size_t process_buffer_size) {
        std::filesystem::path engine_name = model_name;
        // Serialize the specified options into the filename
        if (_precision == TrtEnginePrecision::FP16) {
            engine_name += ".fp16";
        } else if (_precision == TrtEnginePrecision::FP32) {
            engine_name += ".fp32";
        } else {
            engine_name += ".int8";
        }

        engine_name += "." + std::to_string(process_buffer_size) + ".trt";

        return engine_name;
    }

    bool buildTRTModel(Dims opt_in_shape, Dims min_in_shape = Dims{}, Dims max_in_shape = Dims{}) {
        if (!std::filesystem::exists(_onnx_model_path)) {
            spdlog::error("FxTrtEngine.buildTRTModel Could not find model at path: {}", _engine_path.string());
            return false;
        }

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
        if (max_in_shape.nbDims == 0) max_in_shape = opt_in_shape;
        if (min_in_shape.nbDims == 0) min_in_shape = opt_in_shape;
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, min_in_shape);
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, opt_in_shape);
        optProfile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, max_in_shape);
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

        spdlog::info("Building {}", _engine_path.filename().string());
        // Build the engine
        // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
        // Doing so will provide you with more information on why exactly it is failing.
        std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            spdlog::error("FxTrtEngine.buildTRTModel Failed to build TRT engine");
            return false;
        }

        // Write the engine to disk
        std::ofstream outfile(_engine_path.string(), std::ofstream::binary);
        outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

        spdlog::info("Success, saved engine to {}", _engine_path.string());

        gpuErrChk(cudaStreamDestroy(profileStream));
        return true;
    }

    bool loadTRTModel() {
        spdlog::debug("Attempting to load trt engine {}", _engine_path.filename().string());
        if (!std::filesystem::exists(_engine_path)) {
            spdlog::error("Couldn't find engine file at path: {}", _engine_path.string());
            return false;
        }

        std::ifstream file(_engine_path.string(), std::ios::binary | std::ios::ate);
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
        for (size_t i = 0; i < n_bindings; i++) {
            auto dims = _engine->getBindingDimensions(i);
            auto name = std::string(_engine->getBindingName(i));
            if (dims.nbDims < 2 || dims.nbDims > 3) {
                spdlog::error("Buffer dimension must be 2 for single channel or 3 for multi-channel processing. Got {}", std::to_string(dims.nbDims));
                return false;
            }
            if (dims.nbDims == 2) {
                _buffer_specs[name] = BufferSpecs(MemoryContext::DEVICE, dims.d[1]);
            } else {
                _buffer_specs[name] = BufferSpecs(MemoryContext::DEVICE, dims.d[2], dims.d[1]);
            }
            if (_engine->bindingIsInput(i)) {
                _input_names.push_back(name);
            } else {
                _output_names.push_back(name);
            }
        }

        // The execution context contains all of the state associated with a particular invocation
        _context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
        if (!_context) {
            return false;
        }

        spdlog::debug("FxTrtEngine.loadTRTModel Successfully loaded network {}", _engine_path.filename().string());
        return true;
    }

   public:
    TrtEngineImpl(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision) {
        _onnx_model_path = std::filesystem::path(onnx_model_path);
        _trt_model_dir = std::filesystem::path(trt_model_dir);
        _precision = precision;
    }

    TrtEngine* clone() override {
        return new TrtEngineImpl(_onnx_model_path.string(), _trt_model_dir.string(), _precision);
    }

    Buffer* getInputBuffer(const size_t index) override {
        if (index >= _input_names.size()) {
            throw std::out_of_range("Index out of range");
        }
        return _buffers[_input_names[index]];
    }

    Buffer* getOutputBuffer(const size_t index) override {
        if (index >= _output_names.size()) {
            throw std::out_of_range("Index out of range");
        }
        return _buffers[_output_names[index]];
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        _engine_name = getEngineName(_onnx_model_path.stem(), process_buffer_size);
        _engine_path = _trt_model_dir / _engine_name;
        if (n_input_channels > 1 || n_output_channels > 1) {
            spdlog::warn("{} can only process a single channel. n_input_channels and n_output_channels are overwritten by 1", _engine_name.string());
        }

        spdlog::debug("Attempting to load trt engine {}", _engine_name.string());
        if (!std::filesystem::exists(_engine_path)) {
            spdlog::info("No TRT engine found at path: {}. Regenerating ... this could take a while.", (_engine_path).c_str());
            if (!buildTRTModel(Dims2(1, process_buffer_size))) {
                throw std::runtime_error("Failed to build TRT model");
            }
        }
    }

    void allocate() override {
        if (!loadTRTModel()) {
            throw std::runtime_error("Failed to load TRT model");
        }
        // Ensure all dynamic bindings have been defined.
        if (!_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }
        for (auto& [name, specs] : _buffer_specs) {
            auto buffer = Buffer::create(specs);
            _context->setTensorAddress(name.c_str(), buffer->getDataMod());
            _buffers[name] = buffer;
        }
    }

    void setup(cudaStream_t stream) override {
        for (auto& [name, buffer] : _buffers) {
            buffer->clear(stream);
        }
    }

    void inference(cudaStream_t stream) override {
        if (!_context->enqueueV3(stream)) {
            throw std::runtime_error("Failed to inference trt engine" + _engine_path.string());
        }
    }

    void deallocate() override {
        for (auto& [name, buffer] : _buffers) {
            delete buffer;
        }
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
        _context.release();
        _engine.release();
        _runtime.release();
    }
};

TrtEngine* TrtEngine::create(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision) {
    return new TrtEngineImpl(onnx_model_path, trt_model_dir, precision);
}