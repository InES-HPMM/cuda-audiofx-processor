
#include <NvOnnxParser.h>

#include <filesystem>
#include <fstream>
#include <gpu.cuh>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

#include "onnx_parser.cuh"
#include "trt_engine.cuh"
using namespace nvinfer1;

class TrtEngineImpl : public TrtEngine {
   private:
    TrtEnginePrecision _precision;
    std::filesystem::path _onnx_model_path;
    std::filesystem::path _trt_model_dir;
    std::filesystem::path _engine_name;
    std::filesystem::path _engine_path;
    const char* _model_input_name = nullptr;
    const char* _model_output_name = nullptr;
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
            spdlog::error("Unable to read engine file");
            return false;
        }
        file.close();

        // Create a runtime to deserialize the engine file.
        _runtime = std::unique_ptr<IRuntime>{createInferRuntime(nv_infer_logger)};
        if (!_runtime) {
            spdlog::error("Failed to create infer runtime");
            return false;
        }

        // Create an engine, a representation of the optimized model.
        _engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!_engine) {
            spdlog::error("Failed to create engine");
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

        spdlog::debug("Successfully loaded network {}", _engine_path.filename().string());
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

    void configure(size_t process_buffer_size, size_t receptive_field, size_t n_input_channels, size_t n_output_channels) override {
        _engine_name = getEngineName(_onnx_model_path.stem(), process_buffer_size);
        _engine_path = _trt_model_dir / _engine_name;
        if (n_input_channels > 1 || n_output_channels > 1) {
            spdlog::warn("{} can only process a single channel. n_input_channels and n_output_channels are overwritten by 1", _engine_name.string());
        }

        spdlog::debug("Attempting to load trt engine {}", _engine_name.string());
        if (!std::filesystem::exists(_engine_path)) {
            spdlog::info("No TRT engine found at path: {}. Regenerating ... this could take a while.", (_engine_path).c_str());
            if (!createTrtEngineFile(_onnx_model_path, _engine_path, Dims2(1, process_buffer_size + receptive_field))) {
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