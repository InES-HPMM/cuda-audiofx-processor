#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <random>

#include "NvOnnxParser.h"
#include "gpu.cuh"

using namespace nvinfer1;

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Options for the network
struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset
    // directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8
    // inference. Should be set to as large a batch number as your GPU will
    // support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // GPU device index
    int deviceIndex = 0;
    // Directory where the engine file should be saved
    std::string engineFileDir = "/home/nvidia/git/mt/res/models";
};

inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        // spdlog::error(errMsg);
        throw std::runtime_error(errMsg);
    }
}

inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath) {
    std::vector<std::string> fileNames;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().string());
        }
    }
    return fileNames;
}

template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch {
    typename Clock::time_point start_point;

   public:
    Stopwatch() : start_point(Clock::now()) {}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Engine {
   private:
    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    float *dev_input_buffer;
    float *dev_output_buffer;
    float *host_temp_buffer;
    size_t _inference_buffer_size;
    size_t _inference_chunk_size;
    size_t _inference_receptive_field;
    // std::vector<uint32_t> m_outputLengths{};
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
    std::string m_engineName;
    std::string m_enginePath;

    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
        const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
        std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

        // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
        std::vector<std::string> deviceNames;
        getDeviceNames(deviceNames);

        if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
            throw std::runtime_error("Error, provided device index is out of range!");
        }

        auto deviceName = deviceNames[options.deviceIndex];
        // Remove spaces from the device name
        deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

        engineName += "." + deviceName;

        // Serialize the specified options into the filename
        if (options.precision == Precision::FP16) {
            engineName += ".fp16";
        } else if (options.precision == Precision::FP32) {
            engineName += ".fp32";
        } else {
            engineName += ".int8";
        }

        engineName += "." + std::to_string(_inference_buffer_size);

        return engineName;
    }

    void getDeviceNames(std::vector<std::string> &deviceNames) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);

        for (int device = 0; device < numGPUs; device++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            deviceNames.push_back(std::string(prop.name));
        }
    }

    void clearGpuBuffers() {
        if (!m_buffers.empty()) {
            // Free GPU memory of outputs
            const auto numInputs = m_inputDims.size();
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
                gpuErrChk(cudaFree(m_buffers[outputBinding]));
            }
            m_buffers.clear();
        }
    }

   public:
    Engine(const Options &options, size_t inference_buffer_size, size_t inference_chunk_size, size_t inference_receptive_field)
        : m_options(options) {
        _inference_buffer_size = inference_buffer_size;
        _inference_chunk_size = inference_chunk_size;
        _inference_receptive_field = inference_receptive_field;
    }
    ~Engine() {
        // Free the GPU memory
        for (auto &buffer : m_buffers) {
            gpuErrChk(cudaFree(buffer));
        }

        m_buffers.clear();
    }

    size_t getInferenceBufferSize() const { return _inference_buffer_size; }

    // Build the network
    bool build(std::string onnxModelPath) {
        m_engineName = serializeEngineOptions(m_options, onnxModelPath);
        m_enginePath = m_options.engineFileDir + "/" + m_engineName;
        std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

        if (doesFileExist(m_enginePath)) {
            std::cout << "Engine found, not regenerating..." << std::endl;
            return true;
        }

        if (!doesFileExist(onnxModelPath)) {
            throw std::runtime_error("Could not find model at path: " + onnxModelPath);
        }

        // Was not able to find the engine file, generate...
        std::cout << "Engine not found, generating. This could take a while..." << std::endl;

        // Create our engine builder.
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
        if (!builder) {
            std::cout << "Create engine builder failed" << std::endl;
            return false;
        }

        // Define an explicit batch size and then create the network (implicit batch size is deprecated).
        // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
        auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cout << "Network creation failed" << std::endl;
            return false;
        }

        // Create a parser for reading the onnx file.
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
        if (!parser) {
            std::cout << "Parser creation failed" << std::endl;
            return false;
        }

        // Parse the buffer we read into memory.
        auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int32_t>(ILogger::Severity::kVERBOSE));
        if (!parsed) {
            std::cout << "Parsing failed" << std::endl;
            return false;
        }

        // Ensure that all the inputs have the same batch size
        const auto numInputs = network->getNbInputs();
        if (numInputs < 1) {
            throw std::runtime_error("Error, model needs at least 1 input!");
        }
        const auto input0Batch = network->getInput(0)->getDimensions().d[0];
        for (int32_t i = 1; i < numInputs; ++i) {
            if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
                throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
            }
        }

        // Check to see if the model supports dynamic batch size or not
        if (input0Batch == -1) {
            std::cout << "Model supports dynamic batch size" << std::endl;
        } else if (input0Batch == 1) {
            std::cout << "Model only supports fixed batch size of 1" << std::endl;
            // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
            if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
                throw std::runtime_error("Error, model only supports a fixed batch size of 1. Must set Options.optBatchSize and Options.maxBatchSize to 1");
            }
        } else {
            throw std::runtime_error("Implementation currently only supports dynamic batch sizes or a fixed batch size of 1 (your batch size is fixed to " + std::to_string(input0Batch) + ")");
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        // Register a single optimization profile
        IOptimizationProfile *optProfile = builder->createOptimizationProfile();
        for (int32_t i = 0; i < numInputs; ++i) {
            // Must specify dimensions for all the inputs the model expects.
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t inputW = inputDims.d[3];

            // Specify the optimization profile`
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims2(1, _inference_buffer_size));
            optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims2(m_options.optBatchSize, _inference_buffer_size));
            optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims2(m_options.maxBatchSize, _inference_buffer_size));
        }
        config->addOptimizationProfile(optProfile);

        // Ensure the GPU supports TF36 inference
        if (!builder->platformHasTf32()) {
            std::cout << "GPU doesn't support Tf32 precision" << std::endl;
        }
        config->setFlag(BuilderFlag::kTF32);

        // CUDA stream used for profiling by the builder.
        cudaStream_t profileStream;
        gpuErrChk(cudaStreamCreate(&profileStream));
        config->setProfileStream(profileStream);

        std::cout << "Building " << m_engineName << std::endl;
        // Build the engine
        // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
        // Doing so will provide you with more information on why exactly it is failing.
        std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            return false;
        }

        // Write the engine to disk
        std::ofstream outfile(m_enginePath, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        std::cout << "Success, saved engine to " << m_enginePath << std::endl;

        gpuErrChk(cudaStreamDestroy(profileStream));
        return true;
    }

    bool loadNetwork() {
        std::cout << "Loading network " << m_enginePath << std::endl;
        // Read the serialized model from disk
        std::ifstream file(m_enginePath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Unable to read engine file");
        }

        // Create a runtime to deserialize the engine file.
        m_runtime = std::unique_ptr<IRuntime>{createInferRuntime(m_logger)};
        if (!m_runtime) {
            return false;
        }

        // Set the device index
        auto ret = cudaSetDevice(m_options.deviceIndex);
        if (ret != 0) {
            int numGPUs;
            cudaGetDeviceCount(&numGPUs);
            auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                          ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
            throw std::runtime_error(errMsg);
        }

        // Create an engine, a representation of the optimized model.
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!m_engine) {
            return false;
        }

        // The execution context contains all of the state associated with a particular invocation
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context) {
            return false;
        }

        gpuErrChk(cudaMalloc(&dev_input_buffer, _inference_buffer_size * sizeof(float)));
        gpuErrChk(cudaMemset(dev_input_buffer, 0, _inference_buffer_size * sizeof(float)));

        gpuErrChk(cudaMalloc(&dev_output_buffer, _inference_chunk_size * sizeof(float)));
        gpuErrChk(cudaMemset(dev_output_buffer, 0, _inference_chunk_size * sizeof(float)));

        host_temp_buffer = (float *)malloc(_inference_chunk_size * sizeof(float));

        std::cout << "Successfully loaded network " << m_engineName << std::endl;
        return true;
    }

    bool runInference(float *inputs, float *outputs, int inference_index) {
        // std::cout << "Attempting inference of network " << m_engineName << std::endl;
        // Create the cuda stream that will be used for inference
        cudaStream_t inferenceCudaStream;
        gpuErrChk(cudaStreamCreate(&inferenceCudaStream));

        size_t dev_buf_offset = _inference_receptive_field - std ::min(_inference_receptive_field, inference_index * _inference_chunk_size);
        size_t host_buf_offset = std::min(_inference_receptive_field, inference_index * _inference_chunk_size);
        // std::cout << "Inference index: " << inference_index << " dev_buf_offset: " << dev_buf_offset << " host_buf_offset: -" << host_buf_offset << std::endl;

        gpuErrChk(cudaMemcpyAsync(dev_input_buffer + dev_buf_offset, inputs - host_buf_offset, (_inference_buffer_size - dev_buf_offset) * sizeof(float), cudaMemcpyHostToDevice, inferenceCudaStream));
        // gpuErrChk(cudaMemcpyAsync(dev_input_buffer, inputs, _inference_chunk_size * sizeof(float), cudaMemcpyHostToDevice, inferenceCudaStream));
        m_context->setTensorAddress("l_x_", dev_input_buffer);
        m_context->setTensorAddress("view", dev_output_buffer);

        // Ensure all dynamic bindings have been defined.
        if (!m_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }

        // Run inference.
        // void *bindings[] = {dev_input_buffer, dev_output_buffer};
        // bool status = m_context->enqueueV2(bindings, inferenceCudaStream, nullptr);
        bool status = m_context->enqueueV3(inferenceCudaStream);
        if (!status) {
            return false;
        }

        gpuErrChk(cudaMemcpyAsync(outputs, dev_output_buffer, _inference_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
        // gpuErrChk(cudaMemcpyAsync(host_temp_buffer, dev_output_buffer, _inference_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
        // Synchronize the cuda stream
        gpuErrChk(cudaStreamSynchronize(inferenceCudaStream));
        gpuErrChk(cudaStreamDestroy(inferenceCudaStream));

        // for (int i = 0; i < _inference_chunk_size; ++i) {
        //     // double multiplier = 0.5 * (1 - cos(2 * M_PI * i / 2047));
        //     // outputs[i] += host_temp_buffer[i] * multiplier;
        //     outputs[i] += host_temp_buffer[i];
        // }
        // std::cout << "Successfully inferenced network " << m_engineName << std::endl;
        return true;
    }
};

int run_nam_engine(float **output, float **input, int n_samples, int n_channels) {
    const std::string onnxModelPath = "/home/nvidia/git/mt/res/models/nam_convnet.onnx";

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP32;
    // If using INT8 precision, must specify path to directory containing calibration data.
    options.calibrationDataDirectoryPath = "";
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    // size_t inference_receptive_field = 0;
    size_t inference_receptive_field = 8190;
    size_t inference_chunk_size = n_samples;
    size_t inference_buffer_size = inference_chunk_size + inference_receptive_field;
    Engine engine(options, inference_buffer_size, inference_chunk_size, inference_receptive_field);

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(onnxModelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // size_t hop_size = inference_chunk_size / 4;

    for (int s = 0, i = 0; s + inference_chunk_size <= n_samples; s += inference_chunk_size, i++) {
        succ = engine.runInference(input[0] + s, output[0] + s, i);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
    }
    for (int i = 0; i < n_samples; ++i) {
        // output[0][i] *= 0.5;
        output[1][i] = output[0][i];
    }
    // memcpy(output[1], output[0], n_samples * sizeof(float));

    return 0;
}