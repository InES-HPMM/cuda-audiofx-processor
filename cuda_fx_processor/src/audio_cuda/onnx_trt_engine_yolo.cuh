#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
    std::string engineFileDir = ".";
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

static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width, const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0)) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

// Convert NHWC to NCHW and apply scaling and mean subtraction
static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                        const std::array<float, 3> &divVals, bool normalize, bool swapRB = false) {
    if (batchInput.empty() || batchInput[0].channels() != 3) {
        throw std::runtime_error("Input must be a batch of images with 3 channels!");
    }

    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;
    if (swapRB) {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels);  // HWC -> CHW
        }
    } else {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels);  // HWC -> CHW
        }
    }
    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
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

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
   public:
    Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH, const std::string &calibDataDirPath,
                           const std::string &calibTableName, const std::string &inputBlobName,
                           const std::array<float, 3> &subVals = {0.f, 0.f, 0.f}, const std::array<float, 3> &divVals = {1.f, 1.f, 1.f},
                           bool normalize = true, bool readCache = true)
        : m_batchSize(batchSize), m_inputW(inputW), m_inputH(inputH), m_imgIdx(0), m_calibTableName(calibTableName), m_inputBlobName(inputBlobName), m_subVals(subVals), m_divVals(divVals), m_normalize(normalize), m_readCache(readCache) {
        // Allocate GPU memory to hold the entire batch
        m_inputCount = 3 * inputW * inputH * batchSize;
        checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

        // Read the name of all the files in the specified directory.
        if (!doesFileExist(calibDataDirPath)) {
            throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
        }

        m_imgPaths = getFilesInDirectory(calibDataDirPath);
        if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
            throw std::runtime_error("There are fewer calibration images than the specified batch size!");
        }

        // Randomize the calibration data
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
    }

    virtual ~Int8EntropyCalibrator2() {
        checkCudaErrorCode(cudaFree(m_deviceInput));
    };

    // Abstract base class methods which must be implemented
    int32_t getBatchSize() const noexcept {
        // Return the batch size
        return m_batchSize;
    }

    bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept {
        // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

        if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
            // There are not enough images left to satisfy an entire batch
            return false;
        }

        // Read the calibration images into memory for the current batch
        std::vector<cv::cuda::GpuMat> inputImgs;
        for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
            std::cout << "Reading image " << i << ": " << m_imgPaths[i] << std::endl;
            auto cpuImg = cv::imread(m_imgPaths[i]);
            if (cpuImg.empty()) {
                std::cout << "Fatal error: Unable to read image at path: " << m_imgPaths[i] << std::endl;
                return false;
            }

            cv::cuda::GpuMat gpuImg;
            gpuImg.upload(cpuImg);
            cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

            // TODO: Define any preprocessing code here, such as resizing
            auto resized = resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

            inputImgs.emplace_back(std::move(resized));
        }

        // Convert the batch from NHWC to NCHW
        // ALso apply normalization, scaling, and mean subtraction
        auto mfloat = blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
        auto *dataPointer = mfloat.ptr<void>();

        // Copy the GPU buffer to member variable so that it persists
        checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

        m_imgIdx += m_batchSize;
        if (std::string(names[0]) != m_inputBlobName) {
            std::cout << "Error: Incorrect input name provided!" << std::endl;
            return false;
        }
        bindings[0] = m_deviceInput;
        return true;
    }

    void const *readCalibrationCache(std::size_t &length) noexcept {
        std::cout << "Searching for calibration cache: " << m_calibTableName << std::endl;
        m_calibCache.clear();
        std::ifstream input(m_calibTableName, std::ios::binary);
        input >> std::noskipws;
        if (m_readCache && input.good()) {
            std::cout << "Reading calibration cache: " << m_calibTableName << std::endl;
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
        }
        length = m_calibCache.size();
        return length ? m_calibCache.data() : nullptr;
    }

    void writeCalibrationCache(void const *ptr, std::size_t length) noexcept {
        std::cout << "Writing calib cache: " << m_calibTableName << " Size: " << length << " bytes" << std::endl;
        std::ofstream output(m_calibTableName, std::ios::binary);
        output.write(reinterpret_cast<const char *>(ptr), length);
    }

   private:
    const int32_t m_batchSize;
    const int32_t m_inputW;
    const int32_t m_inputH;
    int32_t m_imgIdx;
    std::vector<std::string> m_imgPaths;
    size_t m_inputCount;
    const std::string m_calibTableName;
    const std::string m_inputBlobName;
    const std::array<float, 3> m_subVals;
    const std::array<float, 3> m_divVals;
    const bool m_normalize;
    const bool m_readCache;
    void *m_deviceInput;
    std::vector<char> m_calibCache;
};

class Engine {
   public:
    Engine(const Options &options)
        : m_options(options) {}
    ~Engine() {
        // Free the GPU memory
        for (auto &buffer : m_buffers) {
            checkCudaErrorCode(cudaFree(buffer));
        }

        m_buffers.clear();
    }

    // Build the network
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).
    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool build(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
               const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) {
        m_engineName = serializeEngineOptions(m_options, onnxModelPath);
        std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

        if (doesFileExist(m_engineName)) {
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
            return false;
        }

        // Define an explicit batch size and then create the network (implicit batch size is deprecated).
        // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
        auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            return false;
        }

        // Create a parser for reading the onnx file.
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
        if (!parser) {
            return false;
        }

        // We are going to first read the onnx file into memory, then pass that buffer to the parser.
        // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
        std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Unable to read engine file");
        }

        // Parse the buffer we read into memory.
        auto parsed = parser->parse(buffer.data(), buffer.size());
        if (!parsed) {
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
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        }
        config->addOptimizationProfile(optProfile);

        // Set the precision level
        if (m_options.precision == Precision::FP16) {
            // Ensure the GPU supports FP16 inference
            if (!builder->platformHasFastFp16()) {
                throw std::runtime_error("Error: GPU does not support FP16 precision");
            }
            config->setFlag(BuilderFlag::kFP16);
        } else if (m_options.precision == Precision::INT8) {
            if (numInputs > 1) {
                throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
            }

            // Ensure the GPU supports INT8 Quantization
            if (!builder->platformHasFastInt8()) {
                throw std::runtime_error("Error: GPU does not support INT8 precision");
            }

            // Ensure the user has provided path to calibration data directory
            if (m_options.calibrationDataDirectoryPath.empty()) {
                throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
            }

            config->setFlag((BuilderFlag::kINT8));

            const auto input = network->getInput(0);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            const auto calibrationFileName = m_engineName + ".calibration";

            m_calibrator = std::make_unique<Int8EntropyCalibrator2>(m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2], m_options.calibrationDataDirectoryPath,
                                                                    calibrationFileName, inputName, subVals, divVals, normalize);
            config->setInt8Calibrator(m_calibrator.get());
        }

        // CUDA stream used for profiling by the builder.
        cudaStream_t profileStream;
        checkCudaErrorCode(cudaStreamCreate(&profileStream));
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
        std::ofstream outfile(m_engineName, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        std::cout << "Success, saved engine to " << m_engineName << std::endl;

        checkCudaErrorCode(cudaStreamDestroy(profileStream));
        return true;
    }

    bool loadNetwork() {
        // Read the serialized model from disk
        std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
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

        // Storage for holding the input and output buffers
        // This will be passed to TensorRT for inference
        // TODO: rollback libnvinfer API to TensorRT 8.2.1.9
        // TODO: Needs a way to dynamically get input and output buffer size
        int32_t nb_io_tensors = 2;
        m_buffers.resize(nb_io_tensors);
        // Tensor shape is (bs, chan, height, width)
        nvinfer1::Dims4 inp_tensor_shape = {1, 3, 640, 640};
        nvinfer1::Dims3 out_tensor_shape = {1, 84, 8400};
        // Input and output tensor name
        m_IOTensorNames.emplace_back("images");
        m_IOTensorNames.emplace_back("output0");

        // Allocate GPU memory for input and output buffers
        // Create a cuda stream
        cudaStream_t stream;
        checkCudaErrorCode(cudaStreamCreate(&stream));
        // Define
        // Allocate input
        size_t input_mem_size = m_options.maxBatchSize *
                                inp_tensor_shape.d[1] * inp_tensor_shape.d[2] * inp_tensor_shape.d[3] * sizeof(float);
        // cudaMemcpyAsync
        checkCudaErrorCode(cudaMallocManaged(&m_buffers[0], input_mem_size));
        checkCudaErrorCode(cudaStreamAttachMemAsync(stream, m_buffers[0], 0, cudaMemAttachGlobal));
        m_inputDims.emplace_back(inp_tensor_shape.d[1], inp_tensor_shape.d[2], inp_tensor_shape.d[3]);
        // Allocate output
        m_outputLengthsFloat.clear();
        m_outputDims.push_back(out_tensor_shape);
        uint32_t outputLenFloat = 1;
        for (int j = 1; j < out_tensor_shape.nbDims; ++j) {
            // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
            outputLenFloat *= out_tensor_shape.d[j];
        }
        m_outputLengthsFloat.push_back(outputLenFloat);
        // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
        size_t output_mem_size = m_options.maxBatchSize * outputLenFloat * sizeof(float);
        checkCudaErrorCode(cudaMallocManaged(&m_buffers[1], output_mem_size));
        checkCudaErrorCode(cudaStreamAttachMemAsync(stream, m_buffers[1], 0, cudaMemAttachGlobal));

        // Synchronize and destroy the cuda stream
        checkCudaErrorCode(cudaStreamSynchronize(stream));
        checkCudaErrorCode(cudaStreamDestroy(stream));

        return true;
    }

    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<float>>> &featureVectors) {
        // First we do some error checking
        if (inputs.empty() || inputs[0].empty()) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Provided input vector is empty!" << std::endl;
            return false;
        }

        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Incorrect number of inputs provided!" << std::endl;
            return false;
        }

        // Ensure the batch size does not exceed the max
        if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size is larger than the model expects!" << std::endl;
            std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
            std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        // Make sure the same batch size was provided for all inputs
        for (size_t i = 1; i < inputs.size(); ++i) {
            if (inputs[i].size() != static_cast<size_t>(batchSize)) {
                std::cout << "===== Error =====" << std::endl;
                std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
                return false;
            }
        }

        // Create the cuda stream that will be used for inference
        cudaStream_t inferenceCudaStream;
        checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        // Preprocess all the inputs
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &batchInput = inputs[i];
            const auto &dims = m_inputDims[i];

            auto &input = batchInput[0];
            if (input.channels() != dims.d[0] ||
                input.rows != dims.d[1] ||
                input.cols != dims.d[2]) {
                std::cout << "===== Error =====" << std::endl;
                std::cout << "Input does not have correct size!" << std::endl;
                std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", "
                          << dims.d[2] << ")" << std::endl;
                std::cout << "Got: (" << input.channels() << ", " << input.rows << ", " << input.cols << ")" << std::endl;
                std::cout << "Ensure you resize your input image to the correct size" << std::endl;
                return false;
            }

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            int32_t input_idx = m_engine->getBindingIndex(m_IOTensorNames[i].c_str());
            m_context->setBindingDimensions(input_idx, inputDims);

            // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format.
            // The following method converts NHWC to NCHW.
            // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
            // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
            // Copy over the input data and perform the preprocessing
            auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
            auto *dataPointer = mfloat.ptr<void>();

            checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
                                               mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
                                               cudaMemcpyDeviceToDevice, inferenceCudaStream));
        }

        // Ensure all dynamic bindings have been defined.
        if (!m_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }

        // Run inference.
        void *bindings[] = {m_buffers[0], m_buffers[1]};
        bool status = m_context->enqueueV2(bindings, inferenceCudaStream, nullptr);
        if (!status) {
            return false;
        }

        // Copy the outputs back to CPU
        featureVectors.clear();

        for (int batch = 0; batch < batchSize; ++batch) {
            // Batch
            std::vector<std::vector<float>> batchOutputs{};
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
                // We start at index m_inputDims.size() to account for the inputs in our m_buffers
                std::vector<float> output;
                auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
                output.resize(outputLenFloat);
                // Copy the output
                checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
                batchOutputs.emplace_back(std::move(output));
            }
            featureVectors.emplace_back(std::move(batchOutputs));
        }

        // Synchronize the cuda stream
        checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
        return true;
    }

    // Utility method for resizing an image while maintaining the aspect ratio by
    // adding padding to smaller dimension after scaling While letterbox padding
    // normally adds padding to top & bottom, or left & right sides, this
    // implementation only adds padding to the right or bottom side This is done
    // so that it's easier to convert detected coordinates (ex. YOLO model) back
    // to the original reference frame.

    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple
    // output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output) {
        if (input.size() != 1) {
            throw std::logic_error("The feature vector has incorrect dimensions!");
        }

        output = std::move(input[0]);
    }

    // Utility method for transforming triple nested output array into single
    // array Should be used when the output batch size is 1, and there is only a
    // single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output) {
        if (input.size() != 1 || input[0].size() != 1) {
            throw std::logic_error("The feature vector has incorrect dimensions!");
        }

        output = std::move(input[0][0]);
    }

   private:
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

        engineName += "." + std::to_string(options.maxBatchSize);
        engineName += "." + std::to_string(options.optBatchSize);

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

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    // std::vector<uint32_t> m_outputLengths{};
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
    std::string m_engineName;
};

int run_yolo_engine() {
    const std::string onnxModelPath = "/home/nvidia/git/mt/res/models/yolov8n.onnx";
    const std::string trtModelPath = "/home/nvidia/git/mt/res/models/yolov8n.trt";

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing calibration data.
    options.calibrationDataDirectoryPath = "";
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    Engine engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(onnxModelPath, subVals, divVals, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Read the input image
    // TODO: You will need to read the input image required for your model
    const std::string inputImage = "/home/nvidia/git/mt/res/team.jpg";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + inputImage);
    }

    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects RGB input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // In the following section we populate the input vectors to later pass for inference
    const auto &inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the inputs
    // You should populate your inputs appropriately.
    for (const auto &inputDim : inputDims) {  // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) {  // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
            // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
            auto resized = resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
            //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Warm up the network before we begin the benchmark
    std::cout << "\nWarming up the network..." << std::endl;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
    }

    // Benchmark the inference time
    size_t numIterations = 1000;
    std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n"
              << std::endl;
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    std::cout << "Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Avg time per sample: " << std::endl;
    std::cout << avgElapsedTimeMs << " ms" << std::endl;
    std::cout << "Batch size: " << std::endl;
    std::cout << inputs[0].size() << std::endl;
    std::cout << "Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
    std::cout << "======================\n"
              << std::endl;

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e : featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n"
                      << std::endl;
        }
    }

    // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}