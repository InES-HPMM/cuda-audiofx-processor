

#include <NvOnnxParser.h>

#include <fstream>
#include <gpu.cuh>
#include <opencv2/core/cuda.hpp>

#include "onnx_parser.cuh"
using namespace nvinfer1;

bool createTrtEngineFile(std::filesystem::path onnx_model_path, std::filesystem::path engine_path, Dims opt_in_shape, Dims min_in_shape, Dims max_in_shape) {
    if (!std::filesystem::exists(onnx_model_path)) {
        spdlog::error("Could not find model at path: {}", engine_path.string());
        return false;
    }

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nv_infer_logger));
    if (!builder) {
        spdlog::error("Create engine builder failed");
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        spdlog::error("Network creation failed");
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, nv_infer_logger));
    if (!parser) {
        spdlog::error("Parser creation failed");
        return false;
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parseFromFile(onnx_model_path.c_str(), static_cast<int32_t>(ILogger::Severity::kVERBOSE));
    if (!parsed) {
        spdlog::error("Parsing failed");
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    const auto numOutputs = network->getNbOutputs();
    if (numInputs != 1 || numOutputs != 1) {
        spdlog::error("Model should have exactly 1 input and 1 output but has {} inputs and {} outputs!", numInputs, numOutputs);
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
        spdlog::warn("GPU doesn't support Tf32 precision");
    }
    config->setFlag(BuilderFlag::kTF32);

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    gpuErrChk(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    spdlog::info("Building {}", engine_path.filename().string());
    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        spdlog::error("Failed to build TRT engine");
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(engine_path.string(), std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    spdlog::info("Success, saved engine to {}", engine_path.string());

    gpuErrChk(cudaStreamDestroy(profileStream));
    return true;
}
