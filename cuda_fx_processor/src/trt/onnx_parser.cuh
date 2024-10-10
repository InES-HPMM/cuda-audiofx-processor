#pragma once

#include <filesystem>

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
static NvInverLogger nv_infer_logger;

bool createTrtEngineFile(std::filesystem::path onnx_model_path, std::filesystem::path engine_path, Dims opt_in_shape, Dims min_in_shape = Dims{}, Dims max_in_shape = Dims{});