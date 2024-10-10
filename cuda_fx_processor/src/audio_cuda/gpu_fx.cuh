#pragma once

#include <cuda_runtime.h>

#include <buffer.cuh>
#include <gpu.cuh>
#include <signal.hpp>
#include <string>

#define MATH_PI 3.14159265358979323846

class IBiquadParam {
   public:
    static IBiquadParam* create(BiquadType type, double frequency, double gain, double quality, SampleRate sample_rate = SampleRate::SR_48000);

    virtual ~IBiquadParam() {};

    virtual void setFrequency(double frequency) = 0;
    virtual void setGain(double gain) = 0;
    virtual void setQuality(double quality) = 0;
    virtual IBiquadParam* clone() = 0;
};

class IGpuFx {
   public:
    static IGpuFx* createConv1i1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale = -18, bool force_wet_mix = false);
    static IGpuFx* createConv2i1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale = -18, bool force_wet_mix = false);
    static IGpuFx* createConv2i2(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale = -18, bool force_wet_mix = false);
    static IGpuFx* createPassThrough();
    static IGpuFx* createBiquadEQ(IBiquadParam* param, size_t n_channels = 1);
    static IGpuFx* createBiquadEQ(std::vector<IBiquadParam*> params, size_t n_channels = 1);
    static IGpuFx* createNam(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision, size_t buf_size);
    static IGpuFx* createMixSegment(size_t n_in_channels, size_t n_out_channels);
    static IGpuFx* createMixInterleaved(size_t n_in_channels, size_t n_out_channels);
    static IGpuFx* createInputMap(std::vector<size_t> input_mapping);
    static IGpuFx* createOutputMap(std::vector<size_t> output_mapping);
    static IGpuFx* createGate(float threshold, float attack_time_ms, float release_time_ms, float hold_time_ms, SampleRate sample_rate = SampleRate::SR_48000);

    virtual ~IGpuFx() {};

    virtual std::string getName() = 0;
    virtual size_t getInChannelCount() = 0;
    virtual size_t getOutChannelCount() = 0;
    virtual size_t getProcSampleCount() = 0;
    virtual size_t getOutSampleCount() = 0;

    virtual BufferRackSpecs getInputSpecs() = 0;
    virtual BufferRackSpecs getOutputSpecs() = 0;

    virtual void configure(size_t n_proc_frames, size_t n_in_channels = 0, size_t n_out_channels = 0) = 0;
    virtual cudaGraph_t recordSetupGraph() = 0;

    virtual cudaGraph_t recordProcessGraph(const BufferRack* dest, const BufferRack* src) = 0;

    virtual cudaGraph_t recordPostProcessGraph() = 0;

    virtual cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) = 0;
    virtual cudaStream_t process(cudaStream_t stream, const BufferRack* dest, const BufferRack* src, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) = 0;
    virtual cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) = 0;
    virtual void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) = 0;
    virtual void teardown() = 0;
    virtual IGpuFx* clone() = 0;
};
