#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cuda_ext.cuh>
#include <gpu_fx.cuh>
#include <gpu_fx_eval.cuh>
#include <gpu_signal_graph.cuh>
#include <gpu_signal_graph_eval.cuh>
#include <iostream>
#include <iterator>
#include <log.hpp>
#include <path.hpp>
#include <random>
#include <rmsd.cuh>
#include <signal.hpp>
#include <thread>
#include <vector>

bool ir_2i2_impulse_test() {
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-2c.wav"));
    IGpuFx* fx = IGpuFx::createConv2i2(ir_pcm, 1 << 16, 0, true);

    float* impulse = new float[ir_pcm->getSampleCount() * 2];
    memset(impulse, 0, ir_pcm->getSampleCount() * 2 * sizeof(float));
    impulse[0] = 1.0;
    impulse[1] = 1.0;
    IFPSignal* impulse_signal = IFPSignal::fromBuffer(impulse, ir_pcm->getFrameCount(), 2, ir_pcm->getSampleRate(), ChannelOrder::INTERLEAVED);

    size_t buffer_size = 48;
    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    bool res = gpu_fx_test->testAccuracy(impulse_signal, ir_pcm->toFPSignal(ChannelOrder::INTERLEAVED), buffer_size, false, 6.4e-4, 0, true);
    delete gpu_fx_test;
    delete impulse_signal;
    return res;
}

bool ir_2i1_impulse_test() {
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"));
    IFPSignal* ir_float = IFPSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-2c.wav"), ChannelOrder::INTERLEAVED);
    IGpuFx* fx = IGpuFx::createConv2i1(ir_pcm, ir_pcm->getSampleCount(), 0, true);

    float* impulse = new float[ir_float->getSampleCount()];
    memset(impulse, 0, ir_float->getSampleCount() * sizeof(float));
    impulse[0] = 1.0;
    impulse[1] = 1.0;
    IFPSignal* impulse_signal = IFPSignal::fromBuffer(impulse, ir_float->getFrameCount(), ir_float->getChannelCount(), ir_float->getSampleRate(), ChannelOrder::INTERLEAVED);

    size_t buffer_size = 48;
    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    bool res = gpu_fx_test->testAccuracy(impulse_signal, ir_float, buffer_size, false, 1.3e-8, 0, true);
    delete gpu_fx_test;
    delete ir_float;
    delete impulse_signal;
    return res;
}

bool ir_1i1_impulse_test() {
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"));
    IFPSignal* ir_float = IFPSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"), ChannelOrder::INTERLEAVED);
    IGpuFx* fx = IGpuFx::createConv1i1(ir_pcm, ir_pcm->getSampleCount(), 0, true);

    float* impulse = new float[ir_pcm->getSampleCount()];
    memset(impulse, 0, ir_pcm->getSampleCount() * sizeof(float));
    impulse[0] = 1.0;
    IFPSignal* impulse_signal = IFPSignal::fromBuffer(impulse, ir_pcm->getFrameCount(), 1, ir_pcm->getSampleRate(), ChannelOrder::INTERLEAVED);

    size_t buffer_size = 48;
    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    bool res = gpu_fx_test->testAccuracy(impulse_signal, ir_float, buffer_size, false, 1.2e-8, 0, true);
    delete gpu_fx_test;
    delete ir_float;
    delete impulse_signal;
    return res;
}

bool trt_engine_test() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("git-48k-24b-1c.wav"), ChannelOrder::INTERLEAVED);
    IFPSignal* wet = IFPSignal::readFromFile(path::res("git-pedal-amp-48k-24b-1c.wav"), ChannelOrder::INTERLEAVED);
    IGpuFx* fx = IGpuFx::createNam("/home/nvidia/git/mt/res/models/nam_convnet_pedal_amp.onnx", "/home/nvidia/git/mt/out", TrtEnginePrecision::FP32, buffer_size);

    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    return gpu_fx_test->testAccuracy(dry, wet, buffer_size, false, 1.5e-3, 0, true);
}

// Test files are generated with Reaper Sine Sweep Generator and ReaEQ. ReaEQ probably uses slightly different filter coefficients than the ones used in the model, so a perfect match is not expected.
bool testBiquadEQ() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-2c.wav"), ChannelOrder::INTERLEAVED);
    IFPSignal* wet = IFPSignal::readFromFile(path::res("sine-sweep-p-f1k-g15-q2-48k-24b-2c.wav"), ChannelOrder::INTERLEAVED);
    IGpuFx* fx = IGpuFx::createBiquadEQ({IBiquadParam::create(BiquadType::PEAK, 1000, 21, 0.6)}, dry->getChannelCount());

    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    return gpu_fx_test->testAccuracy(dry, wet, buffer_size, false, 5.4e-2, 0, true);
}

bool testGate() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("git-48k-24b-1c.wav"), ChannelOrder::INTERLEAVED);
    IFPSignal* wet = IFPSignal::readFromFile(path::res("git-48k-24b-1c-gate-0.2t-100a-5r-50h.wav"), ChannelOrder::INTERLEAVED);
    IGpuFx* fx = IGpuFx::createGate(0.2, 100, 5, 50);

    IGpuFxEvaluator* gpu_fx_test = IGpuFxEvaluator::createGraphEval(fx);
    return gpu_fx_test->testAccuracy(dry, wet, buffer_size, false, 4.83e-8, 0, true);
}

bool testIOMap() {
    size_t n_frames = 4;
    size_t n_channels = 8;
    std::vector<size_t> input_mapping = {1, 0, 2, 2, 4, 5};
    std::vector<size_t> output_mapping = {0, 0, 3, 1, 5};

    float** input = new float*[n_channels];
    for (size_t c = 0; c < n_channels; c++) {
        input[c] = new float[n_frames];
        for (size_t s = 0; s < n_frames; s++) {
            input[c][s] = c;
        }
    }
    float** output = new float*[output_mapping.size()];
    float** expected_output = new float*[output_mapping.size()];
    for (size_t c = 0; c < output_mapping.size(); c++) {
        auto out_index = output_mapping[c];
        auto in_index = input_mapping[out_index];
        expected_output[c] = new float[n_frames];
        output[c] = new float[n_frames];
        for (size_t s = 0; s < n_frames; s++) {
            expected_output[c][s] = input[in_index][s];
        }
    }
    IFPSignal* input_signal = IFPSignal::fromBuffer(input, n_frames, n_channels, SampleRate::SR_48000, ChannelOrder::PLANAR);

    float* expected_output_interleaved = new float[n_frames * input_mapping.size()];
    float* output_interleaved = new float[n_frames * input_mapping.size()];
    float* output_dev;
    gpuErrChk(cudaMalloc(&output_dev, n_frames * input_mapping.size() * sizeof(float)));
    for (size_t s = 0; s < n_frames; s++) {
        for (size_t c = 0; c < input_mapping.size(); c++) {
            expected_output_interleaved[s * input_mapping.size() + c] = input[input_mapping[c]][s];
        }
    }
    cudaStream_t stream;
    gpuErrChk(cudaStreamCreate(&stream));
    IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Segmented2Interleaved, output_dev, input, sizeof(float), n_frames, input_mapping.size(), input_mapping, cudaMemcpyHostToDevice, stream);
    gpuErrChk(cudaMemcpyAsync(output_interleaved, output_dev, n_frames * input_mapping.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpuErrChk(cudaStreamSynchronize(stream));
    bool valid = true;
    for (size_t s = 0; s < input_mapping.size() * n_frames; s++) {
        if (output_interleaved[s] != expected_output_interleaved[s]) {
            spdlog::error("output[{}] = {} expected: {}", s, output_interleaved[s], expected_output_interleaved[s]);
            valid = false;
        }
    }
    IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Interleaved2Segmented, output, output_dev, sizeof(float), n_frames, input_mapping.size(), output_mapping, cudaMemcpyDeviceToHost, stream);
    gpuErrChk(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < output_mapping.size(); c++) {
        for (size_t s = 0; s < n_frames; s++) {
            if (output[c][s] != expected_output[c][s]) {
                spdlog::error("output[{}][{}] = {} expected: {}", c, s, output[c][s], expected_output[c][s]);
                valid = false;
            }
        }
    }

    if (valid) {
        spdlog::info("InputMap Test Passed");
    } else {
        spdlog::error("InputMap Test Failed");
    }

    return valid;
}

bool testLinear1i1SignalGraph() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-1c.wav"), ChannelOrder::PLANAR);
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));
    graph->add(IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 100, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 500, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 2000, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 4000, -21, 2),
    }));
    graph->add(IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 100, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 500, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 2000, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 4000, 21, 2),
    }));
    graph->add(IGpuFx::createOutputMap({0}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "linear-1i1-signal-graph");
    bool res = evaluator->testAccuracy(dry, dry, buffer_size, 7.7e-5, 0, true);
    delete evaluator;
    delete dry;
    return res;
}

bool testLinear2i2SignalGraph() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-2c.wav"), ChannelOrder::PLANAR);
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0, 1}));

    graph->add(IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 100, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 500, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 2000, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 4000, -21, 2),
    }));
    graph->add(IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 100, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 500, 21, 2),
        IBiquadParam::create(BiquadType::PEAK, 2000, -21, 2),
        IBiquadParam::create(BiquadType::PEAK, 4000, 21, 2),
    }));
    graph->add(IGpuFx::createOutputMap({0, 1}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "linear-2i2-signal-graph");
    bool res = evaluator->testAccuracy(dry, dry, buffer_size, 7.7e-5, 0, true);
    delete evaluator;
    delete dry;
    return res;
}

bool testParallel1i1SignalGraph() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-1c.wav"), ChannelOrder::PLANAR);
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));

    auto eqs = graph->split({IGpuFx::createBiquadEQ({
                                 IBiquadParam::create(BiquadType::PEAK, 500, 5, 1),
                             }),
                             IGpuFx::createBiquadEQ({
                                 IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                             })},
                            input_map);
    auto mix = graph->merge(IGpuFx::createMixSegment(2, 1), eqs);
    graph->add(IGpuFx::createOutputMap({0}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "parallel-1i1-signal-graph");
    bool res = evaluator->testAccuracy(dry, dry, buffer_size, 3.2e-4, 0, true);
    delete evaluator;
    delete dry;
    return res;
}

bool testParallel2i2SignalGraph() {
    size_t buffer_size = 48;
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-2c.wav"), ChannelOrder::PLANAR);
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0, 1}));

    auto eqs = graph->split({IGpuFx::createBiquadEQ({
                                 IBiquadParam::create(BiquadType::PEAK, 500, 5, 1),
                             }),
                             IGpuFx::createBiquadEQ({
                                 IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                             })},
                            input_map);
    auto mix = graph->merge(IGpuFx::createMixSegment(4, 2), eqs);
    graph->add(IGpuFx::createOutputMap({0, 1}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "parallel-2i2-signal-graph");
    bool res = evaluator->testAccuracy(dry, dry, buffer_size, 3.2e-4, 0, true);
    delete evaluator;
    delete dry;
    return res;
}

// int minimalRMSDStressTest() {
//     size_t n_samples = 2000000;
//     std::random_device rnd_device;
//     std::default_random_engine engine;
//     std::uniform_real_distribution<float> dist(-1.0, 1.0);

//     auto gen = [&]() {
//         return dist(engine);
//     };

//     std::vector<float> vec1(n_samples);
//     std::vector<float> vec2(n_samples);
//     std::generate(vec1.begin(), vec1.end(), gen);

//     int minimal_rmsd_offset;
//     float rmsd = getMinimalRMSD(vec1.data(), vec2.data(), n_samples, 5, &minimal_rmsd_offset);
//     return 0;
// }

int main() {
    if (!testMinimalRMSDIntegrity() ||
        !ir_2i2_impulse_test() ||
        !ir_2i1_impulse_test() ||
        !ir_1i1_impulse_test() ||
        !trt_engine_test() ||
        !testGate() ||
        !testBiquadEQ() ||
        !testIOMap() ||
        !testLinear1i1SignalGraph() ||
        !testLinear2i2SignalGraph() ||
        !testParallel1i1SignalGraph() ||
        !testParallel2i2SignalGraph()) {
        throw std::runtime_error("System Integrity Compromised");
    }
    // ir_2i2_impulse_test();
    // ir_2i1_impulse_test();
    // ir_1i1_impulse_test();
    // trt_engine_test();
    // testGate();
    // testBiquadEQ();
    // testIOMap();
    // testLinear1i1SignalGraph();
    // testLinear2i2SignalGraph();
    // testParallel1i1SignalGraph();
    // testParallel2i2SignalGraph();

    return 0;
}
