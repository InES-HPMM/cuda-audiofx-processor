
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <gpu_fx.cuh>
#include <gpu_fx_eval.cuh>
#include <log.hpp>
#include <path.hpp>
#include <rmsd.cuh>
#include <signal.hpp>
#include <vector>

void testStream(std::vector<PerformanceMeasurement*>& measurements, IGpuFx* fx, size_t _process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    IGpuFxEvaluator* stream_test = IGpuFxEvaluator::createStreamEval(fx);
    measurements.push_back(stream_test->measurePerformance(n_warmup, n_measure, _process_buffer_size, simulate_buffer_intervals, process_in_place));
    delete stream_test;
}

void testGraph(std::vector<PerformanceMeasurement*>& measurements, IGpuFx* fx, size_t _process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    IGpuFxEvaluator* graph_test = IGpuFxEvaluator::createGraphEval(fx);
    measurements.push_back(graph_test->measurePerformance(n_warmup, n_measure, _process_buffer_size, simulate_buffer_intervals, process_in_place));
    delete graph_test;
}

void test(std::vector<PerformanceMeasurement*>& measurements, IGpuFx* fx, size_t _process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    testStream(measurements, fx->clone(), _process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    testGraph(measurements, fx, _process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
}

void stream_vs_graph(size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    std::vector<PerformanceMeasurement*> measurements;
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-1c.wav"));
    test(measurements, IGpuFx::createConv2i2(IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-2c.wav")), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    test(measurements, IGpuFx::createConv2i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    test(measurements, IGpuFx::createConv1i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    test(measurements, IGpuFx::createBiquadEQ(IBiquadParam::create(BiquadType::PEAK, 1000, 21, 0.6)), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    test(measurements, IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, 48), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    test(measurements, IGpuFx::createGate(0.2, 100, 5, 50), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    PerformanceMeasurement::writeStatisticsToCsv(path::out("stream-vs-graph-f" + std::to_string(process_buffer_size) + "-nw" + std::to_string(n_warmup) + "-nm" + std::to_string(n_measure) + "-ns" + std::to_string(process_buffer_size) + "-rt" + std::to_string(simulate_buffer_intervals) + "-ip" + std::to_string(process_in_place) + ".csv"), measurements);
    delete ir_pcm;
}
void stream_all(size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    std::vector<PerformanceMeasurement*> measurements;
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-1c.wav"));
    // testStream(measurements, IGpuFx::createConv2i2(IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-2c.wav")), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testStream(measurements, IGpuFx::createConv2i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testStream(measurements, IGpuFx::createConv1i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testStream(measurements, IGpuFx::createBiquadEQ(IBiquadParam::create(BiquadType::PEAK, 1000, 21, 0.6)), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testStream(measurements, IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, 48), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    testStream(measurements, IGpuFx::createGate(0.2, 100, 5, 50), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    PerformanceMeasurement::writeStatisticsToCsv(path::out("fx-stream-f" + std::to_string(process_buffer_size) + "-nw" + std::to_string(n_warmup) + "-nm" + std::to_string(n_measure) + "-ns" + std::to_string(process_buffer_size) + "-rt" + std::to_string(simulate_buffer_intervals) + "-ip" + std::to_string(process_in_place) + ".csv"), measurements);
    delete ir_pcm;
}

void graph_all(size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, bool process_in_place) {
    std::vector<PerformanceMeasurement*> measurements;
    IPCMSignal* ir_pcm = IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-1c.wav"));
    // testGraph(measurements, IGpuFx::createConv2i2(IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-2c.wav")), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testGraph(measurements, IGpuFx::createConv2i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testGraph(measurements, IGpuFx::createConv1i1(ir_pcm->clone(), 1 << 17, 0, true), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testGraph(measurements, IGpuFx::createBiquadEQ(IBiquadParam::create(BiquadType::PEAK, 1000, 21, 0.6)), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    // testGraph(measurements, IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, 48), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    testGraph(measurements, IGpuFx::createGate(0.2, 100, 5, 50), process_buffer_size, n_warmup, n_measure, simulate_buffer_intervals, process_in_place);
    PerformanceMeasurement::writeStatisticsToCsv(path::out("fx-graph-f" + std::to_string(process_buffer_size) + "-nw" + std::to_string(n_warmup) + "-nm" + std::to_string(n_measure) + "-ns" + std::to_string(process_buffer_size) + "-rt" + std::to_string(simulate_buffer_intervals) + "-ip" + std::to_string(process_in_place) + ".csv"), measurements);
    delete ir_pcm;
}

int main() {
    size_t process_buffer_size = 128;
    size_t n_warmup = 1000;
    size_t n_measure = 10000;

    // stream_vs_graph(process_buffer_size, n_warmup, n_measure, false, true);
    // stream_vs_graph(process_buffer_size, n_warmup, n_measure, true, true);
    // stream_all(process_buffer_size, n_warmup, n_measure, false, true);
    // graph_all(process_buffer_size, n_warmup, n_measure, false, true);
    graph_all(process_buffer_size, n_warmup, n_measure, true, true);

    return 0;
}