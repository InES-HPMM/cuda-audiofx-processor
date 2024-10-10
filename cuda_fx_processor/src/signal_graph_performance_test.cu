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
#include <numeric>
#include <path.hpp>
#include <random>
#include <rmsd.cuh>
#include <signal.hpp>
#include <thread>
#include <vector>

void mixConsole(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));

    std::vector<IGpuSignalVertex*> eq_vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->split({IGpuFx::createGate(0.2, 100, 5, 50)}, input_map)[0];
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        eq_vertices.push_back(vertex);
    }

    auto mix = graph->merge(IGpuFx::createMixSegment(eq_vertices.size(), 1), eq_vertices);
    graph->add(IGpuFx::createOutputMap({0}));

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "mix-console-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 1, simulate_buffer_intervals));
    delete evaluator;
}

void mixConsoleMultiStage(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->addRoot(IGpuFx::createInputMap({0}));

    std::vector<IGpuSignalVertex*> input_channels;
    std::vector<IGpuSignalVertex*> bus_channels;
    std::vector<IGpuSignalVertex*> matrix_channels;

    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->split({IGpuFx::createGate(0.2, 100, 5, 50)}, input_map)[0];
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        input_channels.push_back(vertex);
    }

    for (size_t i = 0; i < n_lanes / 2; i++) {
        auto vertex = graph->merge(IGpuFx::createMixSegment(input_channels.size(), 1), input_channels);
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        bus_channels.push_back(vertex);
    }

    for (size_t i = 0; i < n_lanes / 4; i++) {
        auto vertex = graph->merge(IGpuFx::createMixSegment(bus_channels.size(), 1), bus_channels);
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        matrix_channels.push_back(vertex);
    }

    auto mix = graph->merge(IGpuFx::createMixSegment(matrix_channels.size(), 1), matrix_channels);
    graph->add(IGpuFx::createOutputMap({0}));

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "mix-console-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 1, simulate_buffer_intervals));
    delete evaluator;
}

void parallelNam(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));

    std::vector<IGpuSignalVertex*> vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->split({IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, process_buffer_size)}, input_map)[0];
        vertices.push_back(vertex);
    }

    auto mix = graph->merge(IGpuFx::createMixSegment(vertices.size(), 1), vertices);
    graph->add(IGpuFx::createOutputMap({0}));

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "parallel-nam-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 1, simulate_buffer_intervals));
    delete evaluator;
}

void serialNam(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));

    std::vector<IGpuSignalVertex*> vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->add(IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, process_buffer_size));
        vertices.push_back(vertex);
    }

    graph->add(IGpuFx::createOutputMap({0}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "serial-nam-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 1, simulate_buffer_intervals));
    delete evaluator;
}

void parallelIr(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0, 1}));

    std::vector<IGpuSignalVertex*> vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->split({IGpuFx::createConv2i2(IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-2c.wav")), 1 << 16, 0, false)}, input_map)[0];
        vertices.push_back(vertex);
    }

    auto mix = graph->merge(IGpuFx::createMixSegment(vertices.size() * 2, 2), vertices);
    graph->add(IGpuFx::createOutputMap({0, 1}), mix);

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "parallel-ir-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 2, simulate_buffer_intervals));
    delete evaluator;
}

void serialIr(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    auto input_map = graph->add(IGpuFx::createInputMap({0}));

    std::vector<IGpuSignalVertex*> vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto vertex = graph->add(IGpuFx::createConv1i1(IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav")), 1 << 12, 0, true));
        vertices.push_back(vertex);
    }

    graph->add(IGpuFx::createOutputMap({0}));
    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "serial-ir-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, 1, simulate_buffer_intervals));
    delete evaluator;
}

void gitParallel(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals, size_t n_lanes) {
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-1c.wav"), ChannelOrder::PLANAR);

    std::vector<size_t> channel_mapping(n_lanes);
    std::iota(std::begin(channel_mapping), std::end(channel_mapping), 0);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();

    std::vector<IGpuSignalVertex*> eq_vertices;
    for (size_t i = 0; i < n_lanes; i++) {
        auto input_map = graph->addRoot(IGpuFx::createInputMap({i}));
        auto vertex = graph->add({IGpuFx::createGate(0.2, 100, 5, 50)}, input_map);
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        vertex = graph->add(IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, process_buffer_size), vertex);
        vertex = graph->add(IGpuFx::createConv1i1(IPCMSignal::readFromFile(path::ir("vocal-duo-48k-24b-1c.wav")), 1 << 12, 0, true), vertex);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        eq_vertices.push_back(vertex);
        graph->add(IGpuFx::createOutputMap({0}), vertex);
    }

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "parallel-git-" + std::to_string(n_lanes) + "c");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, n_lanes, simulate_buffer_intervals));
    delete evaluator;
    delete dry;
}

void gitComplex(std::vector<PerformanceMeasurement*>& measurements, size_t process_buffer_size, size_t n_warmup, size_t n_measure, bool simulate_buffer_intervals) {
    IFPSignal* dry = IFPSignal::readFromFile(path::res("sine-sweep-48k-24b-2c.wav"), ChannelOrder::PLANAR);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    size_t n_inputs = 2;

    auto signal = IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"));

    std::vector<IGpuSignalVertex*> eq_vertices;
    for (size_t i = 0; i < n_inputs; i++) {
        auto input_map = graph->addRoot(IGpuFx::createInputMap({i}));
        auto vertex = graph->add({IGpuFx::createGate(0.2, 100, 5, 50)}, input_map);
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        auto vertices = graph->split({
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, process_buffer_size),
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, process_buffer_size),
                                     },
                                     vertex);

        vertices[0] = graph->add(IGpuFx::createConv1i1(signal->clone(), 1 << 12, 0, true), vertices[0]);
        vertices[1] = graph->add(IGpuFx::createConv1i1(signal->clone(), 1 << 12, 0, true), vertices[1]);
        vertex = graph->merge(IGpuFx::createInputMap({0, 1}), vertices);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -13, 3),
                            }),
                            vertex);
        eq_vertices.push_back(vertex);
    }
    graph->merge(IGpuFx::createMixSegment(4, 2), eq_vertices);
    graph->add(IGpuFx::createOutputMap({0, 1}));

    IGpuSignalGraphEvaluator* evaluator = IGpuSignalGraphEvaluator::create(graph, "git-complex");
    measurements.push_back(evaluator->measurePerformance(n_warmup, n_measure, process_buffer_size, n_inputs, simulate_buffer_intervals));
    measurements[0]->print();
    delete evaluator;
    delete dry;
}

int main(int argc, char** argv) {
    size_t process_buffer_size = 32;
    // size_t n_warmup = 0;
    // size_t n_measure = 1000;
    size_t n_warmup = 1000;
    size_t n_measure = 10000;
    size_t n_lanes = std::stoi(argv[1]);
    // size_t n_lanes = 16;
    std::vector<PerformanceMeasurement*> measurements;
    spdlog::info("argv[1]: {}", n_lanes);

    mixConsoleMultiStage(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    PerformanceMeasurement::writeStatisticsToCsv(path::out("mix-console-multi-stage-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // mixConsole(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("mix-console-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // parallelNam(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("parallel-nam-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // serialNam(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("serial-nam-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // parallelIr(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("parallel-ir-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // serialIr(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("serial-ir-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // gitParallel(measurements, process_buffer_size, n_warmup, n_measure, false, n_lanes);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("parallel-git-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    // gitComplex(measurements, process_buffer_size, n_warmup, n_measure, false);
    // PerformanceMeasurement::writeStatisticsToCsv(path::out("parallel-git-" + std::to_string(process_buffer_size) + "f.csv"), measurements, true);

    return 0;
}