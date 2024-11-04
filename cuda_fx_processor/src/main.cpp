#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <cstring>
#include <driver.hpp>
#include <gpu.cuh>
#include <gpu_fx.cuh>
#include <gpu_signal_graph.cuh>
#include <log.hpp>
#include <path.hpp>
#include <signal.hpp>

void run_jack(bool async, size_t n_inputs, size_t buffer_size, int timeout = 0) {
    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    std::vector<IGpuSignalVertex*> eq_vertices;
    IGpuSignalVertex* vertex;
    vertex = graph->addRoot(IGpuFx::createInputMap({0}));
    vertex = graph->add({IGpuFx::createGate(0.005, 100, 5, 50)}, vertex);
    vertex = graph->add(IGpuFx::createNam(path::models("nam_convnet_BDHIII_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size));
    vertex = graph->add(IGpuFx::createConv1c1(IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav")), 1 << 12), vertex);
    graph->add(IGpuFx::createOutputMap({0, 0}));

    driver->addSignalChain((ISignalGraph*)graph, {"capture_2"}, {"playback_1", "playback_2"});

    driver->start(async);

    if (timeout > 0) {
        // sleep(5);
        // spdlog::info("updating mix_ratio to 0.5");
        // fx_nam->setSoftParams(0.5f);
        // sleep(5);
        // spdlog::info("updating mix_ratio to 0.0");
        // fx_nam->setSoftParams(0.0f);
        sleep(timeout);
    } else {
        std::cin.get();
    }
    driver->stop();

    delete driver;
}

void presentation(bool async, size_t n_inputs, size_t buffer_size) {
    auto ir_engl = IPCMSignal::readFromFile(path::ir("ENGL-V30-57-440-48k-24b-2c.wav"));
    auto ir_smg = IPCMSignal::readFromFile(path::ir("SMG-V30-440-57-48k-24b-2c.wav"));
    auto g1_gate = IGpuFx::createGate(0.005, 50, 5, 50);
    auto g1_amp1 = IGpuFx::createNam(path::models("nam_convnet_BDHIII_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g1_amp2 = IGpuFx::createNam(path::models("nam_convnet_BDH5169_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g1_ir = IGpuFx::createConv2c2(ir_engl, 1 << 12, -7);
    auto g1_eq = IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 150, 3, 1),
        IBiquadParam::create(BiquadType::PEAK, 250, -5, 3),
        IBiquadParam::create(BiquadType::PEAK, 500, -1.5, 1),
        IBiquadParam::create(BiquadType::PEAK, 1000, 3, 0.5),
        IBiquadParam::create(BiquadType::PEAK, 2000, -1, 1),
        IBiquadParam::create(BiquadType::PEAK, 4000, 1.5, 0.5),
    });

    auto g2_gate = IGpuFx::createGate(0.01, 50, 5, 50);
    auto g2_amp1 = IGpuFx::createNam(path::models("nam_convnet_BDH5169_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g2_amp2 = IGpuFx::createNam(path::models("nam_convnet_BDHIII_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g2_ir = IGpuFx::createConv2c2(ir_smg, 1 << 12, -7);
    auto g2_eq = IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 150, 3, 1),
        IBiquadParam::create(BiquadType::PEAK, 250, -5, 3),
        IBiquadParam::create(BiquadType::PEAK, 500, -1.5, 1),
        IBiquadParam::create(BiquadType::PEAK, 1000, 3, 0.5),
        IBiquadParam::create(BiquadType::PEAK, 2000, -1, 1),
        IBiquadParam::create(BiquadType::PEAK, 4000, 1.5, 0.5),
    });

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();

    auto g1_vertex = graph->addRoot(IGpuFx::createInputMap({0}));
    auto g2_vertex = graph->addRoot(IGpuFx::createInputMap({1}));

    g1_vertex = graph->add(g1_gate, g1_vertex);
    auto g1_vertices = graph->split({g1_amp1, g1_amp2}, g1_vertex);
    g1_vertex = graph->merge(IGpuFx::createInputMap({0, 1}), g1_vertices);
    g1_vertex = graph->add(g1_ir, g1_vertex);
    g1_vertex = graph->add(g1_eq, g1_vertex);

    g2_vertex = graph->add(g2_gate, g2_vertex);
    auto g2_vertices = graph->split({g2_amp1, g2_amp2}, g2_vertex);
    g2_vertex = graph->merge(IGpuFx::createInputMap({0, 1}), g2_vertices);
    g2_vertex = graph->add(g2_ir, g2_vertex);
    g2_vertex = graph->add(g2_eq, g2_vertex);

    graph->merge(IGpuFx::createMixSegment(4, 2), {g1_vertex, g2_vertex});

    graph->add(IGpuFx::createOutputMap({0, 1}));

    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);
    driver->addSignalChain((ISignalGraph*)graph, {"capture_1", "capture_2"}, {"playback_1", "playback_2"});
    driver->start(async);

    g1_amp1->setSoftParams(0);
    g1_amp2->setSoftParams(0);
    g1_eq->setSoftParams(0);
    g1_gate->setSoftParams(0);
    g1_ir->setSoftParams(0);

    g2_amp1->setSoftParams(0);
    g2_amp2->setSoftParams(0);
    g2_eq->setSoftParams(0);
    g2_gate->setSoftParams(0);
    g2_ir->setSoftParams(0);

    std::cin.get();
    spdlog::info("G1 Amp + IR");
    g1_amp1->setSoftParams(1);
    g1_ir->setSoftParams(1);
    g1_amp2->setSoftParams(1);

    std::cin.get();
    spdlog::info("G1 Gate");
    g1_gate->setSoftParams(1);
    std::cin.get();
    spdlog::info("G1 EQ");
    g1_eq->setSoftParams(1);

    std::cin.get();
    spdlog::info("G2");
    // spdlog::info("G2 Amp1 + IR");
    g2_amp1->setSoftParams(1);
    g2_ir->setSoftParams(1);
    // std::cin.get();
    // spdlog::info("G2 Amp2");
    g2_amp2->setSoftParams(1);
    // std::cin.get();
    // spdlog::info("G2 Gate");
    g2_gate->setSoftParams(1);
    // std::cin.get();
    // spdlog::info("G2 EQ");
    g2_eq->setSoftParams(1);
    std::cin.get();

    driver->stop();

    delete driver;
}

void presentation_live(bool async, size_t n_inputs, size_t buffer_size) {
    auto ir_engl = IPCMSignal::readFromFile(path::ir("ENGL-V30-57-440-48k-24b-2c.wav"));
    auto ir_smg = IPCMSignal::readFromFile(path::ir("SMG-V30-440-57-48k-24b-2c.wav"));
    auto g1_gate = IGpuFx::createGate(0.005, 50, 5, 50);
    auto g1_amp1 = IGpuFx::createNam(path::models("nam_convnet_BDHIII_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g1_amp2 = IGpuFx::createNam(path::models("nam_convnet_BDH5169_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    auto g1_ir = IGpuFx::createConv2c2(ir_engl, 1 << 12, -18);
    auto g1_eq = IGpuFx::createBiquadEQ({
        IBiquadParam::create(BiquadType::PEAK, 150, 3, 1),
        IBiquadParam::create(BiquadType::PEAK, 250, -5, 3),
        IBiquadParam::create(BiquadType::PEAK, 500, -1.5, 1),
        IBiquadParam::create(BiquadType::PEAK, 1000, 3, 0.5),
        IBiquadParam::create(BiquadType::PEAK, 2000, -1, 1),
        IBiquadParam::create(BiquadType::PEAK, 4000, 1.5, 0.5),
    });

    // auto g2_gate = IGpuFx::createGate(0.01, 50, 5, 50);
    // auto g2_amp1 = IGpuFx::createNam(path::models("nam_convnet_BDH5169_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    // auto g2_amp2 = IGpuFx::createNam(path::models("nam_convnet_BDHIII_pedal_amp_E400.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    // auto g2_ir = IGpuFx::createConv2c2(ir_smg, 1 << 12, -7);
    // auto g2_eq = IGpuFx::createBiquadEQ({
    //     IBiquadParam::create(BiquadType::PEAK, 150, 3, 1),
    //     IBiquadParam::create(BiquadType::PEAK, 250, -5, 3),
    //     IBiquadParam::create(BiquadType::PEAK, 500, -1.5, 1),
    //     IBiquadParam::create(BiquadType::PEAK, 1000, 3, 0.5),
    //     IBiquadParam::create(BiquadType::PEAK, 2000, -1, 1),
    //     IBiquadParam::create(BiquadType::PEAK, 4000, 1.5, 0.5),
    // });
    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();

    auto g1_vertex = graph->addRoot(IGpuFx::createInputMap({0}));
    // auto g2_vertex = graph->addRoot(IGpuFx::createInputMap({1}));

    g1_vertex = graph->add(g1_gate, g1_vertex);
    auto g1_vertices = graph->split({g1_amp1, g1_amp2}, g1_vertex);
    g1_vertex = graph->merge(IGpuFx::createInputMap({0, 1}), g1_vertices);
    g1_vertex = graph->add(g1_ir, g1_vertex);
    g1_vertex = graph->add(g1_eq, g1_vertex);

    // g2_vertex = graph->add(g2_gate, g2_vertex);
    // auto g2_vertices = graph->split({g2_amp1, g2_amp2}, g2_vertex);
    // g2_vertex = graph->merge(IGpuFx::createInputMap({0, 1}), g2_vertices);
    // g2_vertex = graph->add(g2_ir, g2_vertex);
    // g2_vertex = graph->add(g2_eq, g2_vertex);

    // graph->merge(IGpuFx::createMixSegment(2, 1), {g1_vertex, g2_vertex});

    graph->add(IGpuFx::createOutputMap({0, 1}));

    driver->addSignalChain((ISignalGraph*)graph, {"capture_1"}, {"playback_1", "playback_2"});
    driver->start(async);

    g1_amp1->setSoftParams(0);
    g1_amp2->setSoftParams(0);
    g1_eq->setSoftParams(0);
    g1_gate->setSoftParams(0);
    g1_ir->setSoftParams(0);

    // g2_amp1->setSoftParams(0);
    // g2_amp2->setSoftParams(0);
    // g2_eq->setSoftParams(0);
    // g2_gate->setSoftParams(0);
    // g2_ir->setSoftParams(0);

    std::cin.get();
    spdlog::info("G1 Amp + IR");
    g1_amp1->setSoftParams(1);
    g1_ir->setSoftParams(1);
    g1_amp2->setSoftParams(1);

    std::cin.get();
    spdlog::info("G1 Gate");
    g1_gate->setSoftParams(1);
    std::cin.get();
    spdlog::info("G1 EQ");
    g1_eq->setSoftParams(1);

    std::cin.get();
    // spdlog::info("G2");
    // // spdlog::info("G2 Amp1 + IR");
    // g2_amp1->setSoftParams(1);
    // g2_ir->setSoftParams(1);
    // // std::cin.get();
    // // spdlog::info("G2 Amp2");
    // g2_amp2->setSoftParams(1);
    // // std::cin.get();
    // // spdlog::info("G2 Gate");
    // g2_gate->setSoftParams(1);
    // // std::cin.get();
    // // spdlog::info("G2 EQ");
    // g2_eq->setSoftParams(1);
    // std::cin.get();

    driver->stop();

    delete driver;
}

int main(int argc, char* argv[]) {
    bool async = false;
    size_t n_inputs = 2;
    size_t buffer_size = 128;
    if (argc >= 2) {
        async = std::stoi(argv[1]);
        spdlog::info("async: {}", async);
        if (argc >= 3) {
            n_inputs = std::stoi(argv[2]);
            spdlog::info("n_inputs: {}", std::to_string(n_inputs));
            if (argc >= 4) {
                buffer_size = std::stoi(argv[3]);
                spdlog::info("buffer_size: {}", std::to_string(buffer_size));
            }
        }
    }

    setup_spdlog();
    spdlog::set_level(spdlog::level::info);
    selectGpu();
    // run_jack(async, n_inputs, buffer_size, 500);
    // presentation(async, n_inputs, buffer_size);
    presentation_live(async, n_inputs, buffer_size);

    return 0;
}
