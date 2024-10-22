#include <jack/jack.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <coding.cuh>
#include <cstring>
#include <driver.hpp>
#include <gpu.cuh>
#include <gpu_fx.cuh>
#include <gpu_signal_graph.cuh>
#include <log.hpp>
#include <path.hpp>
#include <signal.hpp>

void run_jack(bool async, size_t n_inputs, size_t buffer_size, int timeout = 0) {
    selectGpu();

    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);

    auto signal = IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"));

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    std::vector<IGpuSignalVertex*> eq_vertices;
    for (size_t i = 0; i < n_inputs; i++) {
        auto input_map = graph->addRoot(IGpuFx::createInputMap({0}));
        IGpuSignalVertex* vertex;
        vertex = graph->add({IGpuFx::createGate(0.01, 100, 5, 50)}, input_map);
        auto vertices = graph->split({
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size),
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size),
                                     },
                                     vertex);
        vertices[0] = graph->add(IGpuFx::createConv1c1(signal->clone(), 1 << 12), vertices[0]);
        vertices[1] = graph->add(IGpuFx::createConv1c1(signal->clone(), 1 << 12), vertices[1]);
        vertex = graph->merge(IGpuFx::createInputMap({0, 1}), vertices);
        vertex = graph->add(IGpuFx::createBiquadEQ({
                                IBiquadParam::create(BiquadType::PEAK, 150, 3, 1),
                                IBiquadParam::create(BiquadType::PEAK, 250, -5, 3),
                                IBiquadParam::create(BiquadType::PEAK, 500, -1.5, 1),
                                IBiquadParam::create(BiquadType::PEAK, 1000, 3, 0.5),
                                IBiquadParam::create(BiquadType::PEAK, 2000, -1, 1),
                                IBiquadParam::create(BiquadType::PEAK, 4000, 1.5, 0.5),
                            }),
                            vertex);
        vertex = graph->add(IGpuFx::createConv2c2(IPCMSignal::readFromFile(path::ir("drum-room-48k-24b-2c.wav")), 1 << 16, -18, 0.5f), vertex);
        eq_vertices.push_back(vertex);
    }
    if (n_inputs > 2) {
        graph->merge(IGpuFx::createMixSegment(n_inputs, 2), eq_vertices);
    }
    graph->add(IGpuFx::createOutputMap({0, 1}));

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

int main(int argc, char* argv[]) {
    bool async = false;
    size_t n_inputs = 2;
    size_t buffer_size = 128;
    if (argc >= 1) {
        async = std::stoi(argv[1]);
        spdlog::info("async: {}", async);
        if (argc >= 2) {
            n_inputs = std::stoi(argv[2]);
            spdlog::info("n_inputs: {}", std::to_string(n_inputs));
            if (argc >= 3) {
                buffer_size = std::stoi(argv[3]);
                spdlog::info("buffer_size: {}", std::to_string(buffer_size));
            }
        }
    }

    setup_spdlog();
    spdlog::set_level(spdlog::level::info);
    run_jack(async, n_inputs, buffer_size, 500);

    return 0;
}
