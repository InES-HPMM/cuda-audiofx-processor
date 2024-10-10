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
#include <path.hpp>
#include <signal.hpp>

void run_jack(int timeout = 0) {
    selectGpu();
    size_t fft_size = 1 << 17;
    size_t buffer_size = 128;

    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    size_t n_inputs = 2;

    auto signal = IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav"));

    std::vector<IGpuSignalVertex*> eq_vertices;
    for (size_t i = 0; i < n_inputs; i++) {
        auto input_map = graph->addRoot(IGpuFx::createInputMap({0}));
        auto vertex = graph->add({IGpuFx::createGate(0.2, 100, 5, 50)}, input_map);
        vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50), vertex);
        auto vertices = graph->split({
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size),
                                         IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size),
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

    driver->addSignalChain((ISignalGraph*)graph, {"capture_1"}, {"playback_1", "playback_2"});

    driver->start();

    if (timeout > 0) {
        sleep(timeout);
    } else {
        std::cin.get();
    }
    driver->stop();

    delete driver;
}

// void run_signal() {
//     selectGpu();
//     size_t fft_size = 1 << 17;
//     size_t buffer_size = 1 << 5;
//     Driver* driver = Driver::createSignalDriver();
//     IGpuSignalChain* chain = IGpuSignalChain::createGpuSignalChain();
//     chain->configure(buffer_size, 2);
//     driver->setBufferSize(buffer_size);
//     driver->addSignalChain(chain, {""}, {"/home/nvidia/git/mt/res/sine-generator-out.wav"});
//     driver->start();

//     delete chain;
//     delete driver;
// }

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    run_jack(500);

    // run_signal();

    return 0;
}
