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
    size_t buffer_size = 128;

    Driver* driver = Driver::createJackDriver();
    driver->setBufferSize(buffer_size);

    IGpuSignalGraph* graph = IGpuSignalGraph::createGpuSignalGraph();
    size_t n_inputs = 2;

    graph->add(IGpuFx::createInputMap({0}));
    auto vertex = graph->add(IGpuFx::createGate(0.2, 100, 5, 50));
    auto fx_nam = IGpuFx::createNam(path::models("nam_convnet_pedal_amp.onnx"), path::out(), TrtEnginePrecision::FP32, buffer_size);
    vertex = graph->add(fx_nam);

    vertex = graph->add(IGpuFx::createConv1i1(IPCMSignal::readFromFile(path::ir("engl-2022-v30-57-48k-24b-1c.wav")), 1 << 12), vertex);
    vertex = graph->add(IGpuFx::createBiquadEQ({
                            IBiquadParam::create(BiquadType::PEAK, 500, 0, 3),
                            IBiquadParam::create(BiquadType::PEAK, 500, 0, 3),
                            IBiquadParam::create(BiquadType::PEAK, 500, 0, 3),
                            IBiquadParam::create(BiquadType::PEAK, 500, 0, 3),
                            IBiquadParam::create(BiquadType::PEAK, 500, 0, 3),
                        }),
                        vertex);
    graph->add(IGpuFx::createOutputMap({0}));

    driver->addSignalChain((ISignalGraph*)graph, {"capture_1"}, {"playback_1"});

    driver->start(false);

    if (timeout > 0) {
        sleep(5);
        spdlog::info("updating mix_ratio to 0.5");
        fx_nam->setSoftParams(0.5f);
        sleep(5);
        spdlog::info("updating mix_ratio to 0.0");
        fx_nam->setSoftParams(0.0f);
        sleep(timeout);
    } else {
        std::cin.get();
    }
    driver->stop();

    delete driver;
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    run_jack(500);

    return 0;
}
