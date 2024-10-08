This repository contains an implementation of an embedded GPU-based audio processing framework on an Nvidia Jetson hardware platform.
It can combine neural network inference and other audio effects into signal graphs that process within periods as small as 32 frames (0.667ms).

The signal graph does not restrict the number and combination of parallel and serial audio effects as long as the real-time limit is met.
Therefore, the framework has been tested on large numbers of parallel channels, as found in a mixing console, and complex routing options available in high-end audio effect processors, such as the Neural DSP Quad Cortex.

Launching GPU work using the CUDA graph API produces better stability and performance than was observed using the CUDA stream API in a 2017 study.
Processing a signal graph that fully utilises the Jetson's resources by mimicking a 64-channel mixing console on a 128-frame (2.67ms) period has a higher than 99\% success rate.
However, occasional stalling on the GPU can produce worst-case execution times of up to 20ms, regardless of the loaded audio effects.
As a result, the framework can not yet be classified as real-time capable.

Further study of the CUDA scheduler and improvements to the operating system and audio driver may be able to achieve real-time capability in the future.
