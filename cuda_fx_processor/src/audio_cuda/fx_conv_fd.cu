
#include <cuda_runtime.h>
#include <cufft.h>

#include <coding.cuh>
#include <cuda_ext.cuh>
#include <gpu.cuh>
#include <log.hpp>
#include <math_ext.hpp>
#include <operators.cuh>
#include <signal.hpp>

#include "gpu_fx.cu"

__device__ inline cufftComplex conjugate(cufftComplex v) { return {v.x, -v.y}; }
__device__ inline cufftComplex timesj(cufftComplex v) { return {-v.y, v.x}; }

__global__ static void ccc_unpackCto2C(cufftComplex* dest1, cufftComplex* dest2, const cufftComplex* src, size_t fftSize) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < fftSize / 2; s += stride.x) {
        auto idxa = s;
        auto idxb = (fftSize - s);

        auto va = src[idxa];
        auto vb = s ? conjugate(src[idxb]) : va;
        auto la = 0.5f * (va + vb);
        auto lb = timesj(-0.5f * (va - vb));

        dest1[idxa] = la;
        dest2[idxa] = lb;
        if (s) {
            dest1[idxb] = conjugate(la);
            dest2[idxb] = conjugate(lb);
        }
    }
}

__global__ static void ccc_complexMultiplyAndScale(cufftComplex* dst, const cufftComplex* src1, const cufftComplex* src2, size_t n, float scale) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s] = cuCmulf(src1[s], src2[s]) * scale;
    }
}

__global__ static void f2f2f2_multiplyAndScale(float2* dst, const float2* src1, const float2* src2, size_t n, float scale) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s] = src1[s] * src2[s] * scale;
    }
}

__global__ static void f2f2f2_pointwiseAdd(float2* dst, float2* src1, const float2* src2, size_t n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s] = src1[s] + src2[s];
    }
}

__global__ static void f2fff2_pointwiseAdd(float2* dst, float* src1x, float* src1y, const float2* src2, size_t n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s].x = src1x[s] + src2[s].x;
        dst[s].y = src1y[s] + src2[s].y;
    }
}

__global__ static void fff_pointwiseAdd(float* dst, float* src1, const float* src2, size_t n) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s] = src1[s] + src2[s];
    }
}

__global__ static void f2f2f2_mix(float2* dst, const float2* src1, const float2* src2, size_t n, float ratio) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s].x = src1[s].x * (1 - ratio) + src2[s].x * ratio;
        dst[s].y = src1[s].y * (1 - ratio) + src2[s].y * ratio;
    }
}

__global__ static void fff_mix(float* dst, const float* src1, const float* src2, size_t n, float ratio) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        dst[s] = src1[s] * (1 - ratio) + src2[s] * ratio;
    }
}

__global__ static void f2_scale(float2* data, size_t n, float scale) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n; s += stride.x) {
        data[s] *= scale;
    }
}

class FxConvFd : public GpuFx {
   protected:
    IPCMSignal* _ir_signal;

    dim3 _nBlocks;
    dim3 _nThreads;
    size_t _nSharedMem;

    size_t _fft_size;
    size_t _fft_size_non_redundant;
    int _ir_db_scale;
    float _mix_ratio;
    bool _force_wet_mix;

    IMemCpyNode* _src_node = nullptr;
    IKernelNode* _dest_node = nullptr;
    char* _ir_byte_buf = nullptr;

    float getIRAttenuationFactor() const {
        return pow(10, _ir_db_scale * 0.05);  // IRs are usually too loud, so we scale them by _ir_db_scale dB
    }

    size_t getMinIROrFFTSize() const {
        return std::min(_ir_signal->getFrameCount(), _fft_size);
    }

    virtual void allocateBuffers() = 0;
    virtual void deallocateBuffers() = 0;
    virtual cudaStream_t _process(cudaStream_t stream, float* src, const float* dst, cudaStreamCaptureStatus capture_status) = 0;

   public:
    FxConvFd(std::string name, IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) : GpuFx(name), _nBlocks(256), _nThreads(256), _nSharedMem(0), _ir_signal(ir_signal), _ir_db_scale(ir_db_scale), _mix_ratio(0.5), _force_wet_mix(force_wet_mix) {
        if (max_ir_size == 0) {
            max_ir_size = _ir_signal->getFrameCount();
        }
        _fft_size = roundUpToPow2(std::min(_ir_signal->getFrameCount(), max_ir_size));
        _fft_size_non_redundant = _fft_size / 2 + 1;
        if (_ir_signal->getFrameCount() > _fft_size) {
            spdlog::warn("IR file given to {} is longer than specified max fft size of {}. Truncating to {} samples.", name, max_ir_size, _fft_size);
        }
    }

    virtual ~FxConvFd() {
        delete _ir_signal;
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        if (n_input_channels != 0 && n_input_channels != _n_in_channels || n_output_channels != 0 && n_output_channels != _n_out_channels) {
            spdlog::warn("{} is a fixed channel fx ({}i{}). The configured channel count will be ignored {}i{}.", _name, _n_in_channels, _n_out_channels, n_input_channels, n_output_channels);
        }
        GpuFx::configure(process_buffer_size, _n_in_channels, _n_out_channels);
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        _src_node->updateSrcPtr(src->getDataMod(), procGraphExec);
        _dest_node->updateKernelParamAt(0, dst->getDataMod(), procGraphExec);
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        return _process(stream, dst->getDataMod(), src->getDataMod(), capture_status);
    }

    virtual void teardown() override {
        if (_src_node) delete _src_node;
        if (_dest_node) delete _dest_node;
        GpuFx::teardown();
    }
};

class FxConvFd1c1 : public FxConvFd {
   private:
    cufftHandle _plan_r2c;
    cufftHandle _plan_c2r;

    float* _ir_td = nullptr;
    cufftComplex* _ir_fft = nullptr;
    float* _sig_td = nullptr;
    cufftComplex* _sig_fft = nullptr;
    float* _wet_td = nullptr;
    float* _residual_td = nullptr;

    void allocateBuffers() {
        gpuErrChk(cudaMalloc(&_ir_byte_buf, sizeof(char) * _ir_signal->getByteCount()));
        gpuErrChk(cudaMalloc(&_ir_td, sizeof(float) * _fft_size));
        gpuErrChk(cudaMalloc(&_ir_fft, sizeof(cufftComplex) * _fft_size));
        gpuErrChk(cudaMalloc(&_sig_td, sizeof(float) * _fft_size));
        gpuErrChk(cudaMalloc(&_sig_fft, sizeof(cufftComplex) * _fft_size));
        gpuErrChk(cudaMalloc(&_wet_td, sizeof(float) * _fft_size));
        gpuErrChk(cudaMalloc(&_residual_td, sizeof(float) * _fft_size));

        cufftPlan1d(&_plan_r2c, _fft_size, CUFFT_R2C, 1);
        cufftPlan1d(&_plan_c2r, _fft_size, CUFFT_C2R, 1);
    }

    void deallocateBuffers() {
        gpuErrChk(cudaFree(_ir_byte_buf));
        gpuErrChk(cudaFree(_ir_td));
        gpuErrChk(cudaFree(_ir_fft));
        gpuErrChk(cudaFree(_sig_td));
        gpuErrChk(cudaFree(_sig_fft));
        gpuErrChk(cudaFree(_wet_td));
        gpuErrChk(cudaFree(_residual_td));

        cufftDestroy(_plan_r2c);
        cufftDestroy(_plan_c2r);
    }

    cudaStream_t _process(cudaStream_t stream, float* dst, const float* src, cudaStreamCaptureStatus capture_status) override {
        // copy input as contiguous _n_proc_frames * float real signal into complex buffer to enable in-place fft
        IMemCpyNode::launchOrRecord1D(_sig_fft, src, sizeof(float), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _src_node, capture_status);

        cufftSetStream(_plan_r2c, stream);
        cufftExecR2C(_plan_r2c, (cufftReal*)_sig_fft, _sig_fft);

        // Convolution (Colplex Multiplication in Frequency Domain) (scaling is needed to retain unity gain in time domain after inverse fft)
        ccc_complexMultiplyAndScale<<<4, 768, 0, stream>>>(_sig_fft, _sig_fft, _ir_fft, _fft_size_non_redundant, 1.0f / _fft_size);

        // Inverse FFT
        cufftSetStream(_plan_c2r, stream);
        // using inplace transform to avoid allocating additional buffers
        // output is _fft_size * float contiguous real signal, with the other half of the complex buffer being irrelevant
        cufftExecC2R(_plan_c2r, _sig_fft, (cufftReal*)_sig_fft);

        if (_force_wet_mix) {
            // if the fx should always produce a 100% wet output (e.g. amp cab ir), we can skip the mixing step and write the sum of the output and residual directly to the dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)fff_pointwiseAdd, new void*[4]{&dst, &_sig_fft, &_residual_td, &_n_proc_frames}, stream, _dest_node, capture_status);
        } else {
            // combine convolution output with residual and write to dst buffer
            fff_pointwiseAdd<<<1, _n_proc_frames, 0, stream>>>(_wet_td, (float*)_sig_fft, _residual_td, _n_proc_frames);
            // mix dry and wet signal and write to dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)fff_mix, new void*[5]{&dst, &src, &_wet_td, &_n_proc_frames, &_mix_ratio}, stream, _dest_node, capture_status);
        }
        return stream;
    }

   public:
    FxConvFd1c1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) : FxConvFd("FxConvFd1c1", ir_signal, max_ir_size, ir_db_scale, force_wet_mix) {
        if (_ir_signal->getChannelCount() > 1) {
            spdlog::warn("IR given to {} file is not mono. Only the first channel will be used.", _name);
        }
        _n_in_channels = 1;
        _n_out_channels = 1;
    }

    ~FxConvFd1c1() {}

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        gpuErrChk(cudaMemsetAsync(_residual_td, 0, sizeof(float) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_ir_td, 0, sizeof(float) * _fft_size, stream));
        // gpuErrChk(cudaMemsetAsync(_sig_td, 0, sizeof(float) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_sig_fft, 0, sizeof(float2) * _fft_size, stream));

        // Memcopy2D is used since to copy only the first channel from multichannel IRs
        gpuErrChk(cudaMemcpy2DAsync(_ir_byte_buf, _ir_signal->getByteDepth(), _ir_signal->getDataPtrConst(), _ir_signal->getByteDepth() * _ir_signal->getChannelCount(), _ir_signal->getByteDepth(), _ir_signal->getFrameCount(), cudaMemcpyHostToDevice, stream));

        pcm_to_float_interleaved(_ir_td, _ir_byte_buf, getMinIROrFFTSize(), 1, _ir_signal->getBitDepthValue(), stream);
        cufftSetStream(_plan_r2c, stream);
        cufftExecR2C(_plan_r2c, (cufftReal*)_ir_td, _ir_fft);

        // irs are usually normalized to max amplitude and therefore much too loud. We scale them down to avoid clipping
        f2_scale<<<_nBlocks, _nThreads, _nSharedMem, stream>>>(_ir_fft, _fft_size, getIRAttenuationFactor());
        return stream;
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // add convolution output to residual buffer and clear the unused residual tail
        // can't use inplace processing for rewinding, because when kernel is strided (thread count < fft size) or when using memcopy, the out-of-order execution of the threads could cause recursive reads of already processed data
        // which is why I'm missusing the _wet_td buffer as a temporary buffer for the sum of the output and residual
        // fff_pointwiseAdd<<<4, 768, 0, stream>>>(_wet_td, _residual_td + _n_proc_frames, _sig_td + _n_proc_frames, _fft_size - _n_proc_frames);
        fff_pointwiseAdd<<<4, 768, 0, stream>>>(_wet_td, _residual_td + _n_proc_frames, ((float*)_sig_fft) + _n_proc_frames, _fft_size - _n_proc_frames);
        gpuErrChk(cudaMemcpyAsync(_residual_td, _wet_td, sizeof(float) * (_fft_size - _n_proc_frames), cudaMemcpyDeviceToDevice, stream));
        gpuErrChk(cudaMemsetAsync(_residual_td + _fft_size - _n_proc_frames, 0, sizeof(float) * _n_proc_frames, stream));

        // clear the part of _sig_fft that is not overwritten by the input
        // gpuErrChk(cudaMemsetAsync(_sig_td + _n_proc_frames, 0, sizeof(float) * (_fft_size - _n_proc_samples), stream));
        gpuErrChk(cudaMemsetAsync(_sig_fft, 0, sizeof(float2) * _fft_size, stream));

        return stream;
    }

    GpuFx* clone() override {
        return new FxConvFd1c1(_ir_signal->clone(), _fft_size, _ir_db_scale, _force_wet_mix);
    }
};

IGpuFx* IGpuFx::createConv1i1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) {
    return new FxConvFd1c1(ir_signal, max_ir_size, ir_db_scale, force_wet_mix);
}

class FxConvFd2c1 : public FxConvFd {
   private:
    cufftHandle _plan_c2c;

    float* _ir_mono = nullptr;
    float2* _ir_stereo = nullptr;
    cufftComplex* _ir_fft = nullptr;

    cufftComplex* _sig_fft = nullptr;
    float2* _residual = nullptr;
    float2* _wet = nullptr;

    void allocateBuffers() {
        gpuErrChk(cudaMalloc(&_ir_byte_buf, sizeof(char) * _ir_signal->getByteCount()));
        gpuErrChk(cudaMalloc(&_ir_mono, sizeof(float) * getMinIROrFFTSize()));
        gpuErrChk(cudaMalloc(&_ir_stereo, sizeof(float2) * _fft_size));
        gpuErrChk(cudaMalloc(&_ir_fft, sizeof(cufftComplex) * _fft_size));
        gpuErrChk(cudaMalloc(&_sig_fft, sizeof(cufftComplex) * _fft_size));
        gpuErrChk(cudaMalloc(&_residual, sizeof(float2) * _fft_size));
        gpuErrChk(cudaMalloc(&_wet, sizeof(float2) * _fft_size));

        cufftPlan1d(&_plan_c2c, _fft_size, CUFFT_C2C, 1);
    }

    void deallocateBuffers() {
        gpuErrChk(cudaFree(_ir_byte_buf));
        gpuErrChk(cudaFree(_ir_mono));
        gpuErrChk(cudaFree(_ir_stereo));
        gpuErrChk(cudaFree(_ir_fft));
        gpuErrChk(cudaFree(_sig_fft));
        gpuErrChk(cudaFree(_residual));
        gpuErrChk(cudaFree(_wet));

        cufftDestroy(_plan_c2c);
    }

    cudaStream_t _process(cudaStream_t stream, float* dst, const float* src, cudaStreamCaptureStatus capture_status) {
        cufftSetStream(_plan_c2c, stream);

        // pack stereo channels into real and img part of complex struct and perform fft both simultaneously
        // https://web.archive.org/web/20180312110051/http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM
        // since the type cufftComplex is a struct with two float members, we can simply copy our float2 buffer into the cufftComplex buffer
        IMemCpyNode::launchOrRecord1D(_sig_fft, src, sizeof(float2), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _src_node, capture_status);
        cufftExecC2C(_plan_c2c, _sig_fft, _sig_fft, CUFFT_FORWARD);

        // Convolution (Colplex Multiplication in Frequency Domain) (scaling is needed to retain unity gain in time domain after inverse fft)
        // According to the following link, when using the 2for1 method and convolving both signals with the same IR, a simple FLOAT multiplication of the respective real (ch1) and imag (ch2) parts is sufficient
        // https://web.archive.org/web/20180312110051/http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM
        f2f2f2_multiplyAndScale<<<4, 768, 0, stream>>>(_sig_fft, _sig_fft, _ir_fft, _fft_size, 1.0f / _fft_size);

        // Inverse FFT
        cufftExecC2C(_plan_c2c, _sig_fft, _sig_fft, CUFFT_INVERSE);

        if (_force_wet_mix) {
            // if the fx should always produce a 100% wet output (e.g. amp cab ir), we can skip the mixing step and write the sum of the output and residual directly to the dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)f2f2f2_pointwiseAdd, new void*[4]{&dst, &_sig_fft, &_residual, &_n_proc_frames}, stream, _dest_node, capture_status);
        } else {
            // combine convolution output with residual and write to dst buffer
            f2f2f2_pointwiseAdd<<<1, _n_proc_frames, 0, stream>>>(_wet, _sig_fft, _residual, _n_proc_frames);
            // mix dry and wet signal and write to dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)f2f2f2_mix, new void*[5]{&dst, &src, &_wet, &_n_proc_frames, &_mix_ratio}, stream, _dest_node, capture_status);
        }

        return stream;
    }

   public:
    FxConvFd2c1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) : FxConvFd("FxConvFd2c1", ir_signal, max_ir_size, ir_db_scale, force_wet_mix) {
        if (_ir_signal->getChannelCount() > 1) {
            spdlog::warn("IR file given to {} is not mono. Only the first channel will be used.", _name);
        }
        _n_in_channels = 2;
        _n_out_channels = 2;
    }

    ~FxConvFd2c1() {
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        gpuErrChk(cudaMemsetAsync(_residual, 0, sizeof(float2) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_ir_stereo, 0, sizeof(float2) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_sig_fft, 0, sizeof(cufftComplex) * _fft_size, stream));

        // copy ir to device and convert to float. Memcopy2D is used since to copy only the first channel from multichannel IRs
        gpuErrChk(cudaMemcpy2DAsync(_ir_byte_buf, _ir_signal->getByteDepth(), _ir_signal->getDataPtrConst(), _ir_signal->getByteDepth() * _ir_signal->getChannelCount(), _ir_signal->getByteDepth(), _ir_signal->getFrameCount(), cudaMemcpyHostToDevice, stream));
        // here we copy the mono IR to stereo and then cast it to complex buffer to perform an FFT on both channels simultaneously
        // while this seems to unnecessarily redundant, this theqnique allows us to multiply the IR FFT with a stereo signal FFT that is optained the same way
        // as a result, each processing pass only requires one forware and one inverse FFT instead of two for each direction
        // https://web.archive.org/web/20180312110051/http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM
        pcm_to_float_interleaved(_ir_mono, _ir_byte_buf, getMinIROrFFTSize(), 1, _ir_signal->getBitDepthValue(), stream);

        // TODO: produces invalid argument error for the second fx instance in a signal graph. replace 2D copies with multi memcpy once fixed
        // IMemCpyNode::launchOrRecordMulti(MultiMemcpyType::Segmented2Interleaved, _ir_stereo, &_ir_mono, sizeof(float), getMinIROrFFTSize(), 2, {0,0}, cudaMemcpyDeviceToDevice, stream, nullptr, capture_status);
        gpuErrChk(cudaMemcpy2DAsync(_ir_stereo, sizeof(float2), _ir_mono, sizeof(float), sizeof(float), getMinIROrFFTSize(), cudaMemcpyDeviceToDevice, stream));
        gpuErrChk(cudaMemcpy2DAsync(((float*)_ir_stereo) + 1, sizeof(float2), _ir_mono, sizeof(float), sizeof(float), getMinIROrFFTSize(), cudaMemcpyDeviceToDevice, stream));

        cufftSetStream(_plan_c2c, stream);
        cufftExecC2C(_plan_c2c, (cufftComplex*)_ir_stereo, _ir_fft, CUFFT_FORWARD);

        // irs are usually normalized to max amplitude and therefore much too loud. We scale them down to avoid clipping
        f2_scale<<<4, 768, 0, stream>>>(_ir_fft, _fft_size, getIRAttenuationFactor());
        return stream;
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // add convolution output to residual buffer and clear the unused residual tail
        // can't use inplace processing for rewinding, because when kernel is strided (thread count < fft size) or when using memcopy, the out-of-order execution of the threads could cause recursive reads of already processed data
        // which is why I'm missusing the _wet buffer as a temporary buffer for the sum of the output and residual
        f2f2f2_pointwiseAdd<<<4, 768, 0, stream>>>(_wet, _residual + _n_proc_frames, _sig_fft + _n_proc_frames, _fft_size - _n_proc_frames);
        gpuErrChk(cudaMemcpyAsync(_residual, _wet, sizeof(float2) * (_fft_size - _n_proc_frames), cudaMemcpyDeviceToDevice, stream));
        gpuErrChk(cudaMemsetAsync(_residual + _fft_size - _n_proc_frames, 0, sizeof(float2) * _n_proc_frames, stream));

        // clear the part of _sig_fft that is not overwritten by the input
        gpuErrChk(cudaMemsetAsync(_sig_fft + _n_proc_frames, 0, sizeof(float2) * (_fft_size - _n_proc_frames), stream));
        return stream;
    }

    GpuFx* clone() override {
        return new FxConvFd2c1(_ir_signal->clone(), _fft_size, _ir_db_scale, _force_wet_mix);
    }
};
IGpuFx* IGpuFx::createConv2i1(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) {
    return new FxConvFd2c1(ir_signal, max_ir_size, ir_db_scale, force_wet_mix);
}

class FxConvFd2c2 : public FxConvFd {
   private:
    cufftHandle _plan_c2c;
    cufftHandle _plan_c2r;

    float2* _residual;
    float2* _rirBuf;
    float2* _output;
    float2* _wet;
    float* _left;
    float* _right;
    cufftComplex* _rir_fft_packed;
    cufftComplex* _sig_fft_packed;
    cudaStream_t _stream_right;
    struct
    {
        cufftComplex *left, *right;
    } _sigFFT, _rirFFT;

    float2** float2_buffers[9] = {
        &_rir_fft_packed,
        &_rirFFT.left,
        &_rirFFT.right,
        &_sig_fft_packed,
        &_sigFFT.left,
        &_sigFFT.right,
        &_output,
        &_wet,
        &_residual,
    };

    void allocateBuffers() {
        for (size_t i = 0; i < sizeof(float2_buffers) / sizeof(*float2_buffers); i++) {
            gpuErrChk(cudaMalloc(float2_buffers[i], _fft_size * sizeof(*float2_buffers)));
        }
        gpuErrChk(cudaMalloc(&_ir_byte_buf, sizeof(char) * _ir_signal->getByteCount()));
        cufftPlan1d(&_plan_c2c, _fft_size, CUFFT_C2C, 1);

        cufftPlan1d(&_plan_c2r, _fft_size, CUFFT_C2R, 1);
        gpuErrChk(cudaStreamCreate(&_stream_right));
    }

    void deallocateBuffers() {
        for (size_t i = 0; i < sizeof(float2_buffers) / sizeof(*float2_buffers); i++) {
            gpuErrChk(cudaFree(*float2_buffers[i]));
        }
        gpuErrChk(cudaFree(_ir_byte_buf));
        cufftDestroy(_plan_c2c);
        cufftDestroy(_plan_c2r);
        gpuErrChk(cudaStreamDestroy(_stream_right));
    }

    cudaStream_t _process(cudaStream_t stream, float* dst, const float* src, cudaStreamCaptureStatus capture_status) override {
        cufftSetStream(_plan_c2c, stream);
        cufftSetStream(_plan_c2r, stream);

        // pack stereo channels into real and img part of complex struct and perform fft both simultaneously
        // https://web.archive.org/web/20180312110051/http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM
        // since the type cufftComplex is a struct with two float members, we can simply copy our float2 buffer into the cufftComplex buffer
        IMemCpyNode::launchOrRecord1D(_sig_fft_packed, src, sizeof(float2), _n_proc_frames, cudaMemcpyDeviceToDevice, stream, _src_node, capture_status);
        cufftExecC2C(_plan_c2c, _sig_fft_packed, _sig_fft_packed, CUFFT_FORWARD);
        ccc_unpackCto2C<<<4, 768, 0, stream>>>(_sigFFT.left, _sigFFT.right, _sig_fft_packed, _fft_size);

        // Convolution (Colplex Multiplication in Frequency Domain) (scaling is needed to retain unity gain in time domain after inverse fft)
        ccc_complexMultiplyAndScale<<<4, 768, 0, stream>>>(_sigFFT.left, _sigFFT.left, _rirFFT.left, _fft_size_non_redundant, 1.0f / _fft_size);
        ccc_complexMultiplyAndScale<<<4, 768, 0, stream>>>(_sigFFT.right, _sigFFT.right, _rirFFT.right, _fft_size_non_redundant, 1.0f / _fft_size);

        // Inverse FFT
        // using inplace transform to avoid allocating additional buffers
        // output is _fft_size * float contiguous real signal, with the other half of the complex buffer being irrelevant
        cufftExecC2R(_plan_c2r, _sigFFT.left, (cufftReal*)_sigFFT.left);
        cufftExecC2R(_plan_c2r, _sigFFT.right, (cufftReal*)_sigFFT.right);

        if (_force_wet_mix) {
            // if the fx should always produce a 100% wet output (e.g. amp cab ir), we can skip the mixing step and write the sum of the output and residual directly to the dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)f2fff2_pointwiseAdd, new void*[5]{&dst, &_sigFFT.left, &_sigFFT.right, &_residual, &_n_proc_frames}, stream, _dest_node, capture_status);
        } else {
            // combine convolution output with residual and write to dst buffer
            f2fff2_pointwiseAdd<<<1, _n_proc_frames, 0, stream>>>(_wet, (float*)_sigFFT.left, (float*)_sigFFT.right, _residual, _n_proc_frames);
            // mix dry and wet signal and write to dst buffer
            IKernelNode::launchOrRecord(1, _n_proc_frames, 0, (void*)f2f2f2_mix, new void*[5]{&dst, &src, &_wet, &_n_proc_frames, &_mix_ratio}, stream, _dest_node, capture_status);
        }

        return stream;
    }

   public:
    FxConvFd2c2(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) : FxConvFd("FxConvFd2c2", ir_signal, max_ir_size, ir_db_scale, force_wet_mix) {
        if (_ir_signal->getChannelCount() < 2) {
            throw std::runtime_error("IR file given to {} has less than two channels. Stereo IRs are required for this fx.");
        } else if (_ir_signal->getChannelCount() > 2) {
            spdlog::warn("IR file given to {} is not stereo. Only the first two channels will be used.", _name);
        }
        _n_in_channels = 2;
        _n_out_channels = 2;
    }

    ~FxConvFd2c2() {}

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        gpuErrChk(cudaMemsetAsync(_ir_byte_buf, 0, sizeof(char) * _ir_signal->getByteCount(), stream));
        gpuErrChk(cudaMemsetAsync(_residual, 0, sizeof(float2) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_rir_fft_packed, 0, sizeof(cufftComplex) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_sig_fft_packed, 0, sizeof(cufftComplex) * _fft_size, stream));

        // TODO: Remove and fix fft buffer sizes
        gpuErrChk(cudaMemsetAsync(_rirFFT.left, 0, sizeof(cufftComplex) * _fft_size, stream));
        gpuErrChk(cudaMemsetAsync(_rirFFT.right, 0, sizeof(cufftComplex) * _fft_size, stream));

        gpuErrChk(cudaMemcpyAsync(_ir_byte_buf, _ir_signal->getDataPtrConst(), sizeof(char) * _ir_signal->getByteCount(), cudaMemcpyHostToDevice, stream));
        // pack stereo channels into real and img part of complex struct and perform fft both simultaneously
        // https://web.archive.org/web/20180312110051/http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM
        pcm_to_float2((float2*)_rir_fft_packed, _ir_byte_buf, getMinIROrFFTSize(), _ir_signal->getChannelCount(), _ir_signal->getBitDepthValue(), stream);
        cufftSetStream(_plan_c2c, stream);
        cufftExecC2C(_plan_c2c, _rir_fft_packed, _rir_fft_packed, CUFFT_FORWARD);
        ccc_unpackCto2C<<<4, 768, 0, stream>>>(_rirFFT.left, _rirFFT.right, _rir_fft_packed, _fft_size);

        // irs are usually normalized to max amplitude and therefore much too loud. We scale them down to avoid clipping
        f2_scale<<<4, 768, 0, stream>>>(_rirFFT.left, _fft_size, getIRAttenuationFactor());
        f2_scale<<<4, 768, 0, stream>>>(_rirFFT.right, _fft_size, getIRAttenuationFactor());
        gpuErrChk(cudaMemsetAsync(_residual, 0, sizeof(float2) * _fft_size, stream));
        return stream;
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        // add convolution output to residual buffer and clear the unused residual tail
        // can't use inplace processing for rewinding, because when kernel is strided (thread count < fft size) or when using memcopy, the out-of-order execution of the threads could cause recursive reads of already processed data
        // which is why I'm missusing the _wet buffer as a temporary buffer for the sum of the output and residual
        f2fff2_pointwiseAdd<<<4, 768, 0, stream>>>(_wet, ((float*)_sigFFT.left) + _n_proc_frames, ((float*)_sigFFT.right) + _n_proc_frames, _residual + _n_proc_frames, _fft_size - _n_proc_frames);
        gpuErrChk(cudaMemcpyAsync(_residual, _wet, sizeof(float2) * _fft_size - _n_proc_frames, cudaMemcpyDeviceToDevice, stream));
        gpuErrChk(cudaMemsetAsync(_residual + _fft_size - _n_proc_frames, 0, sizeof(float2) * _n_proc_frames, stream));

        // clear the part of _sig_fft_packed that is not overwritten by the input
        gpuErrChk(cudaMemsetAsync(_sig_fft_packed + _n_proc_frames, _nSharedMem, sizeof(cufftComplex) * (_fft_size - _n_proc_frames), stream));
        return stream;
    }

    GpuFx* clone() override {
        return new FxConvFd2c2(_ir_signal->clone(), _fft_size, _ir_db_scale, _force_wet_mix);
    }
};

IGpuFx* IGpuFx::createConv2i2(IPCMSignal* ir_signal, size_t max_ir_size, int ir_db_scale, bool force_wet_mix) {
    return new FxConvFd2c2(ir_signal, max_ir_size, ir_db_scale, force_wet_mix);
}