

#include <cmath>  // pow, sin
#include <cuda_ext.cuh>
#include <kernels.cuh>
#include <operators.cuh>

#include "gpu_fx.cu"

__global__ static void ff_fir(float* dst, const float* src, const float* coef, const size_t n_samples, const size_t n_channels, const size_t degree) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_samples; s += stride.x) {
        float output = 0;
        for (size_t i = 0; i < degree; i++) {
            output += coef[i] * (*(src + s - i * n_channels));
        }
        dst[s] = output;
    }
}

__global__ static void ff_iir(float* dst, float* src, const float* coef, const size_t n_samples, const size_t n_channels, const size_t degree) {
    auto stride = gridDim * blockDim;
    auto offset = blockDim * blockIdx + threadIdx;

    for (auto s = offset.x; s < n_samples; s += stride.x) {
        for (size_t i = 1; i < degree; i++) {
            src[s] += coef[i] * (*(src + s - i * n_channels));
        }
        dst[s] = src[s];
    }
}

class BiquadParam : public IBiquadParam {
   protected:
    size_t _fir_degree = 3;
    size_t _iir_degree = 3;
    double _frequency;
    double _gain_dB;
    double _quality;
    SampleRate _samplerate;

    double b0 = 0;
    double b1 = 0;
    double b2 = 0;
    double a0 = 0;
    double a1 = 0;
    double a2 = 0;

    // Parameters defined in
    // https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    double GetA() const { return pow(10.0, _gain_dB / 40.0); };
    double GetOmega0() const { return 2.0 * MATH_PI * _frequency / as_int(_samplerate); };
    double GetAlpha(const double omega_0) const { return sin(omega_0) / (2.0 * _quality); };
    double GetCosW(const double omega_0) const { return cos(omega_0); };

   public:
    BiquadParam(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : _frequency(frequency), _gain_dB(gain_dB), _quality(quality), _samplerate(sampleRate) {};

    size_t GetFirDegree() const { return _fir_degree; };
    size_t GetIirDegree() const { return _iir_degree; };
    virtual void getCoefficients(float* fir_coef, float* iir_coef) {
        fir_coef[0] = b0 / a0;
        fir_coef[1] = b1 / a0;
        fir_coef[2] = b2 / a0;
        iir_coef[0] = 0;
        iir_coef[1] = -a1 / a0;
        iir_coef[2] = -a2 / a0;
    }
    virtual void setFrequency(double frequency) { _frequency = frequency; };
    virtual void setGain(double gain) { _gain_dB = gain; };
    virtual void setQuality(double quality) { _quality = quality; };
};

class BiquadPeakParams : public BiquadParam {
   public:
    BiquadPeakParams(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : BiquadParam(frequency, gain_dB, quality, sampleRate) {}

    void getCoefficients(float* fir_coef, float* iir_coef) override {
        // Peak filter
        double a = this->GetA();
        double omega_0 = this->GetOmega0();
        double alpha = this->GetAlpha(omega_0);
        double cosw = this->GetCosW(omega_0);

        b0 = 1.0 + alpha * a;
        b1 = -2.0 * cosw;
        b2 = 1.0 - alpha * a;
        a0 = 1.0 + alpha / a;
        a1 = -2.0 * cosw;
        a2 = 1.0 - alpha / a;

        BiquadParam::getCoefficients(fir_coef, iir_coef);
    }
    IBiquadParam* clone() {
        return new BiquadPeakParams(_frequency, _gain_dB, _quality, _samplerate);
    }
};

class BiquadLowPassParams : public BiquadParam {
   public:
    BiquadLowPassParams(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : BiquadParam(frequency, gain_dB, quality, sampleRate) {}

    void getCoefficients(float* fir_coef, float* iir_coef) override {
        // Low pass filter
        double a = this->GetA();
        double omega_0 = this->GetOmega0();
        double alpha = this->GetAlpha(omega_0);
        double cosw = this->GetCosW(omega_0);

        b0 = (1.0 - cosw) / 2.0;
        b1 = 1.0 - cosw;
        b2 = (1.0 - cosw) / 2.0;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosw;
        a2 = 1.0 - alpha;
        BiquadParam::getCoefficients(fir_coef, iir_coef);
    }
    IBiquadParam* clone() {
        return new BiquadLowPassParams(_frequency, _gain_dB, _quality, _samplerate);
    }
};

class BiquadHighPassParams : public BiquadParam {
   public:
    BiquadHighPassParams(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : BiquadParam(frequency, gain_dB, quality, sampleRate) {}

    void getCoefficients(float* fir_coef, float* iir_coef) override {
        // High pass filter
        double a = this->GetA();
        double omega_0 = this->GetOmega0();
        double alpha = this->GetAlpha(omega_0);
        double cosw = this->GetCosW(omega_0);

        double b0 = (1.0 + cosw) / 2.0;
        double b1 = -(1.0 + cosw);
        double b2 = (1.0 + cosw) / 2.0;
        double a0 = 1.0 + alpha;
        double a1 = -2.0 * cosw;
        double a2 = 1.0 - alpha;
        BiquadParam::getCoefficients(fir_coef, iir_coef);
    }
    IBiquadParam* clone() {
        return new BiquadHighPassParams(_frequency, _gain_dB, _quality, _samplerate);
    }
};

class BiquadLowShelfParams : public BiquadParam {
   public:
    BiquadLowShelfParams(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : BiquadParam(frequency, gain_dB, quality, sampleRate) {}

    void getCoefficients(float* fir_coef, float* iir_coef) override {
        // Low shelf filter
        double a = this->GetA();
        double omega_0 = this->GetOmega0();
        double alpha = this->GetAlpha(omega_0);
        double cosw = this->GetCosW(omega_0);
        double sqrt_a = sqrt(a);

        double beta = sqrt_a / _quality;
        double b0 = a * ((a + 1) - (a - 1) * cosw + 2 * sqrt_a * beta);
        double b1 = 2 * a * ((a - 1) - (a + 1) * cosw);
        double b2 = a * ((a + 1) - (a - 1) * cosw - 2 * sqrt_a * beta);
        double a0 = (a + 1) + (a - 1) * cosw + 2 * sqrt_a * beta;
        double a1 = -2 * ((a - 1) + (a + 1) * cosw);
        double a2 = (a + 1) + (a - 1) * cosw - 2 * sqrt_a * beta;
        BiquadParam::getCoefficients(fir_coef, iir_coef);
    }
    IBiquadParam* clone() {
        return new BiquadLowShelfParams(_frequency, _gain_dB, _quality, _samplerate);
    }
};

class BiquadHighShelfParams : public BiquadParam {
   public:
    BiquadHighShelfParams(const double frequency, const double gain_dB, const double quality, const SampleRate sampleRate)
        : BiquadParam(frequency, gain_dB, quality, sampleRate) {}

    void getCoefficients(float* fir_coef, float* iir_coef) override {
        // High shelf filter
        double a = this->GetA();
        double omega_0 = this->GetOmega0();
        double alpha = this->GetAlpha(omega_0);
        double cosw = this->GetCosW(omega_0);
        double sqrt_a = sqrt(a);

        double beta = sqrt_a / _quality;
        double b0 = a * ((a + 1) + (a - 1) * cosw + 2 * sqrt_a * beta);
        double b1 = -2 * a * ((a - 1) + (a + 1) * cosw);
        double b2 = a * ((a + 1) + (a - 1) * cosw - 2 * sqrt_a * beta);
        double a0 = (a + 1) - (a - 1) * cosw + 2 * sqrt_a * beta;
        double a1 = 2 * ((a - 1) - (a + 1) * cosw);
        double a2 = (a + 1) - (a - 1) * cosw - 2 * sqrt_a * beta;
        BiquadParam::getCoefficients(fir_coef, iir_coef);
    }
    IBiquadParam* clone() {
        return new BiquadHighShelfParams(_frequency, _gain_dB, _quality, _samplerate);
    }
};

IBiquadParam*
IBiquadParam::create(BiquadType type, double frequency, double gain_dB, double quality, SampleRate sample_rate) {
    switch (type) {
        case BiquadType::PEAK:
            return new BiquadPeakParams(frequency, gain_dB, quality, sample_rate);
        case BiquadType::LOWPASS:
            return new BiquadLowPassParams(frequency, gain_dB, quality, sample_rate);
        case BiquadType::HIGHPASS:
            return new BiquadHighPassParams(frequency, gain_dB, quality, sample_rate);
        case BiquadType::LOWSHELF:
            return new BiquadLowShelfParams(frequency, gain_dB, quality, sample_rate);
        case BiquadType::HIGHSHELF:
            return new BiquadHighShelfParams(frequency, gain_dB, quality, sample_rate);

        default:
            throw std::runtime_error("Not Implemented");
    }
}

class BiquadBand {
   private:
    float* _fir_coef_dev = nullptr;
    float* _iir_coef_dev = nullptr;
    float* _fir_coef_host = nullptr;
    float* _iir_coef_host = nullptr;
    float* _proc_buffer = nullptr;
    float* _proc_buffer_start_ptr = nullptr;

    size_t _history_size;
    size_t _proc_buffer_size;
    size_t _n_channels;
    size_t _n_frames;

    BiquadParam* _params;

   public:
    BiquadBand(BiquadParam* params) : _params(params), _proc_buffer_size(0), _history_size(0) {
        _fir_coef_host = new float[_params->GetFirDegree()];
        _iir_coef_host = new float[_params->GetIirDegree()];
    }

    ~BiquadBand() {
        if (_fir_coef_host) delete[] _fir_coef_host;
        if (_iir_coef_host) delete[] _iir_coef_host;
        delete _params;
    }

    float* getProcBuffer() { return _proc_buffer_start_ptr; }
    BiquadParam* getParams() { return _params; }

    void configure(size_t n_frames, size_t n_channels) {
        _params->getCoefficients(_fir_coef_host, _iir_coef_host);
        _history_size = (max(_params->GetFirDegree(), _params->GetIirDegree()) - 1) * n_channels;
        _proc_buffer_size = n_frames * n_channels + _history_size;
        _n_channels = n_channels;
        _n_frames = n_frames;
    }

    void allocateBuffers() {
        gpuErrChk(cudaMalloc(&_fir_coef_dev, sizeof(float) * _params->GetFirDegree()));
        gpuErrChk(cudaMalloc(&_iir_coef_dev, sizeof(float) * _params->GetIirDegree()));
        gpuErrChk(cudaMalloc(&_proc_buffer, sizeof(float) * _proc_buffer_size));
        _proc_buffer_start_ptr = (float*)_proc_buffer + _history_size;
    }
    void deallocateBuffers() {
        if (_fir_coef_dev) gpuErrChk(cudaFree(_fir_coef_dev));
        if (_iir_coef_dev) gpuErrChk(cudaFree(_iir_coef_dev));
        if (_proc_buffer) gpuErrChk(cudaFree(_proc_buffer));
    }

    void setup(cudaStream_t stream) {
        gpuErrChk(cudaMemcpyAsync(_fir_coef_dev, _fir_coef_host, sizeof(float) * _params->GetFirDegree(), cudaMemcpyHostToDevice, stream));
        gpuErrChk(cudaMemcpyAsync(_iir_coef_dev, _iir_coef_host, sizeof(float) * _params->GetIirDegree(), cudaMemcpyHostToDevice, stream));
        gpuErrChk(cudaMemsetAsync(_proc_buffer, 0, sizeof(float) * _history_size, stream));
    }

    void process(cudaStream_t stream, float* dst) {
        ff_iir<<<1, 1, 0, stream>>>(_proc_buffer_start_ptr, _proc_buffer_start_ptr, _iir_coef_dev, _n_frames * _n_channels, _n_channels, _params->GetIirDegree());
        ff_fir<<<1, _n_frames * _n_channels, 0, stream>>>(dst, _proc_buffer_start_ptr, _fir_coef_dev, _n_frames * _n_channels, _n_channels, _params->GetFirDegree());
    }

    void postProcess(cudaStream_t stream) {
        // rewind proc buffer
        gpuErrChk(cudaMemcpyAsync(_proc_buffer, _proc_buffer + _n_frames * _n_channels, sizeof(float) * _history_size, cudaMemcpyDeviceToDevice, stream));
    }
};

class FxEq : public GpuFx {
   protected:
    size_t _n_channels_default;
    std::vector<BiquadBand*> _bands;

    IMemCpyNode* _src_node = nullptr;
    float* _wet = nullptr;

    void allocateBuffers() override {
        for (auto band : _bands) {
            band->allocateBuffers();
        }
        gpuErrChk(cudaMalloc(&_wet, sizeof(float) * _n_proc_samples));
    }

    void deallocateBuffers() override {
        for (auto band : _bands) {
            band->deallocateBuffers();
        }
        gpuErrChk(cudaFree(_wet));
        if (_src_node) delete _src_node;
    }

    cudaStream_t _process(cudaStream_t stream, float* dst, const float* src, cudaStreamCaptureStatus capture_status) {
        IMemCpyNode::launchOrRecord1D(_bands[0]->getProcBuffer(), src, sizeof(float), _n_proc_samples, cudaMemcpyDeviceToDevice, stream, &_src_node, capture_status);
        for (size_t i = 0; i < _bands.size(); i++) {
            _bands[i]->process(stream, i < _bands.size() - 1 ? _bands[i + 1]->getProcBuffer() : _wet);
        }
        IKernelNode::launchOrRecord(1, _n_proc_samples, 0, (void*)fff_mix, new void*[5]{&dst, &src, &_wet, &_n_proc_samples, &_mix_ratio}, stream, &_mix_node, capture_status);
        return stream;
    }

   public:
    FxEq(std::vector<IBiquadParam*> params, size_t n_channels) : GpuFx("FxEq"), _n_channels_default(n_channels) {
        for (auto param : params) {
            _bands.push_back(new BiquadBand(static_cast<BiquadParam*>(param)));
        }
    }

    ~FxEq() {
        for (auto band : _bands) {
            delete band;
        }
    }

    void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) override {
        if (n_input_channels == 0) {
            n_input_channels = _n_channels_default;
            n_output_channels = _n_channels_default;
        } else if (n_output_channels == 0) {
            n_output_channels = n_input_channels;
        } else if (n_input_channels != n_output_channels) {
            spdlog::warn("{} must have same number of input and output channels. n_output_channels {} is overwritten by n_input_channels {}", _name, n_output_channels, n_input_channels);
            n_output_channels = n_input_channels;
        }
        GpuFx::configure(process_buffer_size, n_input_channels, n_output_channels);
        for (auto band : _bands) {
            band->configure(_n_proc_frames, _n_proc_channels);
        }
    }

    void updateBufferPtrs(cudaGraphExec_t procGraphExec, const BufferRack* dst, const BufferRack* src) override {
        _src_node->updateExecSrcPtr(src->getDataMod(), procGraphExec);
        _mix_node->updateExecKernelParamAt(0, dst->getDataMod(), procGraphExec);
    }

    cudaStream_t setup(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        GpuFx::setup(stream, capture_status);
        for (auto band : _bands) {
            band->setup(stream);
        }
        return stream;
    }

    cudaStream_t process(cudaStream_t stream, const BufferRack* dst, const BufferRack* src, cudaStreamCaptureStatus capture_status) override {
        return _process(stream, dst->getDataMod(), src->getDataMod(), capture_status);
    }

    cudaStream_t postProcess(cudaStream_t stream, cudaStreamCaptureStatus capture_status) override {
        for (auto band : _bands) {
            band->postProcess(stream);
        }
        return stream;
    }

    GpuFx* clone() override {
        std::vector<IBiquadParam*> biquad_params;
        std::transform(_bands.begin(), _bands.end(), std::back_inserter(biquad_params), [](BiquadBand* band) { return band->getParams()->clone(); });
        return new FxEq(biquad_params, _n_channels_default);
    }
};

IGpuFx* IGpuFx::createBiquadEQ(IBiquadParam* param, size_t n_channels) {
    return new FxEq({param}, n_channels);
}
IGpuFx* IGpuFx::createBiquadEQ(std::vector<IBiquadParam*> params, size_t n_channels) {
    return new FxEq(params, n_channels);
}