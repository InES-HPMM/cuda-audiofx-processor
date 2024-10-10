
#include <buffer.cuh>
#include <enums.hpp>
#include <string>

class TrtEngine {
   public:
    static TrtEngine* create(std::string onnx_model_path, std::string trt_model_dir, TrtEnginePrecision precision);

    virtual Buffer* getInputBuffer(const size_t index) = 0;
    virtual Buffer* getOutputBuffer(const size_t index) = 0;
    virtual void configure(size_t process_buffer_size, size_t n_input_channels, size_t n_output_channels) = 0;
    virtual void allocate() = 0;
    virtual void setup(cudaStream_t stream) = 0;
    virtual void inference(cudaStream_t stream) = 0;
    virtual void deallocate() = 0;

    virtual TrtEngine* clone() = 0;
};