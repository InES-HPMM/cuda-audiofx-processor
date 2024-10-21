
#include <stddef.h>
class RingBuffer {
   public:
    static RingBuffer* create(size_t block_size, size_t n_blocks, size_t n_init_block_filled = 0);
    virtual ~RingBuffer() {}

    virtual size_t getReadSpace() const = 0;
    virtual size_t getWriteSpace() const = 0;
    virtual size_t read(float* dst, size_t n) = 0;
    virtual size_t write(const float* src, size_t n) = 0;

    virtual size_t getReadPtr(float** read_ptr, size_t n) = 0;

    virtual size_t getWritePtr(float** write_ptr, size_t n) = 0;
    virtual size_t advanceReadIndex(size_t n) = 0;
    virtual size_t advanceWriteIndex(size_t n) = 0;
};