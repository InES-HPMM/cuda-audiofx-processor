#include "ringbuffer.hpp"

#include <stddef.h>

#include <cstring>

#include "math_ext.hpp"
#include "spdlog/spdlog.h"
class RingBufferImpl : public RingBuffer {
   private:
    float* _buffer;
    size_t _read_index;
    size_t _write_index;
    size_t _size;
    size_t _size_mask;

   public:
    RingBufferImpl(size_t block_size, size_t n_blocks) : _size(roundUpToPow2(block_size * n_blocks)), _size_mask(_size - 1), _read_index(0), _write_index(0) {
        _buffer = new float[_size];
        spdlog::info("size mask: {}", _size_mask);
        spdlog::info("size mask hex: {:#x}", _size_mask);
    }

    size_t getReadSpace() const override {
        size_t w, r;

        w = _write_index;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        r = _read_index;

        return (r - w) & _size_mask;
    }

    size_t getWriteSpace() const override {
        size_t w, r;

        w = _write_index;
        r = _read_index;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);

        return (r - w - 1) & _size_mask;
    }

    size_t read(float* dst, size_t n) override {
        size_t free_cnt;
        size_t read_end;
        size_t to_read;
        size_t n1, n2;

        if ((free_cnt = getReadSpace()) < n) {
            return 0;
        }

        /* note: relaxed load here, _read_index cannot be
         * modified from writing thread  */
        read_end = _read_index + n;

        if (read_end > _size) {
            n1 = _size - _read_index;
            n2 = read_end & _size_mask;
        } else {
            n1 = n;
            n2 = 0;
        }

        memcpy(dst, &(_buffer[_read_index]), n1 * sizeof(float));
        __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
        _read_index = (_read_index + n1) & _size_mask;

        if (n2) {
            memcpy(dst + n1, &(_buffer[_read_index]), n2 * sizeof(float));
            __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
            _read_index = (_read_index + n2) & _size_mask;
        }

        return n;
    }

    size_t write(const float* src, size_t n) override {
        size_t free_cnt;
        size_t write_end;
        size_t n1, n2;

        // spdlog::info("write_index: {}", _write_index);
        if ((free_cnt = getWriteSpace()) < n) {
            return 0;
        }

        /* note: relaxed load here, _write_index cannot be
         * modified from reading thread  */
        write_end = _write_index + n;

        if (write_end > _size) {
            n1 = _size - _write_index;
            n2 = write_end & _size_mask;
        } else {
            n1 = n;
            n2 = 0;
        }
        // spdlog::info("n1: {}", n1);

        memcpy(&(_buffer[_write_index]), src, n1 * sizeof(float));
        __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
        _write_index = (_write_index + n1) & _size_mask;

        if (n2) {
            memcpy(&(_buffer[_write_index]), src + n1, n2 * sizeof(float));
            __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
            _write_index = (_write_index + n2) & _size_mask;
        }
        // spdlog::info("write_index: {}", _write_index);

        return n;
    }

    size_t getReadPtr(float** read_ptr, size_t n) override {
        if (getReadSpace() < n) {
            return 0;
        }

        *read_ptr = &(_buffer[_read_index]);
        return n;
    }

    size_t getWritePtr(float** write_ptr, size_t n) override {
        if (getWriteSpace() < n) {
            return 0;
        }

        *write_ptr = &(_buffer[_write_index]);
        return n;
    }

    void advanceReadIndex(size_t n) override {
        size_t tmp = (_read_index + n) & _size_mask;
        __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
        _read_index = tmp;
    }

    void advanceWriteIndex(size_t n) override {
        size_t tmp = (_write_index + n) & _size_mask;
        __atomic_thread_fence(__ATOMIC_RELEASE); /* ensure pointer increment happens after copy */
        _write_index = tmp;
    }
};

RingBuffer* RingBuffer::create(size_t block_size, size_t n_blocks) {
    return new RingBufferImpl(block_size, n_blocks);
}