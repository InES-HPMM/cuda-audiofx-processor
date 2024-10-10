#include <condition_variable>
#include <iostream>
#include <log.hpp>
#include <mutex>
#include <thread>

#include "block_buffer.cuh"
#include "gpu.cuh"

using namespace std;

class BlockBuffer : public IBlockBuffer {
   private:
    float** _blocks;
    float** _dev_ptrs;
    float* _memory;
    size_t _read_index;
    size_t _proc_index;
    size_t _write_index;
    size_t _n_channels;
    size_t _n_samples;
    size_t _block_size;
    size_t _block_count;
    size_t _block_count_mask;

    bool _is_full;
    bool _is_empty;
    bool _has_proc;

    mutex _mutex;
    condition_variable _cond_var_proc;
    condition_variable _cond_var_read;
    condition_variable _cond_var_write;

    void incrementReadIndex() {
        lock_guard<mutex> lock(_mutex);
        _read_index = (_read_index + 1) & _block_count_mask;
        _is_empty = _read_index == _proc_index;
        _is_full = false;
        // _cond_var_write.notify_all();
    }

    void incrementWriteIndex() {
        lock_guard<mutex> lock(_mutex);
        _write_index = (_write_index + 1) & _block_count_mask;
        _has_proc = true;
        _is_full = _read_index == _write_index;
        // _cond_var_proc.notify_one();
    }

    void incrementProcIndex() {
        lock_guard<mutex> lock(_mutex);
        _proc_index = (_proc_index + 1) & _block_count_mask;
        _is_empty = false;
        _has_proc = _write_index != _proc_index;
        // _cond_var_read.notify_all();
    }

   public:
    BlockBuffer(size_t n_samples, size_t n_channels, size_t block_count, size_t init_fill_count)
        : _n_samples(n_samples), _n_channels(n_channels), _block_count(block_count), _read_index(0), _write_index(0), _proc_index(0), _is_full(false), _is_empty(true), _has_proc(false) {
        int power_of_two;
        for (power_of_two = 1; 1 << power_of_two < _block_count; power_of_two++);

        _block_count_mask = (1 << power_of_two) - 1;
        _block_size = _n_samples * _n_channels;
        _blocks = new float*[_block_count];
        _dev_ptrs = new float*[_block_count];
        _memory = new float[_block_count * _block_size];
        gpuErrChk(cudaHostAlloc(&_memory, _block_count * _block_size * sizeof(float), cudaHostAllocMapped));
        for (size_t i = 0; i < _block_count; i++) {
            _blocks[i] = _memory + i * _block_size;
            gpuErrChk(cudaHostGetDevicePointer(&_dev_ptrs[i], _blocks[i], 0));
        }

        for (size_t i = 0; i < min(block_count, init_fill_count); i++) {
            incrementWriteIndex();
            incrementProcIndex();
        }
    }

    ~BlockBuffer() {
        for (size_t i = 0; i < _block_count; i++) {
            delete[] _blocks[i];
        }
        delete[] _blocks;
    }

    bool tryReadBlock(float** buffers, bool deinterleave = false) {
        if (_is_empty) {
            Log::warn("BlockBuffer", "Trying to read from empty buffer");
            _cond_var_write.notify_all();
            return false;
        }
        if (deinterleave) {
            float* block = _blocks[_read_index];
            for (size_t c = 0; c < _n_channels; c++) {
                for (size_t s = 0; s < _n_samples; s++) {
                    buffers[c][s] = block[s * _n_channels + c];
                }
            }
        } else {
            for (size_t i = 0; i < _n_channels; i++) {
                memcpy(buffers[i], _blocks[_read_index] + i * _n_samples, _n_samples * sizeof(float));
            }
        }
        incrementReadIndex();
        return true;
    }

    bool tryWriteBlock(const float* const* buffers, bool interleave = false) {
        if (_is_full) {
            Log::warn("BlockBuffer", "Trying to write to full buffer");
            _cond_var_read.notify_all();
            return false;
        }
        if (interleave) {
            float* block = _blocks[_write_index];
            for (size_t c = 0; c < _n_channels; c++) {
                for (size_t s = 0; s < _n_samples; s++) {
                    block[s * _n_channels + c] = buffers[c][s];
                }
            }
        } else {
            for (size_t i = 0; i < _n_channels; i++) {
                memcpy(_blocks[_write_index] + i * _n_samples, buffers[i], _n_samples * sizeof(float));
            }
        }
        incrementWriteIndex();
        return true;
    }

    float* getReadPointer() {
        unique_lock<mutex> lock(_mutex);
        if (_is_empty) {
            return nullptr;
        }
        return _blocks[_read_index];
    }

    float* getWritePointer() {
        unique_lock<mutex> lock(_mutex);
        if (_is_full) {
            return nullptr;
        }
        return _blocks[_write_index];
    }

    bool tryGetProcPointer(float** ptr) {
        if (!_has_proc) {
            return false;
        }
        *ptr = _dev_ptrs[_proc_index];
        // gpuErrChk(cudaHostGetDevicePointer(ptr, _blocks[_proc_index], 0));
        return true;
    }

    float* getReadPointerBlocking() {
        unique_lock<mutex> lock(_mutex);
        if (_is_empty) {
            _cond_var_read.wait(lock, [this] { return !_is_empty; });
        }
        float* dev_ptr = _dev_ptrs[_proc_index];
        // gpuErrChk(cudaHostGetDevicePointer(&dev_ptr, _blocks[_read_index], 0));
        return dev_ptr;
    }

    float* getWritePointerBlocking() {
        unique_lock<mutex> lock(_mutex);
        if (_is_full) {
            _cond_var_write.wait(lock, [this] { return !_is_full; });
        }
        float* dev_ptr = _dev_ptrs[_proc_index];
        // float* dev_ptr;
        // gpuErrChk(cudaHostGetDevicePointer(&dev_ptr, _blocks[_write_index], 0));
        return dev_ptr;
    }

    float* getProcPointerBlocking() {
        unique_lock<mutex> lock(_mutex);
        if (!_has_proc) {
            _cond_var_proc.wait(lock, [this] { return _has_proc; });
        }
        float* dev_ptr = _dev_ptrs[_proc_index];
        // float* dev_ptr;
        // gpuErrChk(cudaHostGetDevicePointer(&dev_ptr, _blocks[_proc_index], 0));
        return dev_ptr;
    }

    bool tryAdvanceReadPointer() {
        if (_is_empty) {
            return false;
        }
        incrementReadIndex();
        return true;
    }

    bool tryAdvanceWritePointer() {
        if (_is_full) {
            return false;
        }
        incrementWriteIndex();
        return true;
    }

    bool tryAdvanceProcPointer() {
        if (!_has_proc) {
            return false;
        }
        incrementProcIndex();
        return true;
    }

    float* getMemoryPointer() {
        float* dev_ptr;
        gpuErrChk(cudaHostGetDevicePointer(&dev_ptr, _memory, 0));
        return dev_ptr;
    }
    size_t getProcOffset() {
        return _proc_index * _block_size;
    }
};

IBlockBuffer* IBlockBuffer::createInstance(size_t n_samples, size_t n_channels, size_t block_count, size_t init_fill_count) {
    return new BlockBuffer(n_samples, n_channels, block_count, init_fill_count);
}