

#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cub/cub.cuh>
#include <log.hpp>
#include <vector>

#include "rmsd.cuh"
#include "spdlog/spdlog.h"

template <typename T>
struct SquaredDifference {
    __host__ __device__
        T
        operator()(thrust::tuple<T, T> tuple) const noexcept { return (thrust::get<0>(tuple) - thrust::get<1>(tuple)) * (thrust::get<0>(tuple) - thrust::get<1>(tuple)); }
};
template <typename T, typename V>
class RootMean {
   private:
   public:
    __host__ __device__
        T
        operator()(T sum, V n) const noexcept { return sqrt(sum / (T)n); }
};

void launchThrustRMSDCalcAsync(thrust::device_ptr<float> v1, thrust::device_ptr<float> v2, thrust::device_ptr<float> res, size_t n, std::vector<cudaStream_t>* streams, std::vector<float>* vector_sizes) {
    // create transform iterator that applies the squared difference functor to each element of the zipped vectors
    auto first = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v1, v2)), SquaredDifference<float>());
    auto last = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v1 + n, v2 + n)), SquaredDifference<float>());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // determine temporary device storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, first, res, n);
    // allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // sum the output vector of the squared differences transform iterator and store the output in the provided result pointer
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, first, res, n, stream);

    // store the vector size for subsequent root mean calculation (using another transform here before stream synchronization would enforce an early stream syncronization prevent all parallelism)
    // startung with thrust 1.16, this could be avoided by using the thrust::cuda::par_nosync.on(stream) execution policy
    vector_sizes->push_back(n);
    // store the stream for later synchronization
    streams->push_back(stream);
}

/// @brief calculates the RMSD beween the two buffers at \param max_offset in both directions and returns the minimal RMSD as well as the offset of the minimal RMSD
/// @param buf1 fist buffer
/// @param buf2 second buffer
/// @param n number of elements in the buffers
/// @param max_offset maximal offset between the two buffers in both directions (e.g max_offset of 2 will calculate the RMSD for the buffers at offset -2, -1, 0, 1, 2)
/// @param minimal_rmsd_offset return value pointer for the offset of the minimal RMSD. negative values indicate that the first buffer is shifted to the left, positive values indicate that the second buffer is shifted to the left
/// @return the minimal RMSD
float getMinimalRMSD(const float* buf1, const float* buf2, const size_t n, const size_t max_offset, int* minimal_rmsd_offset) {
    thrust::device_vector<float> dev_vec_1(buf1, buf1 + n);
    thrust::device_vector<float> dev_vec_2(buf2, buf2 + n);

    std::vector<cudaStream_t> streams;
    std::vector<float> vector_sizes;
    thrust::device_vector<float> results(max_offset * 2 + 1, 0);
    for (size_t offset = 0; offset <= max_offset; offset++) {
        // starting at max offset to ensure propper ordering in the result vector (-max_offset, ..., 0, ..., max_offset)
        auto vector_offset = max_offset - offset;
        launchThrustRMSDCalcAsync(dev_vec_1.data() + vector_offset, dev_vec_2.data(), results.data() + offset, n - vector_offset, &streams, &vector_sizes);
    }

    for (size_t offset = 1; offset <= max_offset; offset++) {
        launchThrustRMSDCalcAsync(dev_vec_1.data(), dev_vec_2.data() + offset, results.data() + offset + max_offset, n - offset, &streams, &vector_sizes);
    }

    std::for_each(streams.begin(), streams.end(), cudaStreamSynchronize);
    std::for_each(streams.begin(), streams.end(), cudaStreamDestroy);

    thrust::device_vector<float> dev_vector_sizes(vector_sizes.begin(), vector_sizes.end());
    // calculate the root mean of the sums of squared differences
    thrust::transform(results.begin(), results.end(), dev_vector_sizes.begin(), results.begin(), RootMean<float, size_t>());

    // get the smalles rmsd with its offset
    auto min_iter = thrust::min_element(results.begin(), results.end());
    if (minimal_rmsd_offset != nullptr) {
        size_t min_index = min_iter - results.begin();
        *minimal_rmsd_offset = min_index - max_offset;
    }

    return *min_iter;
}

bool testMinimalRMSDIntegrity() {
    size_t n_samples = 20;
    float expected_minumal_rmsd = 0.24964974820613861083984375;
    float expected_minimal_rmsd_offset = 8;

    float* expected = new float[n_samples]{0.55, 0.72, 0.6, 0.54, 0.42, 0.65, 0.44, 0.89, 0.96, 0.38, 0.79, 0.53, 0.57, 0.93, 0.07, 0.09, 0.02, 0.83, 0.78, 0.87};
    float* actual = new float[n_samples]{0.98, 0.8, 0.46, 0.78, 0.12, 0.64, 0.14, 0.94, 0.52, 0.41, 0.26, 0.77, 0.46, 0.57, 0.02, 0.62, 0.61, 0.62, 0.94, 0.68};

    int minimal_rmsd_offset = 0;
    float minimal_rmsd = getMinimalRMSD(expected, actual, n_samples, 9, &minimal_rmsd_offset);
    if (minimal_rmsd != expected_minumal_rmsd) {
        spdlog::error("Minimal RMSD is not correct: {} != {}", minimal_rmsd, expected_minumal_rmsd);
        return false;
    }
    if (minimal_rmsd_offset != expected_minimal_rmsd_offset) {
        spdlog::error("Minimal RMSD offset is not correct: {} != {}", minimal_rmsd_offset, expected_minimal_rmsd_offset);
        return false;
    }
    spdlog::info("Minimal RMSD integrity ensured");
    delete expected;
    delete actual;
    return true;
}