#include "helpers.hpp"
#include <fmt/core.h>

handle_ptr make_unique_handle_real()
{
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    assert(DFTI_NO_ERROR == DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, in.size()));
    assert(DFTI_NO_ERROR == DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    assert(DFTI_NO_ERROR == DftiCommitDescriptor(handle));

    return handle_ptr(handle);
}

handle_ptr make_unique_handle_complex()
{
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    assert(DFTI_NO_ERROR == DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, in.size()));
    assert(DFTI_NO_ERROR == DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    assert(DFTI_NO_ERROR == DftiCommitDescriptor(handle));

    return handle_ptr(handle);
}

double random_double()
{
    static std::default_random_engine engine;
    static std::uniform_real_distribution distr(
            std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::max()
    );
    return distr(engine);
}

std::vector<double> make_double_vector(std::size_t size)
{
    std::vector<double> out(size);
    std::generate_n(std::back_inserter(out), size, [] { return random_double(); });
    return out;
}

std::vector<std::complex<double>> make_complex_vector(std::size_t size)
{
    std::vector<std::complex<double>> out(size);
    std::generate_n(std::back_inserter(out), size, []() -> std::complex<double> {
            return { random_double(), random_double() };
    });
    return out;
}

double sum_avx2(const std::vector<double>& vec)
{
    const int size = vec.size();
    const double* data = vec.data();

    // Initialize accumulators to zero
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();

    // Process two elements at a time using AVX2 instructions
    for (int i = 0; i < size; i += 4) {
        __m256d v1 = _mm256_loadu_pd(&data[i]);
        __m256d v2 = _mm256_loadu_pd(&data[i+4]);
        sum1 = _mm256_add_pd(sum1, v1);
        sum2 = _mm256_add_pd(sum2, v2);
    }

    // Add the partial sums together using AVX2 instructions
    sum1 = _mm256_add_pd(sum1, sum2);
    sum1 = _mm256_hadd_pd(sum1, sum1);
    sum1 = _mm256_hadd_pd(sum1, sum1);

    // Extract the result from the AVX2 register
    alignas(32) double result[4];
    _mm256_store_pd(result, sum1);
    return result[0] + result[2];
}
