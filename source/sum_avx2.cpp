#include "sum_avx2.hpp"

double sum_avx2(const std::vector<double>& vec)
{
    const int size = vec.size();
    const double* data = vec.data();

    // Initialize accumulators to zero
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();

    // Process two elements at a time using AVX2 instructions
    for (int i = 0; i < size; i += 4) {
        __m256d v1 = _mm256_load_pd(&data[i]);
        __m256d v2 = _mm256_load_pd(&data[i+4]);
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
