#pragma once

#include <mkl_dfti.h>
#include <immintrin.h>

#include <vector>
#include <memory>
#include <algorithm>

#include <random>
#include <cassert>
#include <complex>
#include <cstddef>

struct dfti_handle_deleter
{
    void operator()(DFTI_DESCRIPTOR_HANDLE handle)
    {
        [[maybe_unused]] auto status = DftiFreeDescriptor(&handle);
        assert(DFTI_NO_ERROR == status);
    }
};

using handle_ptr = std::unique_ptr<DFTI_DESCRIPTOR, dfti_handle_deleter>;

handle_ptr make_unique_handle_complex();
handle_ptr make_unique_handle_real();

double random_double();
std::vector<double> make_double_vector(std::size_t size);
std::vector<std::complex<double>> make_complex_vector(std::size_t size);

double sum_avx2(const std::vector<double> & vec);
