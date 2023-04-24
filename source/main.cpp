#include "mkl_blas.h"
#include "mkl_cblas.h"
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mkl_service.h>
#include <fmt/format.h>
#include <mkl.h>
#include <ranges>
#include <random>
#include <limits>
#include <algorithm>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <utility>
#include <chrono>

using namespace std::chrono_literals;

constexpr std::int64_t SIZE = 1'000'000;

int main()
{
    std::vector<float> A;
    A.resize(SIZE);
    std::vector<float> B;
    B.reserve(SIZE);

    float * mA = (float *)mkl_malloc(SIZE * sizeof(float), 32);
    float * mB = (float *)mkl_malloc(SIZE * sizeof(float), 32);

    for (int i : std::views::iota(0, SIZE))  A[i] = i + .3;
    for (int i : std::views::iota(0, SIZE)) mA[i] = i + .3;

    {
        auto now = std::chrono::steady_clock::now();
        std::ranges::copy(A, std::back_inserter(B));
        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - now;
        fmt::print("std: {}s\n", elapsed.count());
    }
    {
        auto now = std::chrono::steady_clock::now();
        cblas_scopy(SIZE, mA, 1, mB, 1);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - now;
        fmt::print("mkl: {}s\n", elapsed.count());
    }

    mkl_free(mA);
    mkl_free(mB);
}
