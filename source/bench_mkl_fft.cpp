#include <mkl_dfti.h>
#include <benchmark/benchmark.h>
#include <boost/range/numeric.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "helpers.hpp"
#include "sum_avx2.hpp"

#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <cstdint>
#include <cassert>

std::vector<std::complex<double>> fft_complex(auto&& handle, std::vector<std::complex<double>> & in, std::vector<std::complex<double>> & out)
{
    [[maybe_unused]]
    auto status = DftiComputeForward(handle.get(), in.data(), out.data());

    assert(DFTI_NO_ERROR == status);

    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();

    return out;
}

std::vector<double> fft_real(auto&& handle, std::vector<double> & in, std::vector<double> & out)
{
    [[maybe_unused]]
    auto status = DftiComputeForward(handle.get(), in.data(), out.data());

    assert(DFTI_NO_ERROR == status);

    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();

    return out;
}

template <class ...Args>
void benchmark_mkl_fft_real(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto handle = std::move(std::get<0>(args));
    auto input = std::move(std::get<1>(args));
    auto output = std::move(std::get<2>(args));

    benchmark::DoNotOptimize(input);

    for (auto _ : state)
    {
        [[maybe_unused]] auto fft = fft_real(std::move(handle), input, output);

        benchmark::DoNotOptimize(fft);
        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_mkl_fft_complex(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto handle = std::move(std::get<0>(args));
    auto input = std::move(std::get<1>(args));
    auto output = std::move(std::get<2>(args));

    benchmark::DoNotOptimize(input);

    for (auto _ : state)
    {
        [[maybe_unused]] auto fft = fft_complex(std::move(handle), input, output);

        benchmark::DoNotOptimize(fft);
        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_std_memcpy(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto input = std::move(std::get<0>(args));
    auto output = std::move(std::get<1>(args));

    benchmark::DoNotOptimize(input);
    benchmark::DoNotOptimize(output);

    for (auto _ : state)
    {
        std::memcpy(output.data(), input.data(), input.size());

        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_std_accumulate(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto data = std::move(std::get<0>(args));

    benchmark::DoNotOptimize(data);

    for (auto _ : state)
    {
        auto v = std::accumulate(data.begin(), data.end(), 0.0f);

        benchmark::DoNotOptimize(v);
        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_boost_accumulate(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto data = std::move(std::get<0>(args));

    benchmark::DoNotOptimize(data);

    for (auto _ : state)
    {
        auto v = boost::accumulate(data, 0.0f);

        benchmark::DoNotOptimize(v);
        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_boost_accumulators_mean(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto data = std::move(std::get<0>(args));

    benchmark::DoNotOptimize(data);

    using namespace boost::accumulators;

    for (auto _ : state)
    {
        accumulator_set<double, stats<tag::mean>> acc;

        for (const auto & elem : data)
        {
            acc(elem);
        }

        const auto m = mean(acc);

        benchmark::DoNotOptimize(m);
        benchmark::ClobberMemory();
    }
}

template <class ...Args>
void benchmark_quadratic_mean(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto data = std::move(std::get<0>(args));

    benchmark::DoNotOptimize(data);

    for (auto _ : state)
    {
        auto sum_of_squares = sum_avx2(data);
        auto qm = std::sqrt(sum_of_squares / data.size());

        benchmark::DoNotOptimize(qm);
        benchmark::ClobberMemory();
    }
}

BENCHMARK_CAPTURE(benchmark_mkl_fft_real,     100000,   make_unique_handle_real(), make_double_vector(100'000), make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_mkl_fft_real,     1000000,  make_unique_handle_real(), make_double_vector(1'000'000), make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_mkl_fft_real,     10000000, make_unique_handle_real(), make_double_vector(10'000'000), make_double_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_mkl_fft_complex,  100000,   make_unique_handle_complex(), make_complex_vector(100'000), make_complex_vector(100'000));
BENCHMARK_CAPTURE(benchmark_mkl_fft_complex,  1000000,  make_unique_handle_complex(), make_complex_vector(1'000'000), make_complex_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_mkl_fft_complex,  10000000, make_unique_handle_complex(), make_complex_vector(10'000'000), make_complex_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_std_memcpy,       100000,   make_double_vector(100'000),    make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_std_memcpy,       1000000,  make_double_vector(1'000'000),  make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_std_memcpy,       10000000, make_double_vector(10'000'000), make_double_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_std_accumulate,   100000,   make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_std_accumulate,   1000000,  make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_std_accumulate,   10000000, make_double_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_boost_accumulate, 100000,   make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_boost_accumulate, 1000000,  make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_boost_accumulate, 10000000, make_double_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_boost_accumulators_mean, 100000,   make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_boost_accumulators_mean, 1000000,  make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_boost_accumulators_mean, 10000000, make_double_vector(10'000'000));

BENCHMARK_CAPTURE(benchmark_quadratic_mean, 100000,   make_double_vector(100'000));
BENCHMARK_CAPTURE(benchmark_quadratic_mean, 1000000,  make_double_vector(1'000'000));
BENCHMARK_CAPTURE(benchmark_quadratic_mean, 10000000, make_double_vector(10'000'000));

/*
auto make_data(std::uint64_t size)
{
    std::default_random_engine eng;
    std::uniform_real_distribution<double> dst(
            std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::max()
    );

    std::vector<double> input;
    input.reserve(size);
    std::generate_n(std::back_inserter(input), size, [&]{ return dst(eng); });

    std::vector<double> output;
    output.resize(size);
    return std::make_tuple(input, output);
}

auto make_unique_handle()
{
    struct DFTI_HANDLE_deleter
    {
        void operator()(DFTI_DESCRIPTOR_HANDLE handle)
        {
            DftiFreeDescriptor(&handle);
        }
    };

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, 32);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(handle);

    return std::unique_ptr<DFTI_DESCRIPTOR, DFTI_HANDLE_deleter>(handle);
}

template <class ...Args>
void bench_do_out_of_place_fft(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto handle = std::move(std::get<0>(args));
    auto [input, output] = std::move(std::get<1>(args));

    benchmark::DoNotOptimize(handle);
    benchmark::DoNotOptimize(input);
    benchmark::DoNotOptimize(output);

    for (auto _ : state)
    {
        DftiComputeForward(handle.get(), input.data(), output.data());
    }
}

template <class ...Args>
void bench_std_memcpy(benchmark::State & state, Args &&... _args)
{
    auto args = std::make_tuple(std::move(_args)...);
    auto [input, output] = std::move(std::get<0>(args));

    benchmark::DoNotOptimize(input);
    benchmark::DoNotOptimize(output);

    for (auto _ : state)
    {
        std::memcpy(input.data(), output.data(), input.size());
    }
}

BENCHMARK_CAPTURE(bench_do_out_of_place_fft, fft_10000, make_unique_handle(), make_data(10'000));
BENCHMARK_CAPTURE(bench_do_out_of_place_fft, ftt_100000, make_unique_handle(), make_data(100'000));
BENCHMARK_CAPTURE(bench_do_out_of_place_fft, fft_1000000, make_unique_handle(), make_data(1'000'000));
BENCHMARK_CAPTURE(bench_do_out_of_place_fft, fft_10000000, make_unique_handle(), make_data(10'000'000));

BENCHMARK_CAPTURE(bench_std_memcpy, std_memcpy_10000, make_data(10'000));
BENCHMARK_CAPTURE(bench_std_memcpy, std_memcpy_100000, make_data(100'000));
BENCHMARK_CAPTURE(bench_std_memcpy, std_memcpy_1000000, make_data(1'000'000));
BENCHMARK_CAPTURE(bench_std_memcpy, std_memcpy_10000000, make_data(10'000'000));
*/

BENCHMARK_MAIN();
