#include <cmath>
#include <cstdio>
#include <mkl_dfti.h>
#include <fmt/format.h>
#include <chrono>
#include <string_view>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <concepts>
#include <functional>
#include <numeric>
#include <execution>
#include <random>
#include <limits>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <boost/range/numeric.hpp>

static constexpr const auto N = 100'000'000;

template <std::int32_t SIZE, typename T> requires std::integral<T> || std::floating_point<T>
std::vector<T> vec_of()
{
    std::default_random_engine eng;
    std::uniform_real_distribution<T> dst(
            std::numeric_limits<T>::lowest(),
            std::numeric_limits<T>::max()
    );

    std::vector<T> out;
    out.reserve(SIZE);
    for (int i : std::views::iota(0, SIZE))
    {
        out[i] = dst(eng);
    }
    return out;
};

template <typename F, typename... Args>
void measure(std::string_view msg, F&& f, Args&&... args)
{
    auto begin = std::chrono::high_resolution_clock::now();
    std::invoke(f, std::forward<Args...>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    auto nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    fmt::print("{} took {} milliseconds ({} nanoseconds)\n", msg, milli, nano);
}

void test_mkl_fft()
{
    fmt::print("<FFT>\n");
    auto input = vec_of<N, double>();
    decltype(input) output;
    output.reserve(N);

    measure("Overall MKL handling", [&]
    {
        DFTI_DESCRIPTOR_HANDLE handle = nullptr;
        DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, 32);
        DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(handle);

        measure(fmt::format("FFT of {} elements", N), [&]
        {
            DftiComputeForward(handle, input.data(), output.data());
        });

        DftiFreeDescriptor(&handle);
    });
    // please do not optimize all this away
    fmt::print(stderr, "{}", output[N / 2]);
    fmt::print("</FFT>\n\n");
}

void test_vec_memcpy()
{
    fmt::print("<std::memcpy>\n");
    auto input = vec_of<N, double>();
    decltype(input) output;
    output.resize(N);

    measure(fmt::format("memcpy of {} elements", N), [&]
    {
        std::memcpy(output.data(), input.data(), N);
    });
    // please do not optimize all this away
    fmt::print(stderr, "{}", output[N / 2]);
    fmt::print("</std::memcpy>\n\n");
}

void test_vec_std_accumulate()
{
    fmt::print("<std::accumulate>\n");
    auto input = vec_of<N, double>();

    measure(fmt::format("accumulate of {} elements", N), [&]
    {
        [[maybe_unused]]
        volatile const auto _ = std::accumulate(input.begin(), input.end(), 0.0);
    });
    fmt::print("</std::accumulate>\n\n");
}

void test_vec_boost_accumulate()
{
    fmt::print("<boost::accumulate>\n");
    auto input = vec_of<N, double>();

    measure(fmt::format("accumulate of {} elements", N), [&]
    {
        [[maybe_unused]]
        volatile const auto _ = boost::accumulate(input, 0.0);
    });
    fmt::print("</boost::accumulate>\n\n");
}

void test_vec_boost_mean()
{
    fmt::print("<boost::accumulators::mean>\n");
    auto input = vec_of<N, double>();

    using namespace boost::accumulators;
    measure(fmt::format("accumulators::mean of {} elements", N), [&]
    {
        accumulator_set<double, stats<tag::mean>> acc;

        for (const auto elem : vec_of<N, double>())
            acc(elem);

        [[maybe_unused]]
        volatile const auto _m = mean(acc);
    });
    fmt::print("</boost::accumulators::mean>\n\n");
}

void test_quadratic_mean()
{
    fmt::print("<quadratic mean>\n");
    auto input = vec_of<N, double>();

    measure(fmt::format("quadratic mean of {} elements", N), [&]
    {
        auto sum_of_squares = std::accumulate(input.begin(), input.end(), 0.0, std::plus<double>{});
        [[maybe_unused]]
        volatile const auto v = std::sqrt(sum_of_squares / N);

        fmt::print(stderr, "{}", v);
    });
    fmt::print("</quadratic mean>\n\n");
}

void test_geometric_mean()
{
    fmt::print("<geometric mean>\n");
    auto input = vec_of<N, double>();

    measure(fmt::format("geometric mean of {} elements", N), [&]
    {
        auto product_of_squares = std::accumulate(input.begin(), input.end(), 1.0, std::multiplies<double>{});
        [[maybe_unused]]
        volatile const auto v = std::pow(product_of_squares, 1 / N);
    });
    fmt::print("</geometric mean>\n");
}

int main()
{
    test_mkl_fft();
    test_quadratic_mean();
    test_vec_memcpy();
    test_vec_std_accumulate();
    test_vec_boost_accumulate();
    test_vec_boost_mean();
}
