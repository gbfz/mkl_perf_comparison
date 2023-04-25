# With -O2:
-------------------------------------------------------------------------------------
Benchmark                                           Time             CPU   Iterations
-------------------------------------------------------------------------------------
benchmark_mkl_fft_real/100000                  102917 ns       102755 ns         6006
benchmark_mkl_fft_real/1000000                2579475 ns      2571894 ns          255
benchmark_mkl_fft_real/10000000             169020697 ns    168699146 ns            4
benchmark_mkl_fft_complex/100000               551924 ns       548852 ns         1254
benchmark_mkl_fft_complex/1000000             8719524 ns      8517655 ns           71
benchmark_mkl_fft_complex/10000000          323928056 ns    322958382 ns            2
benchmark_std_memcpy/100000                      9518 ns         9507 ns        73159
benchmark_std_memcpy/1000000                   143799 ns       143547 ns         4932
benchmark_std_memcpy/10000000                 3185131 ns      3172664 ns          222
benchmark_std_accumulate/100000               1473936 ns      1454356 ns          486
benchmark_std_accumulate/1000000             14444974 ns     14388304 ns           49
benchmark_std_accumulate/10000000           142447774 ns    142270890 ns            5
benchmark_boost_accumulate/100000             1411213 ns      1409248 ns          498
benchmark_boost_accumulate/1000000           14252100 ns     14232997 ns           49
benchmark_boost_accumulate/10000000         142482754 ns    142284057 ns            5
benchmark_boost_accumulators_mean/100000       402757 ns       402261 ns         1736
benchmark_boost_accumulators_mean/1000000     4187788 ns      4179808 ns          167
benchmark_boost_accumulators_mean/10000000   41945002 ns     41821085 ns           17
benchmark_quadratic_mean/100000                103541 ns       103212 ns         6966
benchmark_quadratic_mean/1000000              1337697 ns      1332065 ns          527
benchmark_quadratic_mean/10000000            13430479 ns     13404835 ns           52

# With -O0:
-------------------------------------------------------------------------------------
Benchmark                                           Time             CPU   Iterations
-------------------------------------------------------------------------------------
benchmark_mkl_fft_real/100000                  126107 ns       125802 ns         6221
benchmark_mkl_fft_real/1000000                2549559 ns      2539847 ns          259
benchmark_mkl_fft_real/10000000             168921508 ns    168599953 ns            4
benchmark_mkl_fft_complex/100000              6119466 ns      6109368 ns          113
benchmark_mkl_fft_complex/1000000            64692087 ns     64595691 ns            8
benchmark_mkl_fft_complex/10000000          910633246 ns    908886870 ns            1
benchmark_std_memcpy/100000                      9921 ns         9909 ns        70782
benchmark_std_memcpy/1000000                   166673 ns       166268 ns         4702
benchmark_std_memcpy/10000000                 3052294 ns      3043990 ns          223
benchmark_std_accumulate/100000               4183339 ns      4178605 ns          170
benchmark_std_accumulate/1000000             41617047 ns     41552299 ns           17
benchmark_std_accumulate/10000000           415147919 ns    414623553 ns            2
benchmark_boost_accumulate/100000             4130205 ns      4096074 ns          175
benchmark_boost_accumulate/1000000           41942301 ns     41746153 ns           17
benchmark_boost_accumulate/10000000         404020468 ns    403487151 ns            2
benchmark_boost_accumulators_mean/100000     65147497 ns     65083617 ns           11
benchmark_boost_accumulators_mean/1000000   651165395 ns    650523348 ns            1
benchmark_boost_accumulators_mean/10000000 6576087106 ns   6568266116 ns            1
benchmark_quadratic_mean/100000                507221 ns       506570 ns         1387
benchmark_quadratic_mean/1000000              5459135 ns      5448857 ns          128
benchmark_quadratic_mean/10000000            54681203 ns     54595713 ns           13
