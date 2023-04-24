# With -O0:
# FFT
FFT of 100000000 elements took 0 milliseconds (7746 nanoseconds)

Overall MKL handling took 5 milliseconds (5163261 nanoseconds)

# quadratic mean
quadratic mean of 100000000 elements took 0 milliseconds (23887 nanoseconds)

# std::memcpy
memcpy of 100000000 elements took 15 milliseconds (15172138 nanoseconds)

# std::accumulate
accumulate of 100000000 elements took 0 milliseconds (490 nanoseconds)

# boost::accumulate
accumulate of 100000000 elements took 0 milliseconds (1213 nanoseconds)

# boost::accumulators::mean
accumulators::mean of 100000000 elements took 10531 milliseconds (10531275846 nanoseconds)

# With -O2:
# FFT
FFT of 100000000 elements took 0 milliseconds (12429 nanoseconds)

Overall MKL handling took 55 milliseconds (55395275 nanoseconds)


# quadratic mean
quadratic mean of 100000000 elements took 0 milliseconds (11583 nanoseconds)


# std::memcpy
memcpy of 100000000 elements took 13 milliseconds (13479683 nanoseconds)


# std::accumulate
accumulate of 100000000 elements took 0 milliseconds (114 nanoseconds)


# boost::accumulate
accumulate of 100000000 elements took 0 milliseconds (114 nanoseconds)


# boost::accumulators::mean
accumulators::mean of 100000000 elements took 2207 milliseconds (2207872484 nanoseconds)


