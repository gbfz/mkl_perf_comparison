cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

project(mkl_perf)

include_directories(
    ${PROJECT_SOURCE_DIR}/source
)

if (NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "Please set the MKLROOT variable to point to mkl install directory")
endif()

# set(ASAN -fsanitize=address,undefined)
set(ASAN "")

set(OPTLEVEL -O0)

add_compile_options(
    ${OPTLEVEL}
    ${ASAN}
    -fPIE
    -Wall
    -Wextra
    -Wpedantic
    -m64
    -I"$ENV{MKLROOT}/include/"
)

include_directories(
    $ENV{MKLROOT}/include
)

add_link_options(
    ${OPTLEVEL}
    ${ASAN}
    -lfmt
    -ltbb
    $ENV{MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a
    -L$ENV{MKLROOT}/lib/intel64
    -lmkl_intel_lp64
    -lmkl_core
    -Wl,--start-group
        $ENV{MKLROOT}/lib/intel64/libmkl_cdft_core.a
        $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a
        $ENV{MKLROOT}/lib/intel64/libmkl_sequential.a
        $ENV{MKLROOT}/lib/intel64/libmkl_core.a
        $ENV{MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a
    -Wl,--end-group
    -lpthread
    -lm
    -ldl
)

add_executable(scalar
    source/scalar.cpp
)
