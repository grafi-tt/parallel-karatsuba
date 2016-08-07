Platform
============
Compiling on GCC 5.4.0 and ICC 12.0.4 is tested.
GCC >= 4.4 and recent versions of clang would also work.

The bechmark code (bench.c) depends on unix-like environment.
In particular it requires
* clock_gettime function (on POSIX) to measure time, and
* GNU MP to check the result.

The assembly implementation depends on x86-64 architecture. SIMD extensions are unused.

Compile
============
Compile with GCC:
make

Compile with ICC:
CC=ICC make

Compile with clang:
Please edit Makefile.

Benchmark
============
Benchmark:
./bench_asm [#bits] [#processors]

Benchmark of pure C implementation:
./bench_noasm [#bits] [#processors]
