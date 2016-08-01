ifeq ($(CC), icc)
CFLAGS = -fast -openmp -std=c99 -Wall
else
CFLAGS = -O2 -march=native -mtune=native -fopenmp -std=c99 -Wall -Wextra -pedantic
endif

all: test bench

test: test.c karatsuba.c test_macro.h
	$(CC) $(CFLAGS) -o $@ test.c

bench: bench.c karatsuba.c karatsuba.h
	$(CC) $(CFLAGS) -o $@ bench.c karatsuba.c -lrt -lgmp

.PHONY: clean
clean:
	rm -f test bench
