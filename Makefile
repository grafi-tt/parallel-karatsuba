ifeq ($(CC), icc)
CFLAGS = -fast -openmp -std=c99 -Wall
else
CFLAGS = -O2 -march=native -mtune=native -fopenmp -std=c99 -Wall -Wextra -pedantic
endif

all: test_asm test_noasm bench_asm bench_noasm

test_asm: test.c karatsuba.c test_macro.h
	$(CC) $(CFLAGS) -DASM -o $@ test.c

test_noasm: test.c karatsuba.c test_macro.h
	$(CC) $(CFLAGS) -o $@ test.c

bench_asm: bench.c karatsuba.c karatsuba.h
	$(CC) $(CFLAGS) -DASM -o $@ bench.c karatsuba.c -lrt -lgmp

bench_noasm: bench.c karatsuba.c karatsuba.h
	$(CC) $(CFLAGS) -o $@ bench.c karatsuba.c -lrt -lgmp

.PHONY: clean
clean:
	rm -f test_asm test_noasm bench_asm bench_noasm
