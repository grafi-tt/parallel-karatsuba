#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <omp.h>
#include <time.h>

#include "karatsuba.h"

/*
 * George Marsaglia, 2003, "Xorshift RNGs", J. Statistical Software 8(14), doi:10.18637/jss.v008.i14
 */
uint64_t xorshift64(uint64_t x) {
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return x;
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		fputs("bench [#bits] [#processors]\n", stderr);
		return 1;
	}
	size_t l = atol(argv[1]) / sizeof(uint64_t) / 8;
	int tnum = atoi(argv[2]);
	if (tnum) omp_set_num_threads(tnum);

	/* initialize */
	uint64_t seed = UINT64_C(14159265358979323846);
	uint64_t *mem = malloc(4*sizeof(uint64_t)*l);
	for (size_t i = 0, r = seed; i < 2*l; i++, r = xorshift64(r)) mem[i] = r; /* fill x and y */
	uint64_t *x = mem;
	uint64_t *y = mem+l;
	uint64_t *r = mem+2*l;

	/* bench */
	struct timespec tp1, tp2;
	clock_gettime(CLOCK_MONOTONIC, &tp1);
	karatsuba_mult(r, x, y, l);
	clock_gettime(CLOCK_MONOTONIC, &tp2);

	/* check */
	mpz_t gmp_x, gmp_y, gmp_r;
	mpz_init(gmp_x);
	mpz_init(gmp_y);
	mpz_init(gmp_r);
	mpz_import(gmp_x, l, -1, sizeof(uint64_t), 0, 0, x);
	mpz_import(gmp_y, l, -1, sizeof(uint64_t), 0, 0, y);
	mpz_mul(gmp_r, gmp_x, gmp_y);
	mpz_export(mem, NULL, -1, sizeof(uint64_t), 0, 0, gmp_r);
	int err = memcmp(r, mem, l);

	unsigned int ms = (tp2.tv_sec - tp1.tv_sec) * 1000 +
		(tp2.tv_nsec - tp1.tv_nsec + 1000*1000*1000) / (1000*1000) - 1000;
	printf("%ums\n", ms);
	if (err) fputs("result error\n", stderr);
	return err;
}
