#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define GRANULARITY 100
#define STANDARD_THRESHOLD 8

/*
 * r = x*y
 * l must satisfy #r = 2l and #x = #y = l
 * elementary school algorithm
 */
static void standard_mult(uint64_t *restrict r,uint64_t *restrict x, uint64_t *restrict y,  size_t l) {
	for (size_t j = 0; j < l; j++) r[j] = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t h = 0;
		for (size_t j = 0; j < l; j++) {
			uint64_t o = r[i+j], p = r[i+j] + h;
			__uint128_t v = (__uint128_t)x[i] * (__uint128_t)y[j];
			r[i+j] = p + (uint64_t)v;
			h = (uint64_t)(v>>64) + (p < o) + (r[i+j] < p);
		}
		r[i+l] = h;
	}
}

/*
 * x += y
 * l must satisfy #y = l
 * the carry is propagated to higher bits of x
 */
static void add_twoop(uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	unsigned int c = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t o = x[i];
		x[i] += y[i] + c;
		if (y[i] + c) c = x[i] < o;
	}
	if (c) {
		for (size_t i = l; ++x[i] == 0; i++);
	}
}

/*
 * x -= y
 * l must satisfy #y = l
 * the borrow is propagated to higher bits of x
 */
static void sub_twoop(uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	unsigned int b = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t o = x[i];
		x[i] -= y[i] + b;
		if (y[i] + b) b = x[i] > o;
	}
	if (b) {
		for (size_t i = l; x[i]-- == 0; i++);
	}
}

/*
 * r = |x-y|
 * returns x>y
 * l must satisfy #x = #y = l
 */
static unsigned int abs_diff_sign(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	unsigned int b = 0;
	for (size_t i = 0; i < l; i++) {
		r[i] = x[i] - (y[i] + b);
		if (y[i] + b) b = r[i] > x[i];
	}
	if (b) {
		size_t i = 0;
		for ( ; r[i] == 0; i++);
		r[i] = -r[i], i++;
		for ( ; i < l; i++) r[i] = ~r[i];
	}
	return b;
}

/*
 * single thread implementation
 * r = x*y
 * l must satisfy #r = 2l and #x = #y = l
 * l must be pow of 2
 * t is temoporaly space; 2l qwords is used per call
 * algorithm:
 *   r[0:l-1]  = x[0:l/2-1] * y[0:l/2-1]
 *   r[l:2l-1] = x[l/2:l-1] * y[l/2:l-1]
 *   r[l/2:3*l/2-1] += x[0:l/2-1] * y[0:l/2-1] (propagating carrry)
 *   r[l/2:3*l/2-1] += x[l/2:l-1] * y[l/2:l-1] (propagating carrry)
 *   r[l/2:3*l/2-1] -= (x[0:l/2-1] - x[l/2:l-1]) * (y[0:l/2-1] - y[l/2:l-1]) (propagating borrow)
 */
static void karatsuba_mult_sing_do(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l,
		uint64_t *restrict t) {
	if (l <= STANDARD_THRESHOLD) {
		standard_mult(r, x, y, l);
		return;
	}
	karatsuba_mult_sing_do(r, x, y, l/2, t+2*l);
	karatsuba_mult_sing_do(r+l, x+l/2, y+l/2, l/2, t+2*l);
	memcpy(t, r, 2*l*sizeof(uint64_t));
	add_twoop(r+l/2, t, l);
	add_twoop(r+l/2, t+l, l);
	unsigned int s1 = abs_diff_sign(t, x, x+l/2, l/2);
	unsigned int s2 = abs_diff_sign(t+l/2, y, y+l/2, l/2);
	karatsuba_mult_sing_do(t+l, t, t+l/2, l/2, t+2*l);
	(s1^s2 ? add_twoop : sub_twoop)(r+l/2, t+l, l);
}

/*
 * single thread implementation
 * r = x*y
 * l must satisfy #r = 2l and #x = #y = l
 * l must be pow of 2
 */
static void karatsuba_mult_sing(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	uint64_t *t = malloc(4*l*sizeof(uint64_t));
	karatsuba_mult_sing_do(r, x, y, l, t);
	free(t);
}

typedef struct kmul_cont {
	int s;
	uint64_t *r;
	uint64_t *t;
	size_t l;
	struct kmul_cont *next;
} kmul_cont_t;

/*
 * save (arguments of) a continuation
 */
static void karatsuba_mult_schd_add_cont(kmul_cont_t **argsp, int s, uint64_t *r, uint64_t *t, size_t l) {
	kmul_cont_t *args = malloc(sizeof(kmul_cont_t));
	args->s = s;
	args->r = r;
	args->t = t;
	args->l = l;
	args->next = *argsp;
	*argsp = args;
}

/*
 * the procedure after recursive calls
 */
static void karatsuba_mult_schd_cont_proc(int s, uint64_t *restrict r, uint64_t *restrict t, size_t l) {
	memcpy(t, r, 2*l*sizeof(uint64_t));
	add_twoop(r+l/2, t, l);
	add_twoop(r+l/2, t+l, l);
	(s ? add_twoop : sub_twoop)(r+l/2, t+2*l, l);
	free(t);
}

/*
 * call continuation with the saved arguments, in post-order of the call tree
 */
static void karatsuba_mult_schd_call_conts(kmul_cont_t *args) {
	if (!args) return;
	karatsuba_mult_schd_call_conts(args->next);
	karatsuba_mult_schd_cont_proc(args->s, args->r, args->t, args->l);
	free(args);
}

/*
 * divide the call tree and assign to threads
 */
static void karatsuba_mult_schd(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l,
		int pos, int free_deg, int deg, int tnum, kmul_cont_t **argsp) {
	int leaf = 0;
	if (deg >= tnum * GRANULARITY || l <= STANDARD_THRESHOLD) {
		leaf = 1;
	} else if (free_deg >= tnum) {
		int border = free_deg / tnum * tnum;
		if (pos < border) {
			leaf = 1;
		} else {
			if (pos == border) {
#pragma omp taskwait
				karatsuba_mult_schd_call_conts(*argsp);
				*argsp = NULL;
			}
			free_deg -= border;
			pos -= border;
		}
	}
	if (leaf) {
#pragma omp task
		karatsuba_mult_sing(r, x, y, l);
	} else {
		uint64_t *t = malloc(3*l*sizeof(uint64_t));
		unsigned int s1 = abs_diff_sign(t, x, x+l/2, l/2);
		unsigned int s2 = abs_diff_sign(t+l/2, y, y+l/2, l/2);
		karatsuba_mult_schd(r, x, y, l/2, 3*pos, 3*free_deg, 3*deg, tnum, argsp);
		karatsuba_mult_schd(r+l, x+l/2, y+l/2, l/2, 3*pos+1, 3*free_deg+1, 3*deg, tnum, argsp);
		karatsuba_mult_schd(t+2*l, t, t+l/2, l/2, 3*pos+2, 3*free_deg+2, 3*deg, tnum, argsp);
		/* save the continuation and go ahead to the next node */
		karatsuba_mult_schd_add_cont(argsp, s1^s2, r, t, l);
	}
}

/*
 * r = x*y
 * l must satisfiy #r = 2l and #x = #y = l
 * here the unit of l is bit
 * if l isn't pow of 2, do nothing
 */
void karatsuba_mult(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	if (!l || l&(l-1)) return;
#pragma omp parallel
#pragma omp single nowait
	{
		int tnum = omp_get_num_threads();
		kmul_cont_t *args = NULL;
		karatsuba_mult_schd(r, x, y, l, 0, 1, 1, tnum, &args);
#pragma omp taskwait
		karatsuba_mult_schd_call_conts(args);
	}
}
