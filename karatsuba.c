#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* thread granularity */
#define GRANULARITY 8192

/* threshold to standard multiplication */
#define STANDARD_THRESHOLD 8
#define ST "8" /* "STANDARD_THRESHOLD" */
#define ST8 "64" /* "STANDARD_THRESHOLD * 8" */
#define UNROLL(c) c(0) c(8) c(16) c(24) c(32) c(40) c(48) c(56) /* c(i*8) where i in STANDARD_THRESHOLD */

/*
 * r = x*y
 * l must satisfy #r = 2l and #x = #y = l
 * elementary school algorithm
 */
static void standard_mult(uint64_t *restrict r,uint64_t *restrict x, uint64_t *restrict y,  size_t l) {
	for (size_t j = 0; j < l; j++) r[j] = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t c = 0;
		for (size_t j = 0; j < l; j++) {
			uint64_t vl, vh;
#ifdef __INTEL_COMPILER
			__asm__("mulq %3": "=a"(vl), "=d"(vh) : "a"(x[i]), "g"(y[j]));
#else
			__uint128_t v = (__uint128_t)x[i] * (__uint128_t)y[j];
			vl = (uint64_t)v, vh = (uint64_t)(v>>64);
#endif
			vl += c;
			vh += vl < c;
			r[i+j] += vl;
			vh += r[i+j] < vl;
			c = vh;
		}
		r[i+l] = c;
	}
}

/*
 * r = x*y
 * #x = #y = STANDARD_THRESHOLD is assumed
 * elementary school algorithm
 */
static void standard_mult_three_fixed(uint64_t *restrict r1, uint64_t *restrict x1, uint64_t *restrict y1,
		uint64_t *restrict r2, uint64_t *restrict x2, uint64_t *restrict y2,
		uint64_t *restrict r3, uint64_t *restrict x3, uint64_t *restrict y3) {
	for (size_t j = 0; j < STANDARD_THRESHOLD; j++) r1[j] = 0;
	for (size_t j = 0; j < STANDARD_THRESHOLD; j++) r2[j] = 0;
	for (size_t j = 0; j < STANDARD_THRESHOLD; j++) r3[j] = 0;
#ifdef ASM
	int cnt = STANDARD_THRESHOLD;
#define SMA_KERNEL(j) \
	"movq "#j"(%2), %%rax\n\t" /* rax = y1[j] */ \
	"mulq (%1)\n\t" /* rdx:rax = rax * r8 */ \
	"addq %%r11, %%rax\n\t" /* rax += r11 */ \
	"setc %%r11b\n\t" \
	"movzx %%r11b, %%r11\n\t" /* r11 = carry */ \
	"addq %%rax, "#j"(%0)\n\t" /* r1[j] += rax */ \
	"adcq %%rdx, %%r11\n\t" /* r11 += rdx + carry */ \
	\
	"movq "#j"(%5), %%rax\n\t" /* rax = y2[j] */ \
	"mulq (%4)\n\t" /* rdx:rax = rax * r9 */ \
	"addq %%r12, %%rax\n\t" /* rax += r12 */ \
	"setc %%r12b\n\t" \
	"movzx %%r12b, %%r12\n\t" /* r12 = carry */ \
	"addq %%rax, "#j"(%3)\n\t" /* r2[j] += rax */ \
	"adcq %%rdx, %%r12\n\t" /* r12 += rdx + carry */ \
	\
	"movq "#j"(%8), %%rax\n\t" /* rax = y3[j] */ \
	"mulq (%7)\n\t" /* rdx:rax = rax * r10 */ \
	"addq %%r13, %%rax\n\t" /* rax += r12 */ \
	"setc %%r13b\n\t" \
	"movzx %%r13b, %%r13\n\t" /* r13 = carry */ \
	"addq %%rax, "#j"(%6)\n\t" /* r3[j] += rax */ \
	"adcq %%rdx, %%r13\n\t" /* r13 += rdx + carry */

	__asm__ __volatile__(
		"1:\n\t"
		"xorq %%r11, %%r11\n\t" /* r11 = 0 */
		"xorq %%r12, %%r12\n\t" /* r12 = 0 */
		"xorq %%r13, %%r13\n\t" /* r13 = 0 */
		UNROLL(SMA_KERNEL)
		"movq %%r11, "ST8"(%0)\n\t" /* r1[STANDARD_THRESHOLD] = r11 */
		"movq %%r12, "ST8"(%3)\n\t" /* r2[STANDARD_THRESHOLD] = r12 */
		"movq %%r13, "ST8"(%6)\n\t" /* r3[STANDARD_THRESHOLD] = r13 */
		"leaq 8(%0), %0\n\t" /* r1++ */
		"leaq 8(%1), %1\n\t" /* x1++ */
		"leaq 8(%3), %3\n\t" /* r2++ */
		"leaq 8(%4), %4\n\t" /* x2++ */
		"leaq 8(%6), %6\n\t" /* r3++ */
		"leaq 8(%7), %7\n\t" /* x3++ */
		"sub $1, %9\n\t" /* cnt-- */
		"jnz 1b\n\t" /* jmp if cnt != 0 */
	: "+r"(r1), "+r"(x1), "+r"(y1), "+r"(r2), "+r"(x2), "+r"(y2), "+r"(r3), "+r"(x3), "+r"(y3), "+g"(cnt)
	:
	: "memory", "rax", "rdx", "r11", "r12", "r13");

#else
	for (size_t i = 0; i < STANDARD_THRESHOLD; i++) {
		uint64_t c1 = 0, c2 = 0, c3 = 0;
		for (size_t j = 0; j < STANDARD_THRESHOLD; j++) {
			uint64_t vl1, vh1, vl2, vh2, vl3, vh3;
#ifdef __INTEL_COMPILER
			__asm__("mulq %3": "=a"(vl1), "=d"(vh1) : "a"(x1[i]), "g"(y1[j]));
			__asm__("mulq %3": "=a"(vl2), "=d"(vh2) : "a"(x2[i]), "g"(y2[j]));
			__asm__("mulq %3": "=a"(vl3), "=d"(vh3) : "a"(x3[i]), "g"(y3[j]));
#else
			__uint128_t v1 = (__uint128_t)x1[i] * (__uint128_t)y1[j];
			vl1 = (uint64_t)v1, vh1 = (uint64_t)(v1>>64);
			__uint128_t v2 = (__uint128_t)x2[i] * (__uint128_t)y2[j];
			vl2 = (uint64_t)v2, vh2 = (uint64_t)(v2>>64);
			__uint128_t v3 = (__uint128_t)x3[i] * (__uint128_t)y3[j];
			vl3 = (uint64_t)v3, vh3 = (uint64_t)(v3>>64);
#endif
			vl1 += c1;
			vh1 += vl1 < c1;
			r1[i+j] += vl1;
			vh1 += r1[i+j] < vl1;
			c1 = vh1;
			vl2 += c2;
			vh2 += vl2 < c2;
			r2[i+j] += vl2;
			vh2 += r2[i+j] < vl2;
			c2 = vh2;
			vl3 += c3;
			vh3 += vl3 < c3;
			r3[i+j] += vl3;
			vh3 += r3[i+j] < vl3;
			c3 = vh3;
		}
		r1[i+STANDARD_THRESHOLD] = c1;
		r2[i+STANDARD_THRESHOLD] = c2;
		r3[i+STANDARD_THRESHOLD] = c3;
	}
#endif
}

/*
 * x += y + z
 * l must satisfy #y = #z = l
 * the carry is propagated to higher bits of x
 */
static void add_threeop(uint64_t *x, uint64_t *y, uint64_t *z, size_t l) {
	char cy, cz;
	uint64_t o;
#ifdef ASM
	uint64_t *xorig = x;
#define ATH_KERNEL1(j) \
	"movq "#j"(%3), %%rax\n\t" /* rax = y[j] */ \
	"adcq %%rax, "#j"(%2)\n\t" /* x[j] += rax + carry */
#define ATH_KERNEL2(j) \
	"movq "#j"(%4), %%rbx\n\t" /* rbx = z[j] */ \
	"adcq %%rbx, "#j"(%2)\n\t" /* x[j] += rbx + carry */
	__asm__ __volatile__(
		"xorb %0, %0\n\t" /* cx = 0 */
		"xorb %1, %1\n\t" /* cy = 0 */
		"1:\n\t"
		"addb $-1, %0\n\t" /* (cx += 0xFF), carry = cx */
		UNROLL(ATH_KERNEL1)
		"setc %0\n\t" /* cx = carry */
		"addb $-1, %1\n\t" /* (cy += 0xFF), carry = cy */
		UNROLL(ATH_KERNEL2)
		"setc %1\n\t" /* cy = carry */
		"leaq "ST8"(%2), %2\n\t" /* x += STANDARD_THRESHOLD */
		"leaq "ST8"(%3), %3\n\t" /* y += STANDARD_THRESHOLD */
		"leaq "ST8"(%4), %4\n\t" /* z += STANDARD_THRESHOLD */
		"cmpq %5, %2\n\t"
		"jl 1b\n\t" /* jmp if x < xorig+l */
	: "=&a"(cy), "=&b"(cz), "+&r"(x), "+&r"(y), "+&r"(z) : "r"(x+l) : "memory");
	x = xorig;
#else
	cy = 0, cz = 0;
	for (size_t i = 0; i < l; i++) {
		o = x[i];
		x[i] += y[i] + cy;
		if (y[i] + cy) cy = x[i] < o;
		o = x[i];
		x[i] += z[i] + cz;
		if (z[i] + cz) cz = x[i] < o;
	}
#endif
	o = x[l];
	if ((x[l] += cy + cz) < o) {
		for (size_t i = l+1; ++x[i] == 0; i++);
	}
}

/*
 * x += y
 * l must satisfy #y = l
 * the carry is propagated to higher bits of x
 */
static void add_twoop(uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	char c;
#ifdef ASM
	uint64_t *xorig = x;
#define ATO_KERNEL(j) \
	"movq "#j"(%2), %%rax\n\t" /* rax = x[j] */ \
	"adcq %%rax, "#j"(%1)\n\t" /* y[j] += rax + carry */
	__asm__ __volatile__(
		"xorb %0, %0\n\t" /* c = 0, carry = 0 */
		"1:\n\t"
		"addb $-1, %0\n\t" /* (c += 0xFF), carry = c */
		UNROLL(ATO_KERNEL)
		"leaq "ST8"(%1), %1\n\t" /* x += STANDARD_THRESHOLD */
		"leaq "ST8"(%2), %2\n\t" /* y += STANDARD_THRESHOLD */
		"setc %0\n\t" /* c = carry */
		"cmpq %3, %1\n\t"
		"jl 1b\n\t" /* jmp if x < xorig+l */
	: "=&a"(c), "+&r"(x), "+&r"(y) : "r"(x+l) : "memory");
	x = xorig;
#else
	c = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t o = x[i];
		x[i] += y[i] + c;
		if (y[i] + c) c = x[i] < o;
	}
#endif
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
	char b;
#ifdef ASM
	uint64_t *xorig = x;
#define STO_KERNEL(j) \
	"movq "#j"(%2), %%rax\n\t" /* rax = x[j] */ \
	"sbbq %%rax, "#j"(%1)\n\t" /* y[j] -= rax + borrow */
	__asm__ __volatile__(
		"xorb %0, %0\n\t" /* b = 0, borrow = 0 */
		"1:\n\t"
		"addb $-1, %0\n\t" /* (b += 0xFF), borrow = b */
		UNROLL(STO_KERNEL)
		"leaq "ST8"(%1), %1\n\t" /* x += STANDARD_THRESHOLD */
		"leaq "ST8"(%2), %2\n\t" /* y += STANDARD_THRESHOLD */
		"setc %0\n\t" /* b = borrow */
		"cmpq %3, %1\n\t"
		"jl 1b\n\t" /* jmp if x < xorig+l */
	: "=&a"(b), "+&r"(x), "+&r"(y) : "r"(x+l) : "memory");
	x = xorig;
#else
	b = 0;
	for (size_t i = 0; i < l; i++) {
		uint64_t o = x[i];
		x[i] -= y[i] + b;
		if (y[i] + b) b = x[i] > o;
	}
#endif
	if (b) {
		for (size_t i = l; x[i]-- == 0; i++);
	}
}

/*
 * r = |x-y|
 * returns x>y
 * l must satisfy #x = #y = l
 */
static char abs_diff_sign(uint64_t *restrict r, uint64_t *restrict x, uint64_t *restrict y, size_t l) {
	char s = 0;
	for (size_t i = l; i--; ) {
		if (x[i] != y[i]) {
			s = x[i] < y[i];
			break;
		}
	}
	if (s) {
		uint64_t *restrict t = y;
		y = x;
		x = t;
	}
#ifdef ASM
#define ADS_KERNEL(j) \
	"movq "#j"(%1), %%rax\n\t" \
	"sbbq "#j"(%2), %%rax\n\t" \
	"movq %%rax, "#j"(%0)\n\t"
	__asm__ __volatile__(
		"xorb %%al, %%al\n\t"
		"1:\n\t"
		"addb $-1, %%al\n\t"
		UNROLL(ADS_KERNEL)
		"leaq "ST8"(%0), %0\n\t"
		"leaq "ST8"(%1), %1\n\t"
		"leaq "ST8"(%2), %2\n\t"
		"setc %%al\n\t"
		"cmpq %3, %0\n\t"
		"jl 1b\n\t"
	: "+&r"(r), "+&r"(x), "+&r"(y) : "r"(r+l) : "memory", "rax");
#else
	char b = 0;
	for (size_t i = 0; i < l; i++) {
		r[i] = x[i] - (y[i] + b);
		if (y[i] + b) b = r[i] > x[i];
	}
#endif
	return s;
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
	char s1 = abs_diff_sign(t, x, x+l/2, l/2);
	char s2 = abs_diff_sign(t+l/2, y, y+l/2, l/2);
	if (l == 2*STANDARD_THRESHOLD) {
		standard_mult_three_fixed(r, x, y, r+l, x+l/2, y+l/2, t+l, t, t+l/2);
	} else {
		karatsuba_mult_sing_do(r, x, y, l/2, t+2*l);
		karatsuba_mult_sing_do(r+l, x+l/2, y+l/2, l/2, t+2*l);
		karatsuba_mult_sing_do(t+l, t, t+l/2, l/2, t+2*l);
	}
	memcpy(t, r, l*sizeof(uint64_t));
	add_threeop(r+l/2, t, r+l, l);
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
	char s;
	uint64_t *r;
	uint64_t *t;
	size_t l;
	struct kmul_cont *next;
} kmul_cont_t;

/*
 * save (arguments of) a continuation
 */
static void karatsuba_mult_schd_add_cont(kmul_cont_t **argsp, char s, uint64_t *r, uint64_t *t, size_t l) {
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
static void karatsuba_mult_schd_cont_proc(char s, uint64_t *restrict r, uint64_t *restrict t, size_t l) {
	memcpy(t, r, l*sizeof(uint64_t));
	add_threeop(r+l/2, t, r+l, l);
	(s ? add_twoop : sub_twoop)(r+l/2, t+l, l);
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
	if (l <= GRANULARITY) {
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
		uint64_t *t = malloc(2*l*sizeof(uint64_t));
		char s1 = abs_diff_sign(t, x, x+l/2, l/2);
		char s2 = abs_diff_sign(t+l/2, y, y+l/2, l/2);
		karatsuba_mult_schd(r, x, y, l/2, 3*pos, 3*free_deg, 3*deg, tnum, argsp);
		karatsuba_mult_schd(r+l, x+l/2, y+l/2, l/2, 3*pos+1, 3*free_deg, 3*deg, tnum, argsp);
		karatsuba_mult_schd(t+l, t, t+l/2, l/2, 3*pos+2, 3*free_deg, 3*deg, tnum, argsp);
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
	if (l <= STANDARD_THRESHOLD) {
		standard_mult(r, x, y, l);
		return;
	}
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
