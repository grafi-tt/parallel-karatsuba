#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#include "karatsuba.c"
#include "test_macro.h"

static int err;

void assert_eq(uint64_t *r, uint64_t *a, size_t l) {
	if (memcmp(a, r, l*sizeof(uint64_t))) {
		err++;
		for (size_t i = 0; i < l; i++) printf("%016"PRIX64, r[l-i-1]);
		printf(" != ");
		for (size_t i = 0; i < l; i++) printf("%016"PRIX64, a[l-i-1]);
		printf("\n");
	}
}

void test_standard_mult() {
	uint64_t x[2] = NUM(0x123456789ABCDEF0, 0x42424242FEDCBA98);
	uint64_t y[2] = NUM(0xFEDCBA9876543210, 0x0123456748484848);
	uint64_t a[4] = NUM(0x121FA00AD77D7422, 0x65791E6719403D1F, 0x3B7679C114D15188, 0xE3AD483CFD4F3AC0);
	uint64_t r[4] = NUM(0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE); /* filler */
	standard_mult(r, x, y, 2);
	assert_eq(r, a, 4);
}

void test_add_twoop() {
	uint64_t x[4] = NUM(0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0, 0x42424242FEDCBA98);
	uint64_t y[2] = NUM(0xFEDCBA9876543210, 0x0123456748484848);
	uint64_t a[4] = NUM(0x0000000000000001, 0x0000000000000000, 0x1111111111111100, 0x436587AA472502E0);
	add_twoop(x, y, 2);
	assert_eq(x, a, 4);
}

void test_sub_twoop() {
	uint64_t x[4] = NUM(0x0000000000000001, 0x0000000000000000, 0x1111111111111100, 0x436587AA472502E0);
	uint64_t y[2] = NUM(0xFEDCBA9876543210, 0x0123456748484848);
	uint64_t a[4] = NUM(0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0, 0x42424242FEDCBA98);
	sub_twoop(x, y, 2);
	assert_eq(x, a, 4);
}

void test_abs_diff_sign() {
	uint64_t x[4] = NUM(0xFEDCBA9876543210, 0x123456789ABCDEF0, 0x0123456748484848, 0x42424242FEDCBA98);
	uint64_t y[4] = NUM(0x123456789ABCDEF0, 0xFFFFFFFFFFFFFFFF, 0x42424242FEDCBA98, 0x42424242FEDCBA98);
	uint64_t a[4] = NUM(0xECA8641FDB97531F, 0x123456789ABCDEF0, 0xBEE10324496B8DB0, 0x0000000000000000);
	uint64_t r1[4] = NUM(0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE); /* filler */
	uint64_t r2[4] = NUM(0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE); /* filler */
	unsigned int s = abs_diff_sign(r1, x, y, 4);
	if (s != 0) {
		puts("s should be 0");
		err++;
	}
	assert_eq(r1, a, 4);
	s = abs_diff_sign(r2, y, x, 4);
	if (s != 1) {
		puts("s should be 1");
		err++;
	}
	assert_eq(r2, a, 4);
}

void test_karatsuba_mult(int tnum) {
	uint64_t x[32] = NUM(
			0x239B290CA32C613E, 0x91D8C1E59E110C59, 0xD552ED9073F1B48A, 0x26431FBA103288D4,
			0x7B7F823A11D2AB51, 0xA1B925FCC2134288, 0xB2C9CD71D4450AFE, 0x89D0E9BF509ED293,
			0x6079F3547DD6609E, 0x3C6E5073A783E309, 0xDC28FB9D430459EA, 0x992410EEB777B192,
			0x72BFE8A877AA49AE, 0x37BAD75C64D0BC38, 0x164811574C405945, 0x3268C93385C511F7,
			0xAD3625C6EA248405, 0x1D8A9D49C030D5F6, 0x5FB9D0F7D4AD0544, 0x03D360FFF2F51758,
			0xB91981C1420C9D52, 0x35500B9EA541B8E7, 0x10E9091651F1297B, 0xF016A2DCC9805843,
			0xFDA88C171286E99D, 0x1FBEA498422B2DB3, 0x137927EF4E58F5A4, 0x042FEDF54D9FDB2B,
			0xC33F53AC7BEFA9CD, 0x64A3CA1DC723BBB3, 0x4F87696A34D42C00, 0xDC0DB3D55CAE9D70);
	uint64_t y[32] = NUM(
			0x78C06972C1022531, 0x380BD0971B65E3DF, 0x16D57C403D15F99F, 0x0394A88DEE3F3404,
			0xC6DB255DAF6F0137, 0x7845159199455187, 0x51C6CEA37486B7A7, 0x1D856C31CF0DA100,
			0x8CBAA214CB4C663D, 0x9CAB40B52E4A16F9, 0x5447F01B28B263EA, 0x54F08C1A51CFB697,
			0xD032DFABF9816BD4, 0x81A8B2A02F9EA826, 0x9498BEC71CAAB62C, 0x226E1B6122E26329,
			0xEB8FB9B2642A6356, 0x890750D2CADF7D96, 0x5AB34136599E6782, 0x178486187A0F8B37,
			0xB3A82547751DE43C, 0x205332E9A5C0E9B0, 0x5EBABD32961B8AED, 0xA9C35544EAFC364A,
			0x448DD962B3FC1691, 0xE413723F6E12607A, 0x7BB42888BDB8305D, 0xB30CD3D2A27F43E7,
			0x553B9553360D08CD, 0x50D35733F34AB65E, 0x146E2F56770CE5B6, 0x6B95812FCEAB8B26);
	uint64_t a[64] = NUM(
			0x10CB7E474FBDEBC1, 0x3F7BF57619460309, 0xEF3686CCED21DE5E, 0xBB1BC92ECC1AB9C4,
			0x921099E9448A4624, 0x1C27663161AD4F42, 0x3C1D4B8CA0715C48, 0x66A1B3C7C0C7FA94,
			0x2CA8821B0E71434F, 0xA6ACC662699287A9, 0x66B956E51948333A, 0x32208CBA5771AB45,
			0x081C112E6F1B8FEE, 0x491D0EF476145473, 0x90E887C2396EC9EC, 0xC87A432BC44B8E8C,
			0x0D0E2F5B56ADAE34, 0x7F61B18A6058F8AD, 0x820A74FB1EBC9B02, 0x984C240B768352A8,
			0x39BD9F8BD74F243A, 0x1917324AA1AF2020, 0x0DD60718115C9D35, 0x190F6A1969371A57,
			0xE827664F0EDEC249, 0xB7FC10E2C3DAA15F, 0x689CE9AE1F98D341, 0x1A253DCC56C95C02,
			0x186EC92BC36FC742, 0x510192D93F7199C0, 0x9C6B8E8035C067A6, 0x70197A08284B159A,
			0x5357293D957C7E46, 0x6D20D8FC41DDC6D2, 0x437EB96EE51C6CB7, 0x4805025C90CEF7E5,
			0x56BD3FADE240B9A7, 0x468EC4FC4A530FE9, 0xF78C467FCFD8FE74, 0x9BF118A3D052ED24,
			0xD9FC8FBEC8592717, 0x0B587414D4D4BE2E, 0x803170C928B24EE8, 0x5D762F1303E61CC5,
			0xD7EB65FAAA94F769, 0x209C9B7C36655D57, 0x7C3E675C0CC3D339, 0x0C6B6B8B99546CD3,
			0xA26A0CAB240FEC7B, 0xDE68EABFDEC022CA, 0x221E4B5D11353CB7, 0x5A2D3062EC14EF81,
			0xBB23E65445FFA372, 0xB6F536C2C2670C53, 0x190168B54F300D79, 0x8E813D11EBF7E457,
			0xA2D8A1C07E61F43B, 0xD96F3210A4D3B0D3, 0xEC1985DF481A0296, 0x3C018074AA17AE33,
			0x9C217696A524A86D, 0xE117F294665BD4EA, 0xB1C6535AD0509668, 0x5E5B4DE1DB372EA0);
	uint64_t r[64];
	for (size_t i = 0; i < 64; i++) r[i] = 0xEEEEEEEEEEEEEEEE; // filler
	if (tnum == 0) {
		karatsuba_mult_sing(r, x, y, 32);
	} else {
		int orig_tnum = omp_get_max_threads();
		omp_set_num_threads(tnum);
		karatsuba_mult(r, x, y, 32);
		omp_set_num_threads(orig_tnum);
	}
	assert_eq(r, a, 64);
}

int main() {
	test_standard_mult();
#ifndef ASM
	test_add_twoop();
	test_sub_twoop();
	test_abs_diff_sign();
#endif
	test_karatsuba_mult(0);
	test_karatsuba_mult(1);
	test_karatsuba_mult(2);
	test_karatsuba_mult(3);
	test_karatsuba_mult(4);
	if (!err) puts("ok");
	return err;
}
