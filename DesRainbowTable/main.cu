#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rainbow_table.h"

#include <stdio.h>
#include <stdlib.h>
// do testów time
#include <time.h>
#include <cuda.h>

#include "indexer.cuh"
#include "des.cuh"

int main() {
	//unsigned char plain_text[8];
	//plain_text[0] = 1;
	//plain_text[1] = 35;
	//plain_text[2] = 69;
	//plain_text[3] = 103;
	//plain_text[4] = 137;
	//plain_text[5] = 171;
	//plain_text[6] = 205;
	//plain_text[7] = 239;

	//des_t* d = DES_init(8);
	//d->plain_text = plain_text;
	//d->key64[0] = 19;
	//d->key64[1] = 52;
	//d->key64[2] = 87;
	//d->key64[3] = 121;
	//d->key64[4] = 155;
	//d->key64[5] = 188;
	//d->key64[6] = 223;
	//d->key64[7] = 241;
	///*for (int i = 0; i < 8; i++) {
	//	d->key64[i] = 12 + (i*15);
	//}*/
	//DES_encrypt(d);
	//

	//return 0;

	rainbow_table_t* t = RainbowTable_allocate(8, 8, 10240);

	unsigned char key_bytes[8];
	unsigned char key_bits[64];

	int key_index = 1025;

	for (short i = 0; i < 8; i++) {
		key_bytes[i] = key_index % 256;
		key_index /= 256;
	}

	unsigned char* test;
	cudaMalloc(&test, 8);

	unsigned char** d_keys_pointers;
	unsigned char** d_encoded_passwords_pointers;
	unsigned char** d_origins;
	RainbowTable_cuda_allocate(t, &d_keys_pointers, &d_encoded_passwords_pointers, &d_origins);
	GenerateRainbowTable <<<1, 1024>>> (d_origins, d_keys_pointers, d_encoded_passwords_pointers, t->key_size, t->encoded_password_size, NULL, 8, 0, test);
	RainbowTable_cuda_copy_results_to_host(t, d_origins);
	RainbowTable_cuda_free(t, d_keys_pointers, d_encoded_passwords_pointers, d_origins);

	RainbowTable_write_to_file(t, "test.bin");

	cudaFree(test);
	RainbowTable_free(t);

	return 0;
	// dla jednego wątku działa
}