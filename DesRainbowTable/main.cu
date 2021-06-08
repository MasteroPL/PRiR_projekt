#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rainbow_table.h"

#include <stdio.h>
#include <stdlib.h>
// do testów time
#include <time.h>
#include <cuda.h>

#include "indexer.cuh"
#include "kernel.h"


int main() {

	rainbow_table_t* t = RainbowTable_allocate(8, 16, 65535);

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
	GenerateRainbowTable <<<65535, 1>>> (d_origins, d_keys_pointers, d_encoded_passwords_pointers, t->key_size, t->encoded_password_size, NULL, 8, 0, test);
	RainbowTable_cuda_copy_results_to_host(t, d_origins);
	RainbowTable_cuda_free(t, d_keys_pointers, d_encoded_passwords_pointers, d_origins);

	RainbowTable_write_to_file(t, "test.bin");

	cudaFree(test);
	RainbowTable_free(t);

	return 0;
	// dla jednego wątku działa
}