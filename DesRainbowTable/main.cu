#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rainbow_table.h"
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
// do testów time
#include <time.h>
#include <cuda.h>

#include "indexer.cuh"
#include "des.cuh"

#define DEFAULT_PASSWORD "PRiR_fn!"

int main(int argc, char** argv) {
	char* tmp;
	unsigned char* plain_password;
	int rainbow_table_id = 0;
	char filename_buffor[50] = "rainbow_table_";
	char index_buffor[21];
	index_buffor[20] = '\0';

	// Wczytywanie indeksu tablicy
	if (argc > 1) {
		rainbow_table_id = atoi(argv[1]);
	}

	// Wczytywanie hasła
	if (argc > 2) {
		tmp = argv[2];
	}
	else {
		tmp = DEFAULT_PASSWORD;
	}
	tmp = DEFAULT_PASSWORD;
	int pass_len = 0;
	for (pass_len; tmp[pass_len] != '\0'; pass_len++) {}

	plain_password = (unsigned char*)malloc(sizeof(unsigned char) * pass_len);
	for (int i = 0; i < pass_len; i++) {
		plain_password[i] = (unsigned char)tmp[i];
	}
	int offset_len = pass_len + ((8 - pass_len % 8) % 8);
	

	rainbow_table_t* t = RainbowTable_allocate(8, offset_len, RAINBOW_TABLE_SIZE);

	unsigned char* test;
	cudaMalloc(&test, 8);

	unsigned char* d_plain_text;
	cudaMalloc(&d_plain_text, sizeof(unsigned char) * pass_len);
	cudaMemcpy(d_plain_text, plain_password, sizeof(unsigned char) * pass_len, cudaMemcpyHostToDevice);

	unsigned char** d_keys_pointers;
	unsigned char** d_encoded_passwords_pointers;
	unsigned char** d_origins;
	RainbowTable_cuda_allocate(t, &d_keys_pointers, &d_encoded_passwords_pointers, &d_origins);
	GenerateRainbowTable <<<2048,512>>> (d_origins, d_keys_pointers, d_encoded_passwords_pointers, t->key_size, t->encoded_password_size, d_plain_text, pass_len, 100, test);
	RainbowTable_cuda_copy_results_to_host(t, d_origins);
	RainbowTable_cuda_free(t, d_keys_pointers, d_encoded_passwords_pointers, d_origins);

	int tmp_index = rainbow_table_id;
	for (int i = 19; i >= 0; i--) {
		index_buffor[i] = 48 + (tmp_index % 10);
		tmp_index /= 10;
	}
	strcat(filename_buffor, index_buffor);
	strcat(filename_buffor, ".bin");
	RainbowTable_write_to_file(t, filename_buffor);

	cudaFree(test);
	RainbowTable_free(t);
	free(plain_password);

	return 0;
	// dla jednego wątku działa
}