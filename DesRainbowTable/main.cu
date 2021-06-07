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


	rainbow_table_t* t = RainbowTable_allocate(8, 16, 2048);


	/*for (int i = 0; i < 20; i++) {

		t->nodes[i].key[0] = (char)(65 + i);
		t->nodes[i].encoded_password[0] = (char)(97 + i);
		t->nodes[i].key[7] = '\0';
		t->nodes[i].encoded_password[7] = '\0';

		for (int j = 0; j < 6; j++) {
			t->nodes[i].key[1 + j] = (char)(48 + j);
			t->nodes[i].encoded_password[1 + j] = (char)(48 + j);
		}
	}*/

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
	GenerateRainbowTable <<<1889, 1>>> (d_origins, d_keys_pointers, d_encoded_passwords_pointers, t->key_size, t->encoded_password_size, NULL, 8, 0, test);
	RainbowTable_cuda_copy_results_to_host(t, d_origins);
	RainbowTable_cuda_free(t, d_keys_pointers, d_encoded_passwords_pointers, d_origins);

	RainbowTable_write_to_file(t, "test.bin");

	cudaFree(test);
	RainbowTable_free(t);

	return 0;

	// testowanie dla jednego watku
	//srand(time(NULL));
	//unsigned char key_host[64] = {1,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0} ;

	//unsigned char plain_host[] = { 'n','@',',','d','1','J','o','?' };
	//
	//unsigned char* encrypted_host = (unsigned char*)malloc(sizeof(unsigned char)*8);

	//unsigned char* test;
	//cudaMalloc(&test, 8);
	//unsigned char* key;
	//cudaMalloc(&key, sizeof(unsigned char) * 64);
	//unsigned char* plain;
	//cudaMalloc(&plain, sizeof(unsigned char) * 8);
	//unsigned char* encrypted;
	//cudaMalloc(&encrypted, sizeof(unsigned char) * 8);

	//cudaMemcpy(key, key_host, sizeof(unsigned char) * 64, cudaMemcpyHostToDevice);
	//cudaMemcpy(plain, plain_host, sizeof(unsigned char) * 8, cudaMemcpyHostToDevice);

	////printf("%s\n", plain_host);
	////DESCipher<<<1,1>>>(key,plain,encrypted,test);

	//cudaMemcpy(encrypted_host, encrypted, sizeof(unsigned char) * 8, cudaMemcpyDeviceToHost);
	//cudaFree(key);
	//cudaFree(plain);
	//cudaFree(encrypted);
	//cudaFree(test);
	////printf("%s\n", encrypted_host);

	//for (int i = 0; i < 8; i++) {
	//	printf("%d ",encrypted_host[i]);
	//}
	//printf("\n");
	//free(encrypted_host);


	//RainbowTable_free(t);

	//return 0;
}