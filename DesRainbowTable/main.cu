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


	rainbow_table_t* t = RainbowTable_allocate(8, 16, 1000000);


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



	// testowanie dla jednego watku
	srand(time(NULL));
	unsigned char key_host[64] = {1,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0,1,0, 1,0,0,1,1,1,0,1,0} ;

	unsigned char plain_host[] = { 'n','@',',','d','1','J','o','?' };
	
	unsigned char* encrypted_host = (unsigned char*)malloc(sizeof(unsigned char)*8);

	unsigned char* test;
	cudaMalloc(&test, 8);
	unsigned char* key;
	cudaMalloc(&key, sizeof(unsigned char) * 64);
	unsigned char* plain;
	cudaMalloc(&plain, sizeof(unsigned char) * 8);
	unsigned char* encrypted;
	cudaMalloc(&encrypted, sizeof(unsigned char) * 8);

	cudaMemcpy(key, key_host, sizeof(unsigned char) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(plain, plain_host, sizeof(unsigned char) * 8, cudaMemcpyHostToDevice);

	//printf("%s\n", plain_host);
	DESCipher<<<1,1>>>(key,plain,encrypted,test);

	cudaMemcpy(encrypted_host, encrypted, sizeof(unsigned char) * 8, cudaMemcpyDeviceToHost);
	cudaFree(key);
	cudaFree(plain);
	cudaFree(encrypted);
	cudaFree(test);
	//printf("%s\n", encrypted_host);

	for (int i = 0; i < 8; i++) {
		printf("%d ",encrypted_host[i]);
	}
	printf("\n");
	free(encrypted_host);


	RainbowTable_free(t);

	return 0;
}