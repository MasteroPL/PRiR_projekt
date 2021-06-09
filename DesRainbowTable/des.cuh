#ifndef _DES_CUH__
#define _DES_CUH__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct DES {
	unsigned char key64[8];
	unsigned char _key56[7];
	unsigned char _key48[6];

	int shift_number;

	unsigned char* plain_text;
	int plain_text_length;

	unsigned char* _tmp_storage;
	unsigned char _r_expand[6];
	unsigned char _r_expand_6b[8];

	unsigned char* encrypted_text;
	int encrypted_text_length;
} des_t;

__device__ des_t* DES_init(int plain_text_length);
__device__ void DES_encrypt(des_t* self);
__global__ void GenerateRainbowTable(
	unsigned char** origins,
	unsigned char** keys_pointers,
	unsigned char** encoded_passwords_pointers,
	int key_size,
	int encoded_password_size,
	char* plain_password,
	int plain_password_length,
	int start_rainbow_table_index,
	unsigned char* test
);

#endif