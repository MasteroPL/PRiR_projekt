#ifndef _DES_CUH__
#define _DES_CUH__

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

des_t* DES_init(int plain_text_length);
void DES_encrypt(des_t* self);
void DES_free(des_t* self);


#endif