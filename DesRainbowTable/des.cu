#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "des.cuh"
#include <stdlib.h>
#include <stdio.h>
#include "rainbow_table.h"

__device__ int DES_IP_MESSAGE_INDICES[] = {
	58, 50, 42, 34, 26, 18, 10, 2,
	60, 52, 44, 36, 28, 20, 12, 4,
	62, 54, 46, 38, 30, 22, 14, 6,
	64, 56, 48, 40, 32, 24, 16, 8,
	57, 49, 41, 33, 25, 17,  9, 1,
	59, 51, 43, 35, 27, 19, 11, 3,
	61, 53, 45, 37, 29, 21, 13, 5,
	63, 55, 47, 39, 31, 23, 15, 7
};
__device__ int PC1[] = {
	57, 49, 41, 33, 25, 17,  9,
	1, 58, 50, 42, 34, 26, 18,
	10,  2, 59, 51, 43, 35, 27,
	19, 11,  3, 60, 52, 44, 36,
	63, 55, 47, 39, 31, 23, 15,
	7, 62, 54, 46, 38, 30, 22,
	14,  6, 61, 53, 45, 37, 29,
	21, 13,  5, 28, 20, 12,  4
};
__device__ int PC2[] = {
	14, 17, 11, 24,  1,  5,
	 3, 28, 15,  6, 21, 10,
	23, 19, 12,  4, 26,  8,
	16,  7, 27, 20, 13,  2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};
__device__ unsigned char DES_BITS_CONJUCTIONS[] = {
	128, 64, 32, 16, 8, 4, 2, 1
};

__device__ int SHIFTS[] = {
	1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};
__device__ unsigned char SHIFTS_CONJUCTIONS[] = {
	// Jeden bit, dwa bity
	128, 192,
	// Jeden bit, dwa bity (specjalny przypadek środka)
	8, 12,
	// Jeden bit, dwa bity (zerowanie bitów środkowych po przesunięciu)
	239, 207
};

__device__ int R_SELECTION_TABLE[] = {
	32,  1,  2,  3,  4,  5,
	 4,  5,  6,  7,  8,  9,
	 8,  9, 10, 11, 12, 13,
	12, 13, 14, 15, 16, 17,
	16, 17, 18, 19, 20, 21,
	20, 21, 22, 23, 24, 25,
	24, 25, 26, 27, 28, 29,
	28, 29, 30, 31, 32,  1
};
__device__ int R_FINAL[] = {
	16,  7, 20, 21,
	29, 12, 28, 17,
	 1, 15, 23, 26,
	 5, 18, 31, 10,
	 2,  8, 24, 14,
	32, 27,  3,  9,
	19, 13, 30,  6,
	22, 11,  4, 25
};

__device__ int S_BOXES[8][4][16] = {
	{
		14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
		0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
		4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
		15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13
	},
	{
		15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
		3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
		0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
		13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9
	},
	{
		10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
		13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
		13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
		1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12
	},
	{
		7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
		13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
		10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
		3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14
	},
	{
		2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
		14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
		4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
		11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3
	},
	{
		12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
		10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
		9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
		4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13
	},
	{
		4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
		13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
		1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
		6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12
	},
	{
		13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
		1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
		7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
		2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11
	}
};

__device__ int FINAL_PERMUTATION[] = {
	40,  8, 48, 16, 56, 24, 64, 32,
	39,  7, 47, 15, 55, 23, 63, 31,
	38,  6, 46, 14, 54, 22, 62, 30,
	37,  5, 45, 13, 53, 21, 61, 29,
	36,  4, 44, 12, 52, 20, 60, 28,
	35,  3, 43, 11, 51, 19, 59, 27,
	34,  2, 42, 10, 50, 18, 58, 26,
	33,  1, 41,  9, 49, 17, 57, 25
};

/// <summary>
/// Inicjalizuje instancje strukture DES używaną przy kodowaniu tekstu jawnego, alokowana jest przestrzeń dla:
/// * kluczy (64, 56)
/// * tekstu kodowanego
/// 
/// Tekst jawny należy przekazać jako przypisanie do plain_text
/// </summary>
/// <param name="plain_text_length">Długość tekstu jawnego</param>
/// <returns>Wskaźnik na strukturę z zaalokowaną pamięcią dla tekstu szyfrowanego</returns>
__device__ des_t* DES_init(int plain_text_length) {
	des_t* result = (des_t*)malloc(sizeof(des_t));
	result->plain_text_length = plain_text_length;
	result->encrypted_text_length = plain_text_length;

	int modulo_8 = plain_text_length % 8;
	result->encrypted_text_length += ((8 - modulo_8) % 8);
	result->encrypted_text = (unsigned char*)malloc(sizeof(unsigned char) * result->encrypted_text_length);
	result->_tmp_storage = (unsigned char*)malloc(sizeof(unsigned char) * (result->encrypted_text_length));

	return result;
}

__device__ void DES_free(des_t* self) {
	free(self->_tmp_storage);
	free(self->encrypted_text);
	free(self);
}

// Krok 0
__device__ void DES_start(des_t* self) {
	for (int i = 0; i < self->plain_text_length; i++) {
		self->_tmp_storage[i] = self->plain_text[i];
	}
	for (int i = self->plain_text_length; i < self->encrypted_text_length; i++) {
		self->_tmp_storage[i] = 0;
	}

	// Czyszczenie śmieci
	for (int i = 0; i < self->encrypted_text_length; i++) {
		self->encrypted_text[i] = 0;
	}

	self->shift_number = 0;
}

 // Krok 1
__device__ void DES_initial_permutation(des_t* self) {
	int ip_index;
	int ip_byte;
	int ip_bit;
	unsigned char current_bit_value;

	// Dla kazdego bloku
	for (int i_block = 0; i_block < self->encrypted_text_length / 8; i_block++) {
		// Dla każdego bajtu w bloku
		for (int i_byte = 0; i_byte < 8; i_byte++) {
			// Dla każdego bitu w bajcie
			for (int i_bit = 0; i_bit < 8; i_bit++) {
				ip_index = DES_IP_MESSAGE_INDICES[i_byte * 8 + i_bit] - 1;
				ip_byte = ip_index / 8;
				ip_bit = ip_index % 8;

				current_bit_value = (self->_tmp_storage[i_block * 8 + ip_byte] >> (7 - ip_bit)) & 1;
				self->encrypted_text[i_block * 8 + i_byte] |= (
					 DES_BITS_CONJUCTIONS[i_bit] * current_bit_value
				);
			}
		}
	}
}

__device__ void DES_key_to_56(des_t* self) {
	unsigned char current_bit_value;
	// Czyszczenie śmieci
	for (int i = 0; i < 7; i++) {
		self->_key56[i] = 0;
	}

	int pc_index;
	int key56_byte;
	int key56_bit;
	int i_key56_index = 0;
	for (int i_byte = 0; i_byte < 8; i_byte++) {
		for (int i_bit = 0; i_bit < 7; i_bit++, i_key56_index++) {
			pc_index = PC1[i_key56_index] - 1;
			key56_byte = i_key56_index / 8;
			key56_bit = i_key56_index % 8;
			
			current_bit_value = (self->key64[pc_index / 8] >> (7 - (pc_index % 8))) & 1;
			self->_key56[key56_byte] |= (
				DES_BITS_CONJUCTIONS[key56_bit] * current_bit_value
			);
		}
	}
}

// Następne przesunięcie klucza
__device__ void DES_shift_key56(des_t* self) {
	unsigned char current_shift;
	unsigned char previous_shift;
	int shift = SHIFTS[self->shift_number];
	int conjuction = SHIFTS_CONJUCTIONS[shift - 1];

	// Zaczynam od skrajnej prawej
	previous_shift = (self->_key56[3] & SHIFTS_CONJUCTIONS[shift + 1]);
	previous_shift >>= (4 - shift);

	for (int i = 6; i >= 0; i--) {
		current_shift = (self->_key56[i] & conjuction);
		self->_key56[i] <<= shift;
		self->_key56[i] |= previous_shift;
		previous_shift = current_shift >> (8 - shift);
	}

	// Czystka dwóch/jednego bitu z lewego klucza
	self->_key56[3] &= SHIFTS_CONJUCTIONS[shift + 3];
	// I dopiero wtedy wpisanie odpowiednich wartości
	previous_shift <<= 4;
	self->_key56[3] |= previous_shift;

	self->shift_number++;
}

__device__ void DES_key56_to_48(des_t* self) {
	unsigned char current_bit_value;
	// Czyszczenie śmieci
	for (int i = 0; i < 6; i++) {
		self->_key48[i] = 0;
	}

	int pc2_index;
	for (int i_byte = 0; i_byte < 6; i_byte++) {
		for (int i_bit = 0; i_bit < 8; i_bit++) {
			pc2_index = PC2[i_byte * 8 + i_bit] - 1;
			current_bit_value = (self->_key56[pc2_index / 8] >> (7 - (pc2_index % 8))) & 1;
			self->_key48[i_byte] |= (
				DES_BITS_CONJUCTIONS[i_bit] * current_bit_value
			);
		}
	}
}

__device__ void DES_round(des_t* self) {
	unsigned char current_bit_value;
	int r_select_index;
	unsigned char current_r;
	int shifts[2] = { 4, 0 };
	char s_box_i1;
	char s_box_i2;

	// Kopiowaie do bufora i czyszczenie łańcucha wynikowego
	for (int i = 0; i < self->encrypted_text_length; i++) {
		self->_tmp_storage[i] = self->encrypted_text[i];
		self->encrypted_text[i] = 0;
	}
	
	for (int i_block = 0; i_block < self->encrypted_text_length / 8; i_block++) {
		// Wypełnianie lewej strony
		for (int i_byte = 0; i_byte < 4; i_byte++) {
			self->encrypted_text[i_block * 8 + i_byte] = self->_tmp_storage[i_block * 8 + 4 + i_byte];
		}
		// Wypełnianie prawej strony
		for (int i_byte = 0; i_byte < 6; i_byte++) {
			self->_r_expand[i_byte] = 0;
			for (int i_bit = 0; i_bit < 8; i_bit++) {
				r_select_index = R_SELECTION_TABLE[i_byte * 8 + i_bit] - 1;
				current_bit_value = (self->_tmp_storage[4 + r_select_index / 8] >> (7 - (r_select_index % 8))) & 1;
				self->_r_expand[i_byte] |= DES_BITS_CONJUCTIONS[i_bit] * current_bit_value;
			}
			self->_r_expand[i_byte] ^= self->_key48[i_byte];
		}
		// Generowanie 6 bitowych adresów dla S-boxów -> wpisanie wartości odpowiadającej adresowi zwrotnie
		for (int i_byte = 0, r_index=0; i_byte < 8; i_byte++) {
			self->_r_expand_6b[i_byte] = 0;
			for (int i_bit = 2; i_bit < 8; i_bit++, r_index++) {
				current_bit_value = (self->_r_expand[r_index / 8] >> (7 - (r_index % 8))) & 1;
				self->_r_expand_6b[i_byte] |= DES_BITS_CONJUCTIONS[i_bit] * current_bit_value;
			}
			s_box_i1 = 0;
			s_box_i1 |= (self->_r_expand_6b[i_byte]) & 1;
			s_box_i1 |= (self->_r_expand_6b[i_byte] >> 4) & 2;

			s_box_i2 = (self->_r_expand_6b[i_byte] >> 1) & 15;
			self->_r_expand_6b[i_byte] = S_BOXES[i_byte][s_box_i1][s_box_i2];
		}

		for (int i_byte = 0, r_index = 0; i_byte < 8; i_byte++) {
			current_r = 0;
			for (int i_bit = 4; i_bit < 8; i_bit++, r_index++) {
				r_select_index = R_FINAL[r_index] - 1;
				current_bit_value = (self->_r_expand_6b[r_select_index / 4] >> (3 - (r_select_index % 4))) & 1;

				current_r |= DES_BITS_CONJUCTIONS[i_bit] * current_bit_value;
			}
			current_r <<= shifts[i_byte % 2];
			self->encrypted_text[i_block * 8 + 4 + (i_byte / 2)] |= current_r;
		}

		// Finalny XOR prawej strony
		for (int i = 0; i < 4; i++) {
			self->encrypted_text[i_block * 8 + 4 + i] ^= self->_tmp_storage[i_block * 8 + i];
		}
	}
		
}

__device__ void DES_reverse(des_t* self) {
	char tmp;
	for (int i_block = 0; i_block < self->encrypted_text_length / 8; i_block++) {
		for (int i_byte = 0; i_byte < 4; i_byte++) {
			tmp = self->encrypted_text[i_block * 8 + i_byte];
			self->encrypted_text[i_block * 8 + i_byte] = self->encrypted_text[i_block * 8 + 4 + i_byte];
			self->encrypted_text[i_block * 8 + 4 + i_byte] = tmp;
		}
	}
}

__device__ void DES_final_permutation(des_t* self) {
	for (int i = 0; i < self->encrypted_text_length; i++) {
		self->_tmp_storage[i] = self->encrypted_text[i];
		self->encrypted_text[i] = 0;
	}
	int fp_index;
	char current_bit_value;

	for (int i_block = 0; i_block < self->encrypted_text_length / 8; i_block++) {
		for (int i_byte = 0, _index = 0; i_byte < 8; i_byte++) {

			for (int i_bit = 0; i_bit < 8; i_bit++, _index++) {
				fp_index = FINAL_PERMUTATION[_index] - 1;
				current_bit_value = (self->_tmp_storage[i_block * 8 + fp_index / 8] >> (7 - (fp_index % 8))) & 1;
				self->encrypted_text[i_block * 8 + i_byte] |= DES_BITS_CONJUCTIONS[i_bit] * current_bit_value;
			}
		}
	}
}

/*void DES_print_56(des_t* self) {
	unsigned char current_bit_value;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 8; j++) {
			current_bit_value = (self->_key56[i] >> (7 - j)) & 1;
			printf("%d", current_bit_value);

			if (i == 3 && j == 3) {
				printf(" | ");
			}
		}
		printf(" ");
	}
	printf("\n");
}
void DES_print_48(des_t* self) {
	unsigned char current_bit_value;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 8; j++) {
			current_bit_value = (self->_key48[i] >> (7 - j)) & 1;
			printf("%d", current_bit_value);
		}
		printf(" ");
	}
	printf("\n");
}
void DES_print_encrypted(des_t* self) {
	unsigned char current_bit_value;
	for (int i = 0; i < self->encrypted_text_length; i++) {
		for (int j = 0; j < 8; j++) {
			current_bit_value = (self->encrypted_text[i] >> (7 - j)) & 1;
			printf("%d", current_bit_value);
		}
		printf(" ");
	}
	printf("\n");
}
*/
__device__ void DES_encrypt(des_t* self) {
	DES_start(self);

	DES_initial_permutation(self);
	DES_key_to_56(self);

	for (int i = 0; i < 1; i++) {
		DES_shift_key56(self);
		DES_key56_to_48(self);
		DES_round(self);
	}
	/*DES_reverse(self);
	DES_final_permutation(self);*/
}

/// <summary>
/// Równoległe generowanie tablic tęczowych dla wybranego zakresu
/// </summary>
/// <param name="srainbow_table_index">Pierwszy indeks dla generowania tablic tęczowych (kolejna od zera tablica do wygenerowania)</param>
__global__ void GenerateRainbowTable(
	unsigned char** origins,
	unsigned char** keys_pointers,
	unsigned char** encoded_passwords_pointers,
	int key_size,
	int encoded_password_size,
	char* plain_password,
	int plain_password_length,
	int rainbow_table_index,
	unsigned char* test
) {
	unsigned char plain_text[8];
	plain_text[0] = 1;
	plain_text[1] = 35;
	plain_text[2] = 69;
	plain_text[3] = 103;
	plain_text[4] = 137;
	plain_text[5] = 171;
	plain_text[6] = 205;
	plain_text[7] = 239;
	int index = (blockIdx.x * 1024) + threadIdx.x;
	des_t* d = DES_init(8);
	d->plain_text = plain_text;

	int key_index = (rainbow_table_index)*RAINBOW_TABLE_SIZE + (blockIdx.x * 1024) + threadIdx.x;
	

	for (short i = 0; i < 8; i++) {
		d->key64[i] = key_index % 256;
		keys_pointers[index][i] = d->key64[i];
		key_index /= 256;
	}

	DES_encrypt(d);
	//for (short i = 0; i < 8; i++) {
	//	//encoded_passwords_pointers[index][i] = d->encrypted_text[i];
	//}
	DES_free(d);
	//if (threadIdx.x == 3) {
		encoded_passwords_pointers[index][0] = 'P';
		encoded_passwords_pointers[index][1] = 'A';
		encoded_passwords_pointers[index][2] = 'N';
		encoded_passwords_pointers[index][3] = '_';
		encoded_passwords_pointers[index][4] = 'C';
		encoded_passwords_pointers[index][5] = 'H';
		encoded_passwords_pointers[index][6] = 'U';
		encoded_passwords_pointers[index][7] = 'J';
	//}
}