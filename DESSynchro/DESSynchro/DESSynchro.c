// DESSynchro.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "rainbow_table.h"
#include "des.h"

#define DEFAULT_PASSWORD "PRiR_fn!"

void getKeyForIndex(unsigned long index, unsigned short numBytes, unsigned unsigned char* outBytes) {
	for (short i = 0; i < numBytes; i++) {
		outBytes[i] = index % 256;
		index /= 256;
	}
}

int main(int argc,char** argv){
	rainbow_table_t* t = RainbowTable_allocate(8, 8, RAINBOW_TABLE_SIZE);
	unsigned char* plain_text = malloc(sizeof(unsigned char) * 8);

	for (int i = 0; i < 8; i++) {
		plain_text[i] = (unsigned char)DEFAULT_PASSWORD[i];
	}

	clock_t begin = clock();

	des_t* d = DES_init(8);
	d->plain_text = plain_text;

	for (int i = 0; i < RAINBOW_TABLE_SIZE; i++) {
		getKeyForIndex(i, 8, d->key64);
		DES_encrypt(d);
	}

	DES_free(d);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Czas trwania: %lf\n", time_spent);

	free(plain_text);
	
}


