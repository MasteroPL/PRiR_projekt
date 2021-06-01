#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rainbow_table.h"

#include <stdio.h>
#include <stdlib.h>

#include "indexer.cuh"

int main() {
	/*unsigned char key[8];
	for (int i = 0; i < 5000; i += 1) {
		printf("Klucz[%d]: ", i);
		getKeyForIndex(i, 8, key);
		for (int j = 0; j < 8; j++) {
			printf("%d ", key[j]);
		}
		printf("\n");
	}*/

	//rainbow_table_t* t = RainbowTable_allocate(8, 8, 20);
	rainbow_table_t* t = RainbowTable_allocate(8, 16, 1000000);

	for (int i = 0; i < 20; i++) {

		t->nodes[i].key[0] = (char)(65 + i);
		t->nodes[i].encoded_password[0] = (char)(97 + i);
		t->nodes[i].key[7] = '\0';
		t->nodes[i].encoded_password[7] = '\0';

		for (int j = 0; j < 6; j++) {
			t->nodes[i].key[1 + j] = (char)(48 + j);
			t->nodes[i].encoded_password[1 + j] = (char)(48 + j);
		}
	}

	for (int i = 0; i < 20; i++) {
		printf("%i: %s | %s\n", i, t->nodes[i].key, t->nodes[i].encoded_password);
	}
	/*

	RainbowTable_write_to_file(t, "test.txt");*/

	RainbowTable_free(t);

	return 0;
}