#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "indexer.cuh"

int main() {
	unsigned char key[8];
	for (int i = 0; i < 5000; i += 1) {
		printf("Klucz[%d]: ", i);
		getKeyForIndex(i, 8, key);
		for (int j = 0; j < 8; j++) {
			printf("%d ", key[j]);
		}
		printf("\n");
	}

	return 0;
}