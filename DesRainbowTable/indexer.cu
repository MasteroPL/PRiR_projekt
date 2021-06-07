#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "indexer.cuh"

void getKeyForIndex(unsigned long index, unsigned short numBytes, unsigned unsigned char* outBytes) {
	for (short i = 0; i < numBytes; i++) {
		outBytes[i] = index % 256;
		index /= 256;
	}
}