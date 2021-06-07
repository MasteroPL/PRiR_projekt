#ifndef _KERNEL_H__
#define _KERNEL_K__
#include "rainbow_table.h"

/// <summary>
/// szyfrowanie DES
/// </summary>
/// <param name="key">in: 64-bitowy klucz rozbity na pojedyncze bity w wektorze unsigned char[64] </param>
/// <param name="text">in: 64- bitowa wiadomoœæ do zaszywrowania</param>
/// <param name="finale">out: wynik szyfrowania</param>
/// <returns></returns>
__device__ void DESCipher(unsigned char key[64], unsigned char text[8], unsigned char finale[8], unsigned char* test);


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