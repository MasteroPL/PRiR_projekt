//używam MSB

//-----------------------------------------------------------------------
//                           IMPORTANT
// nsight nie widzi zaalokowanych przez karte graficzną tabeli, ale można 
// przekopiować do tabeli stworzonej przez cudaMalloc i wtedy pokazuje wartości
//---------------------------------------------------------------------------



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#include "rainbow_table.h"

#define MIK_WARP_PER_SM 4
#define MIK_SM_NM 28
#define THREADS_IN_WARP 32

/*__device__ int IP[] =
{
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
};
obecnie zbędne
przerobiłem fragment matematycznie, jak się okazało jest w miarę regularne
*/

__device__ int E[] =
{
    32,  1,  2,  3,  4,  5,
    4,  5,  6,  7,  8,  9,
    8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
};

__device__ int P[] =
{
    16,  7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2,  8, 24, 14,
    32, 27,  3,  9,
    19, 13, 30,  6,
    22, 11,  4, 25
};

__device__ int FP[] =
{
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
};

__device__ int s_boxes[8][4][16] = { 
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

__device__ int PC1[] =
{
    57, 49, 41, 33, 25, 17,  9,
    1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
};

__device__ int PC2[] =
{
    14, 17, 11, 24,  1,  5,
    3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8,
    16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
};

__device__ int SHIFTS[] = { 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };

//MSB
__device__ unsigned char bits[] = { 128,64,32,16,8,4,2,1};

__device__ int bitsIP[] = { 1,3,5,7,0,2,4,6 };


/// <summary>
/// usuwa bity parzystości z klucza oraz wykonuje odpowiednie przestawienie bitów
/// </summary>
/// <param name="key">64-bitowy klucz rozbity na pojedyńcze bity w wektorze unsigned char</param>
/// <returns>unsigned char[56], produkt potrzebny do dalszego ustalenie klucza dla danego cyklu</returns>
__device__ unsigned char* key_to_56(unsigned char key[64]) {
    unsigned char* key56 = (unsigned char*)malloc(sizeof(unsigned char) * 56);
    for (int i = 0; i < 56; i++) {
        key56[i] = key[PC1[i]-1];
    }
    return key56;
}

/// <summary>
/// przesuwa połówki klucza o odpowiednią ilość bitów
/// </summary>
/// <param name="key56"> wynik funkcji key_to_56</param>
/// <param name="cicle">numer cyklu</param>
/// <returns>przesunięty klucz 56</returns>
__device__ void key_shift(unsigned char* key56_permuted, int cicle) {
    unsigned int tmp;
    unsigned int tmp2;

    switch (SHIFTS[cicle]) {
    case 1:
        tmp = key56_permuted[0];
        tmp2 = key56_permuted[28];
        for (int i = 1; i < 27; i++) {
            key56_permuted[i - 1] = key56_permuted[i];
            key56_permuted[i + 27] = key56_permuted[i + 28];
        }
        key56_permuted[27] = tmp;
        key56_permuted[55] = tmp2;
        break;
    case 2:
        tmp = key56_permuted[0];
        tmp2 = key56_permuted[1];
        unsigned int tmp3 = key56_permuted[28];
        unsigned int tmp4 = key56_permuted[29];
        for (int i = 2; i < 26; i++) {
            key56_permuted[i - 2] = key56_permuted[i];
            key56_permuted[i + 25] = key56_permuted[i + 30];
        }
        key56_permuted[26] = tmp;
        key56_permuted[27] = tmp2;
        key56_permuted[54] = tmp3;
        key56_permuted[55] = tmp4;
        break;
    }
    
}

/// <summary>
/// permutacja pc2 zmniejszająca długość klucza do 48 bitów;
/// </summary>
/// <param name="key">wynik funkcji key_shift</param>
/// <returns>klucz w postaci wektora unsigned char [6] bity połączone potrzebny do funkcji feistela dla opowiedniej iteracji</returns>
__device__ unsigned char* key_to_48(unsigned char key[56], unsigned char* test) {
    for (int i = 0; i < 8; i++) {
        test[i] = key[i];
    }
    unsigned char key_48[48];
    for (int i = 0; i < 48; i++) {
        key_48[i] = key[PC2[i] - 1];
    }

    unsigned char* key_final = (unsigned char*)malloc(sizeof(unsigned char) * 6);
    if (key_final){
        for (int i = 0; i < 6; i++) {
            key_final[i] = 0;
        }
    }


    for (int i = 0; i < 48; i++) {
        int target_byte = (int)i / 8;
        int target_bit = i % 8;
        key_final[target_byte] = key_final[target_byte] | (key_48[i] << (7 - target_bit));
    }
    return key_final;
}

/// <summary>
/// początkowa permutacja na wiadomości
/// </summary>
/// <param name="plain">niezaszywrowana wiadomość</param>
/// <returns>wiadomość do zaszywrowania z poprzestawianymi bitami</returns>
__device__ unsigned char* initial_permutation(unsigned char plain[8]) {
   
    unsigned char* plain_permuted = (unsigned char*) malloc(8);
    for (int i = 0; i < 8; i++) {
        plain_permuted[i] = 0;
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 7; j >= 0; j--) {
            int bit_shift = 7 - j - bitsIP[i];
            if (bit_shift >= 0){
                plain_permuted[i] |= ((plain[j] & bits[bitsIP[i]]) >> bit_shift);
            }
            else {
                plain_permuted[i] |= ((plain[j] & bits[bitsIP[i]]) << (-bit_shift));
            }

        }
    }
    return plain_permuted;
}

/// <summary>
/// funkcja feistela, albo przynajmniej jej część
/// </summary>
/// <param name="key">klucz 48 bitowy, wynik funkcji key_to_48</param>
/// <param name="text">prawa połowa wiadomości</param>
/// <returns>wynik funkcji feistela</returns>
__device__ unsigned char* feistel(unsigned char* key, unsigned char* text, unsigned char* test) {
    // permutacja rozszerzająca
    unsigned char right_extended[6];
    for (int i = 0; i < 6; i++) {
        right_extended[i] = 0;
    }
    for (int i = 0; i < 48; i++) {
        int target_byte = (int)(i / 8);
        int target_bit = i % 8;
        int source_byte = (int)((E[i] -1) / 8);
        int source_bit = (E[i] -1) % 8;
        if ((target_bit - source_bit) >= 0) {
            right_extended[target_byte] |= ((text[source_byte] & bits[source_bit]) >> (target_bit - source_bit));
        }
        else
        {
            right_extended[target_byte] |= ((text[source_byte] & bits[source_bit]) << (source_bit - target_bit));
        }

    }
    // xor prawej części z kluczem
    unsigned char right_xored[6];
    for (int i = 0; i < 6; i++) {
        right_xored[i] = right_extended[i] ^ key[i];
    }
    // s-blox
    // tak wiem że zapewne da się to zrobić lepiej
    unsigned char right_s_box[4];
    for (int i = 0; i < 4; i++) {
        right_s_box[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
        int y1 = ((right_xored[i*3+0] & bits[0]) >> 6) | ((right_xored[i * 3 + 0] & bits[5]) >> 2);
        int y2 = (right_xored[i * 3 + 0] & bits[6]) | ((right_xored[i * 3 + 1] & bits[3]) >> 4);
        int y3 = ((right_xored[i * 3 + 1] & bits[4]) >> 2) | ((right_xored[i * 3 + 2] & bits[1]) >> 6);
        int y4 = ((right_xored[i * 3 + 2] & bits[2]) >> 4) | (right_xored[i * 3 + 2] & bits[7]);
        int x1 = ((bits[1] | bits[2] | bits[3] | bits[4]) & right_xored[i * 3 + 0]) >> 3;
        int x2 = ((right_xored[i * 3 + 0] & bits[7]) << 3) | ((right_xored[i * 3 + 1] & (bits[0] | bits[1] | bits[2])) >> 5);
        int x3 = ((right_xored[i * 3 + 1] & (bits[5] | bits[6] | bits[7])) << 1) | ((right_xored[i * 3 + 2] & bits[0]) >> 7);
        int x4 = (right_xored[i * 3 + 2] &(bits[3] | bits[4] | bits[5] | bits[6])) >> 1;
        right_s_box[i * 2 + 0] = ((s_boxes[i * 4 + 0][y1][x1] << 4) | (s_boxes[i * 4 + 1][y2][x2]));
        right_s_box[i * 2 + 1] = ((s_boxes[i * 4 + 2][y3][x3] << 4) | (s_boxes[i * 4 + 3][y4][x4]));
    }
    //permutacja P
    unsigned char* right_final = (unsigned char*)malloc(sizeof(unsigned char) * 4);
    for (int i = 0; i < 4; i++) {
        right_final[i] = 0;
    }
    for (int i = 0; i < 32; i++) {
        int target_byte = (int)(i / 8);
        int target_bit = i % 8;
        int source_byte = (int)((P[i] - 1) / 8);
        int source_bit = (P[i] - 1) % 8;
        if ((target_bit - source_bit) >= 0) {
            right_final[target_byte] |= ((right_s_box[source_byte] & bits[source_bit]) >> (target_bit - source_bit));
        }
        else {
            right_final[target_byte] |= ((right_s_box[source_byte] & bits[source_bit]) << (source_bit - target_bit));
        }
        
    }
    return right_final;
}

/// <summary>
/// szyfrowanie DES
/// </summary>
/// <param name="key">in: 64-bitowy klucz rozbity na pojedyncze bity w wektorze unsigned char[64] </param>
/// <param name="text">in: 64- bitowa wiadomość do zaszywrowania</param>
/// <param name="finale">out: wynik szyfrowania</param>
/// <returns></returns>
__device__ void DESCipher(unsigned char key[64], unsigned char text[8], unsigned char finale[8], unsigned char* test) {

    unsigned char* plain_permuted = initial_permutation(text);
    unsigned char* key_56 = key_to_56(key);
    unsigned char left[4] = { plain_permuted[0] ,plain_permuted[1] ,plain_permuted[2] ,plain_permuted[3]};
    unsigned char right[4] = { plain_permuted[4] ,plain_permuted[5], plain_permuted[6] ,plain_permuted[7]};
    //część cykliczna
    for (int i = 0; i < 16; i++) {
        // w sumie to to jest funkcja feistela
        key_shift(key_56,i);
        unsigned char* key_48 = key_to_48(key_56,test);
        unsigned char* feistel_r = feistel(key_48, right, test);
        unsigned char l_xor_f[4];
        // xor r_next i left
        for (int i = 0; i < 4; i++) {
            l_xor_f[i] = left[i] ^ feistel_r[i];
        }
        // lewa połowa danych staje się nową prawą połową, natomiast poprzednia prawa połowa staje się nową lewą połową
        for (int i = 0; i < 4; i++) {
            left[i] = right[i];
            right[i] = l_xor_f[i];
        }
        free(key_48);
        free(feistel_r);

    }
    // połączenie lewaprawa
    unsigned char po_feistelu[8] = { left[0] ,left[1] ,left[2] ,left[3], right[0] ,right[1] ,right[2] ,right[3] };


    for (int i = 0; i < 64; i++) {
        int target_byte = (int)(i / 8);
        int target_bit = i % 8;
        int source_byte = (int)((FP[i]-1) / 8);
        int source_bit = (FP[i]-1) % 8;
        if ((target_bit - source_bit) >= 0) {
            finale[target_byte] |= (po_feistelu[source_byte] & bits[source_bit]) >> (target_bit - source_bit);
        }
        else {
            finale[target_byte] |= (po_feistelu[source_byte] & bits[source_bit]) << (source_bit - target_bit);
        }

    }
    free(plain_permuted);
    free(key_56);
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
    unsigned char key_bytes[8];
    unsigned char key_bits[64];
    unsigned char encoded[8];
    unsigned char text[8] = { '1', '2', '3', '4', '5', '6', '7', '9' }; // debug

    int key_index = (rainbow_table_index) * RAINBOW_TABLE_SIZE + blockIdx.x;

    for (short i = 0; i < 8; i++) {
        key_bytes[i] = key_index % 256;
        keys_pointers[blockIdx.x][i] = key_bytes[i];
        key_index /= 256;
    }

    unsigned char cur_byte;
    for (short i = 0; i < 8; i++) {
        cur_byte = key_bytes[i];

        for (short j = 0; j < 8; j++) {
            key_bits[i * 8 + 7 - j] = cur_byte % 2;
            cur_byte /= 2;
        }
    }

    DESCipher(key_bits, text, encoded, test);
    for (short i = 0; i < 8; i++) {
        encoded_passwords_pointers[blockIdx.x][i] = encoded[i];
    }
    if (blockIdx.x == 3) {
        encoded_passwords_pointers[blockIdx.x][0] = 'P';
        encoded_passwords_pointers[blockIdx.x][1] = 'A';
        encoded_passwords_pointers[blockIdx.x][2] = 'N';
        encoded_passwords_pointers[blockIdx.x][3] = '_';
        encoded_passwords_pointers[blockIdx.x][4] = 'C';
        encoded_passwords_pointers[blockIdx.x][5] = 'H';
        encoded_passwords_pointers[blockIdx.x][6] = 'U';
        encoded_passwords_pointers[blockIdx.x][7] = 'J';
    }
}