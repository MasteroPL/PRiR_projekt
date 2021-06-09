#include "rainbow_table.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/// <summary>
/// Alokuje pamiec dla tablicy teczowej o wybranym rozmiarze
/// </summary>
/// <param name="key_size">Rozmiar (w bajtach) kluczy (zawsze 8)</param>
/// <param name="encoded_password_size">Rozmiar (w bajtach) zaszyfrowanej treœci hasla</param>
/// <param name="num_of_entries">Liczba wpisow do zaalokowania</param>
/// <returns>Wskaznik na zaalokowan¹ pamiec</returns>
rainbow_table_t* RainbowTable_allocate(short key_size, short encoded_password_size, int num_of_entries) {
	rainbow_table_t* result = (rainbow_table_t*)malloc(sizeof(rainbow_table_t));
	if (result == NULL) {
		return NULL;
	}
	int per_chunk = num_of_entries;
	int entry_size = sizeof(unsigned char) * key_size + sizeof(unsigned char) * encoded_password_size;

	result->key_size = key_size;
	result->encoded_password_size = encoded_password_size;
	result->nodes_size = num_of_entries;

	result->nodes = (rainbow_table_node_t*)malloc(sizeof(rainbow_table_node_t) * num_of_entries);

	if (result->nodes == NULL) {
		free(result);
		return NULL;
	}

	int cur_malloc_size = entry_size * num_of_entries;

	// Trzeba podzielic rezerwowan¹ pamiêc
	if (cur_malloc_size > MAX_MALLOC_SIZE) {
		int num_of_chunks = cur_malloc_size / MAX_MALLOC_SIZE;
		if (cur_malloc_size % MAX_MALLOC_SIZE > 0) {
			num_of_chunks++;
		}
		per_chunk = num_of_entries / num_of_chunks;
		if (num_of_entries % num_of_chunks > 0) {
			num_of_chunks++;
			per_chunk = num_of_entries / num_of_chunks;
			if (num_of_entries % num_of_chunks > 0) {
				per_chunk++;
			}
		}
		int tmp_entries = num_of_entries;

		result->_nodes_data_ref = (unsigned char**)malloc(sizeof(char*) * num_of_chunks);
		result->_num_of_refs = num_of_chunks;

		if (result->_nodes_data_ref == NULL) {
			free(result);
			return NULL;
		}

		result->_ref_sizes = (int*)malloc(sizeof(int) * num_of_chunks);

		if (result->_ref_sizes == NULL) {
			free(result->_nodes_data_ref);
			free(result);
			return NULL;
		}

		for (int i = 0; i < num_of_chunks; i++) {
			if (tmp_entries > per_chunk) {
				result->_nodes_data_ref[i] = (unsigned char*)malloc(
					entry_size * per_chunk
				);
				result->_ref_sizes[i] = per_chunk;
			}
			else {
				result->_nodes_data_ref[i] = (unsigned char*)malloc(
					entry_size * tmp_entries
				);
				result->_ref_sizes[i] = per_chunk;
			}

			if (result->_nodes_data_ref[i] == NULL) {
				for (int j = 0; j < i; j++) {
					free(result->_nodes_data_ref[i]);
				}
				free(result->_nodes_data_ref);
				free(result->_ref_sizes);
				free(result);
				return NULL;
			}

			tmp_entries -= per_chunk;
		}
	}
	// Wszystko mo¿na zarezerwowaæ w jednym miejscu
	else {
		result->_nodes_data_ref = (unsigned char**)malloc(sizeof(unsigned char*));
		if (result->_nodes_data_ref == NULL) {
			free(result);
			return NULL;
		}
		result->_ref_sizes = (int*)malloc(sizeof(int));
		if (result->_ref_sizes == NULL) {
			free(result->_nodes_data_ref);
			free(result);
			return NULL;
		}
		result->_nodes_data_ref[0] = (unsigned char*)malloc(cur_malloc_size);
		if (result->_nodes_data_ref[0] == NULL) {
			free(result->_nodes_data_ref);
			free(result->_ref_sizes);
			free(result);
			return NULL;
		}
		result->_num_of_refs = 1;
		result->_ref_sizes[0] = num_of_entries;
	}

	unsigned char** tmp_data_ref = result->_nodes_data_ref;

	result->nodes[0].key = &(tmp_data_ref[0][0]);
	result->nodes[0].encoded_password = &(tmp_data_ref[0][key_size]);

	// Przypisanie wskazañ na odpowiednie miejsca w pamiêci do wpisywania bajtów kluczy i zaszyfrowanych hase³
	int chunk_index = 0;
	int tmp_i = 1;
	for (int i = 1; i < num_of_entries; i++, tmp_i++) {
		if (i % per_chunk == 0) {
			chunk_index++;
			tmp_i = 0;
		}

		result->nodes[i].key = (unsigned char*)&(tmp_data_ref[chunk_index][tmp_i * (key_size + encoded_password_size)]);
		result->nodes[i].encoded_password = (unsigned char*)&(tmp_data_ref[chunk_index][tmp_i * (key_size + encoded_password_size) + key_size]);
	}

	return result;
}

void RainbowTable_free(rainbow_table_t* self) {
	for (int i = 0; i < self->_num_of_refs; i++) {
		free(self->_nodes_data_ref[i]);
	}

	free(self->_ref_sizes);
	free(self->_nodes_data_ref);
	free(self->nodes);
}

int RainbowTable_write_to_file(rainbow_table_t* self, const char* filename) {
	FILE* f = fopen(filename, "wb");

	// Sprawdzenie czy plik jest otwarty
	if (f == NULL) {
		return 1;
	}

	// Wpisywanie meta danych
	char meta_data[8];
	int* num_of_entries = (int*)&(meta_data[0]);
	short* key_size = (short*)&(meta_data[4]);
	short* password_size = (short*)&(meta_data[6]);

	*num_of_entries = self->nodes_size;
	*key_size = self->key_size;
	*password_size = self->encoded_password_size;

	fwrite(meta_data, 1, 8, f);

	for (int i = 0; i < self->_num_of_refs; i++) {
		fwrite(self->_nodes_data_ref[i], sizeof(unsigned char) * self->key_size + sizeof(unsigned char) * self->encoded_password_size, self->_ref_sizes[i], f);
	}

	fclose(f);
	return 0;
}


rainbow_table_t* RainbowTable_read_from_file(const char* filename) {
	FILE* f = fopen(filename, "rb");

	// Czy plik otwarty
	if (f == NULL) {
		return NULL;
	}

	char meta_data[8];
	
	fread(meta_data, 1, 8, f);
	int* num_of_entries = (int*)&(meta_data[0]);
	short* key_size = (short*)&(meta_data[4]);
	short* password_size = (short*)&(meta_data[6]);

	rainbow_table_t* t = RainbowTable_allocate(*key_size, *password_size, *num_of_entries);

	if (t == NULL) {
		// Nie mo¿na zarezerwowaæ pamiêci
		fclose(f);
		return NULL;
	}

	rainbow_table_node_t tmp_node;
	for (int i = 0; i < *num_of_entries; i++) {
		tmp_node = t->nodes[i];

		// Pamiêæ zosta³a ju¿ zaalokowana
		
		if (fread(tmp_node.key, 1, t->key_size, f) < 0) {
			// Nieprawid³owy plik
			fclose(f);
			return NULL;
		}
		if (fread(tmp_node.encoded_password, 1, t->encoded_password_size, f) < 0) {
			// Nieprawid³owy plik
			fclose(f);
			return NULL;
		}
	}

	fclose(f);
	return t;
}


void RainbowTable_cuda_allocate(rainbow_table_t* ref_rainbow_table, unsigned char*** keys_pointers, unsigned char*** encoded_passwords_pointers, unsigned char*** origins_refs) {
	rainbow_table_t* ref = ref_rainbow_table;
	int entry_size = sizeof(unsigned char) * ref->key_size + sizeof(unsigned char) * ref->encoded_password_size;

	cudaMalloc(keys_pointers, sizeof(unsigned char*) * ref->nodes_size);
	cudaMalloc(encoded_passwords_pointers, sizeof(unsigned char*) * ref->nodes_size);
	cudaMalloc(origins_refs, sizeof(unsigned char*) * ref->_num_of_refs);

	unsigned char** h_keys_pointers = (unsigned char**)malloc(sizeof(unsigned char*) * ref->nodes_size);
	unsigned char** h_encoded_passwords_pointers = (unsigned char**)malloc(sizeof(unsigned char*) * ref->nodes_size);
	unsigned char** h_origins_refs = (unsigned char**)malloc(sizeof(unsigned char*) * ref->_num_of_refs);
	for (int i = 0; i < ref->_num_of_refs; i++) {
		cudaMalloc(&((h_origins_refs)[i]), entry_size * ref->_ref_sizes[i]);
	}
	cudaMemcpy(*origins_refs, h_origins_refs, sizeof(unsigned char*) * ref->_num_of_refs, cudaMemcpyHostToDevice);
	cudaMemcpy(h_origins_refs, *origins_refs, sizeof(unsigned char*) * ref->_num_of_refs, cudaMemcpyDeviceToHost);

	int cur_size = ref->_ref_sizes[0];
	int cur_tmp = 0;
	int cur_ref = 0;
	for (int i = 0; i < ref->nodes_size; i++, cur_tmp++) {
		if (cur_tmp == cur_size) {
			cur_tmp = 0;
			cur_ref++;
			cur_size = ref->_ref_sizes[cur_ref];
		}

		if (i == 1024) {
			int c = 0;
		}

		int a = cur_tmp * entry_size;
		int b = (cur_tmp * entry_size) + ref->key_size;
		(h_keys_pointers)[i] = &(h_origins_refs)[cur_ref][cur_tmp * entry_size];
		(h_encoded_passwords_pointers)[i] = &(h_origins_refs)[cur_ref][(cur_tmp * entry_size) + ref->key_size];
	}

	cudaMemcpy(*keys_pointers, h_keys_pointers, sizeof(unsigned char**) * ref->nodes_size, cudaMemcpyHostToDevice);
	cudaMemcpy(*encoded_passwords_pointers, h_encoded_passwords_pointers, sizeof(unsigned char**) * ref->nodes_size, cudaMemcpyHostToDevice);

	free(h_keys_pointers);
	free(h_encoded_passwords_pointers);
	free(h_origins_refs);
}

void RainbowTable_cuda_copy_results_to_host(rainbow_table_t* ref_rainbow_table, unsigned char** origins_refs) {
	rainbow_table_t* ref = ref_rainbow_table;
	int entry_size = sizeof(unsigned char) * ref->key_size + sizeof(unsigned char) * ref->encoded_password_size;

	unsigned char** h_origins_refs = (unsigned char** )malloc(sizeof(unsigned char*) * ref->_num_of_refs);
	cudaMemcpy(h_origins_refs, origins_refs, sizeof(unsigned char*) * ref->_num_of_refs, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ref->_num_of_refs; i++) {
		cudaMemcpy(ref->_nodes_data_ref[i], h_origins_refs[i], entry_size * ref->_ref_sizes[i], cudaMemcpyDeviceToHost);
	}

	free(h_origins_refs);
}

void RainbowTable_cuda_free(rainbow_table_t* ref_rainbow_table, unsigned char** keys_pointers, unsigned char** encoded_passwords_pointers, unsigned char** origins_refs) {
	rainbow_table_t* ref = ref_rainbow_table;

	cudaFree(keys_pointers);
	cudaFree(encoded_passwords_pointers);

	unsigned char** h_origins_refs = (unsigned char**)malloc(sizeof(unsigned char*) * ref->_num_of_refs);
	cudaMemcpy(h_origins_refs, origins_refs, sizeof(unsigned char*) * ref->_num_of_refs, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ref->_num_of_refs; i++) {
		cudaFree(h_origins_refs[i]);
	}
	cudaFree(origins_refs);
}