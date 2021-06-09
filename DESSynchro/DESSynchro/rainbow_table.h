
#ifndef _RAINBOW_TABLE_H__
#define _RAINBOW_TABLE_H__
#define MAX_MALLOC_SIZE 16711568
#define RAINBOW_TABLE_SIZE 1048576

typedef struct RainbowTableNode {
	unsigned char* key;
	unsigned char* encoded_password;
} rainbow_table_node_t;

typedef struct RainbowTable {
	short key_size; // Tu zawsze bedzie 8
	short encoded_password_size; // Tutaj w zale¿noœci od has³a sta³a wartoœæ dla s³ownika
	int nodes_size;

	int _num_of_refs;
	unsigned char** _nodes_data_ref;
	int* _ref_sizes;

	rainbow_table_node_t* nodes;
} rainbow_table_t;

/// <summary>
/// Alokuje pamiec dla tablicy teczowej o wybranym rozmiarze
/// </summary>
/// <param name="key_size">Rozmiar (w bajtach) kluczy (zawsze 8)</param>
/// <param name="encoded_password_size">Rozmiar (w bajtach) zaszyfrowanej treœci hasla</param>
/// <param name="num_of_entries">Liczba wpisow do zaalokowania</param>
/// <returns>Wskaznik na zaalokowan¹ pamiec</returns>
rainbow_table_t* RainbowTable_allocate(short key_size, short encoded_password_size, int num_of_entries);

/// <summary>
/// Czyœci pamiêæ wybranej tablicy
/// </summary>
void RainbowTable_free(rainbow_table_t* self);

/// <summary>
/// Wypisuje tablicê do pliku
/// </summary>
/// <param name="filename">Nazwa pliku do wypisania</param>
/// <returns>0 - zapis udany, 1 - b³¹d otwarcia pliku, 2 - b³¹d zapisu</returns>
int RainbowTable_write_to_file(rainbow_table_t* self, const char* filename);

/// <summary>
/// Wczytuje tablicê têczow¹ z pliku
/// </summary>
/// <param name="filename">Nazwa pliku do wczytania</param>
/// <returns>WskaŸnik do wczytanego obiektu lub NULL, jeœli wyczytywanie siê nie powiedzie</returns>
rainbow_table_t* RainbowTable_read_from_file(const char* filename);

void RainbowTable_cuda_allocate(rainbow_table_t* ref_rainbow_table, unsigned char*** keys_pointers, unsigned char*** encoded_passwords_pointers, unsigned char*** origins_refs);

void RainbowTable_cuda_copy_results_to_host(rainbow_table_t* ref_rainbow_table, unsigned char** origins_refs);

void RainbowTable_cuda_free(rainbow_table_t* ref_rainbow_table, unsigned char** keys_pointers, unsigned char** encoded_passwords_pointers, unsigned char** origins_refs);

#endif