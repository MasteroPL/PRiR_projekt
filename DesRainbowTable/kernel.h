/// <summary>
/// szyfrowanie DES
/// </summary>
/// <param name="key">in: 64-bitowy klucz rozbity na pojedyncze bity w wektorze unsigned char[64] </param>
/// <param name="text">in: 64- bitowa wiadomoœæ do zaszywrowania</param>
/// <param name="finale">out: wynik szyfrowania</param>
/// <returns></returns>
__global__ void DESCipher(unsigned char key[64], unsigned char text[8], unsigned char finale[8]);