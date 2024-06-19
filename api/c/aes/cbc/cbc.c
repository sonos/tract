#include "common.h"
#include <string.h>

#include "mbedtls/aes.h"
#include "mbedtls/platform.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"
#include "mbedtls/platform.h"
#include "ctr.h"
#include <unistd.h>

unsigned char *
mbedtls_aes_ecnrypt(mbedtls_aes_context *aes_encrypt, uint8_t *key, uint8_t *iv, unsigned char *plain_text, int data_length)
{
    mbedtls_aes_setkey_enc(aes_encrypt, key, 256);
    unsigned char *encrypted = (unsigned char *)malloc(data_length);
    if (encrypted == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    mbedtls_aes_crypt_cbc(aes_encrypt, MBEDTLS_AES_ENCRYPT, data_length, iv, plain_text, encrypted);

    return encrypted;
}

unsigned char *
mbedtls_aes_decrypt(mbedtls_aes_context *aes_decrypt, uint8_t *key, uint8_t *iv, unsigned char *encrypted, int data_length)
{
    mbedtls_aes_setkey_dec(aes_decrypt, key, 256);
    unsigned char *decrypted = (unsigned char *)malloc(data_length);
    if (decrypted == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    mbedtls_aes_crypt_cbc(aes_decrypt, MBEDTLS_AES_DECRYPT, data_length, iv, encrypted, decrypted);

    return decrypted;
}

uint8_t *
generate_random_data(int length)
{
    uint8_t *data = (uint8_t *)malloc(length);
    if (!data) {
        fprintf(stderr, "Memory allocation for data failed\n");
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        data[i] = rand() % 256;
    }

    return data;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model>\n", argv[0]);
        return 1;
    }

    // read the file and put it in a buffer
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Model opening failed\n");
        return 1;
    }

    int file_size;
    if ((file_size = lseek(fileno(fp), 0, SEEK_END)) < 0) {
        fprintf(stderr, "lseek failed\n");
        return 1;
    }

    unsigned char *model = (unsigned char *)malloc(file_size);
    if (!model) {
        fprintf(stderr, "Memory allocation for model failed\n");
        return 1;
    }
    fread(model, 1, file_size, fp);
    fclose(fp);


    uint8_t *key = generate_random_data(32);

    mbedtls_aes_context aes_encrypt, aes_decrypt;
    uint8_t *iv = generate_random_data(16);
    uint8_t iv1[16];
    memcpy(iv1, iv, 16);


    // padding below
    if((file_size % 16) != 0) file_size += 16 - (file_size % 16);

    unsigned char *encrypted = mbedtls_aes_ecnrypt(&aes_encrypt, key, iv, model, file_size);

    unsigned char *decrypted = mbedtls_aes_decrypt(&aes_decrypt, key, iv1, encrypted, file_size);
    
    if (strcmp((char *)decrypted, (char *)model) == 0)
        fprintf(stderr, "SUCCESS\n");
    else 
        fprintf(stderr, "FAILURE\n");
    
    free(iv);
    free(key);
    free(model);
    free(encrypted);
    free(decrypted);

    return 0;
}