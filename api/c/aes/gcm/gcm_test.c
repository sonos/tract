#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "mbedtls/entropy.h"    // mbedtls_entropy_context
#include "mbedtls/ctr_drbg.h"   // mbedtls_ctr_drbg_context
#include "mbedtls/cipher.h"     // MBEDTLS_CIPHER_ID_AES
#include "mbedtls/gcm.h"        // mbedtls_gcm_context

#define KEY_BYTES 32
#define IV_BYTES 12
#define TAG_BYTES 16
#define BUF_SIZE 128

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model>\n", argv[0]);
        return 1;
    }

    // read the file and put it in a buffer
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "Model opening failed\n");
        return 1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "fseek failed\n");
        fclose(fp);
        return 1;
    }

    long plain_len = ftell(fp);
    if (plain_len < 0) {
        fprintf(stderr, "ftell failed\n");
        fclose(fp);
        return 1;
    }

    // Reset file position to the beginning
    rewind(fp);

    unsigned char *model = (unsigned char *)malloc(plain_len + 1);
    if (!model) {
        fprintf(stderr, "Memory allocation for model failed\n");
        return 1;
    }
    size_t read_len = fread(model, 1, plain_len, fp);
    if (read_len != (size_t)(plain_len)) {
        fprintf(stderr, "fread failed\n");
        free(model);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    mbedtls_ctr_drbg_context ctr_drbg;
    mbedtls_entropy_context entropy;
    mbedtls_gcm_context gcm;

    unsigned char key[KEY_BYTES + 1];
    unsigned char iv[IV_BYTES + 1];
    unsigned char add_data[BUF_SIZE];
    unsigned char *output = malloc(plain_len + 1);
    unsigned char *decrypted = malloc(plain_len + 1);
    unsigned char tag[TAG_BYTES];

    if (!output || !decrypted) {
        fprintf(stderr, "Memory allocation failed\n");
        goto exit;
    }

    // The personalization string should be unique to the application in order to add some
    // personalized starting randomness to the random sources.
    char *pers = "aes generate key";
    int ret;

    mbedtls_entropy_init(&entropy);
    mbedtls_ctr_drbg_init(&ctr_drbg);
    mbedtls_gcm_init(&gcm);

    // Seed the random number generator
    ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, (unsigned char *)pers, strlen(pers));
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_seed() failed - returned -0x%04x\n", -ret);
        goto exit;
    }

    memset(key, 0, KEY_BYTES);
    memset(iv, 0, IV_BYTES);

    // Generate random bytes for the key (32 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, key, KEY_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract key - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Generate random bytes for the IV (12 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, iv, IV_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract IV - returned -0x%04x\n", -ret);
        goto exit;
    }
    
    fprintf(stderr, "plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        fprintf(stderr, "%02x", model[i]);
    }
    fprintf(stderr, "\n");

    // Fill the additional data with something
    memset(add_data, 0, BUF_SIZE);
    snprintf((char *)add_data, BUF_SIZE, "authenticated but not encrypted payload");
    size_t add_data_len = strlen((char *)add_data);
    fprintf(stderr, "additional: '%s'  (length %zu)\n", add_data, add_data_len);


    // Initialize the GCM context with our key and desired cipher
    ret = mbedtls_gcm_setkey(&gcm,                      // GCM context to be initialized
                             MBEDTLS_CIPHER_ID_AES,     // cipher to use (a 128-bit block cipher)
                             key,                       // encryption key
                             KEY_BYTES * 8);            // key bits
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher - returned -0x%04x\n", -ret);
        goto exit;
    }

    // GCM buffer encryption using a block cipher (NOTE: GCM mode doesn't require padding)
    ret = mbedtls_gcm_crypt_and_tag( &gcm,                // GCM context
                                     MBEDTLS_GCM_ENCRYPT, // mode
                                     plain_len,           // length of input data
                                     iv,                  // initialization vector
                                     IV_BYTES,            // length of IV
                                     add_data,            // additional data
                                     add_data_len,        // length of additional data
                                     model,               // buffer holding the input data
                                     output,              // buffer for holding the output data
                                     TAG_BYTES,           // length of the tag to generate
                                     tag);                // buffer for holding the tag
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_crypt_and_tag failed to encrypt the data - returned -0x%04x\n", -ret);
        goto exit;
    }
    fprintf(stderr, "ciphertext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        fprintf(stderr, "%02x", output[i]);
    }
    fprintf(stderr, "\n");


    // Uncomment this line to corrupt the add_data so that GCM will fail to authenticate on decryption
    //memset(add_data, 0, BUF_SIZE);

    // GCM buffer authenticated decryption using a block cipher
    ret = mbedtls_gcm_auth_decrypt(&gcm,                // GCM context
                                    plain_len,          // length of the input ciphertext data (always same as plain)
                                    iv,                 // initialization vector
                                    IV_BYTES,           // length of IV
                                    add_data,           // additional data
                                    add_data_len,       // length of additional data
                                    tag,                // buffer holding the tag
                                    TAG_BYTES,          // length of the tag
                                    output,             // buffer holding the input ciphertext data
                                    decrypted);         // buffer for holding the output decrypted data
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_auth_decrypt failed to decrypt the ciphertext - tag doesn't match\n");
        goto exit;
    }

    fprintf(stderr, "plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        fprintf(stderr, "%02x", decrypted[i]);
    }
    fprintf(stderr, "\n");
    if (memcmp(model, decrypted, plain_len) != 0) {
        fprintf(stderr, "Decrypted data doesn't match the original data!\n");
        goto exit;
    }
    fprintf(stderr, "Decrypted data matches the original data!\n");


exit:
    free(model);
    free(output);
    free(decrypted);

    mbedtls_entropy_free(&entropy);
    mbedtls_ctr_drbg_free(&ctr_drbg);
    mbedtls_gcm_free(&gcm);

    return ret;
}