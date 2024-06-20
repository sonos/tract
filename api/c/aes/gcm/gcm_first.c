#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "mbedtls/entropy.h"    // mbedtls_entropy_context
#include "mbedtls/ctr_drbg.h"   // mbedtls_ctr_drbg_context
#include "mbedtls/cipher.h"     // MBEDTLS_CIPHER_ID_AES
#include "mbedtls/gcm.h"        // mbedtls_gcm_context

#define mbedtls_printf printf
#define KEY_BYTES 32
#define KEY_BITS KEY_BYTES * 8
#define IV_BYTES 12
#define TAG_BYTES 16
#define ADD_DATA_BYTES 64

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

    unsigned char key[KEY_BYTES];
    unsigned char iv[IV_BYTES];
    unsigned char add_data[ADD_DATA_BYTES];
    unsigned char *output = malloc(plain_len);
    unsigned char *decrypted = malloc(plain_len);
    unsigned char tag[TAG_BYTES];
    int ret;

    if (!output || !decrypted) {
        fprintf(stderr, "Memory allocation failed\n");
        ret = 1;
        goto exit;
    }

    // The personalization string should be unique to the application in order to add some
    // personalized starting randomness to the random sources.
    char *pers = "aes generate key";

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
    memset(add_data, 0, ADD_DATA_BYTES);
   
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

    // Generate random bytes for the additional_data (64 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, add_data, ADD_DATA_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract add_data - returned -0x%04x\n", -ret);
        goto exit;
    }
    
    mbedtls_printf("plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", model[i]);
    }
    mbedtls_printf("\n");

    // Initialize the GCM context with our key and desired cipher
    ret = mbedtls_gcm_setkey(&gcm,                      // GCM context to be initialized
                             MBEDTLS_CIPHER_ID_AES,     // cipher to use (a 128-bit block cipher)
                             key,                       // encryption key
                             KEY_BITS);                 // key bits
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher - returned -0x%04x\n", -ret);
        goto exit;
    }

    // GCM buffer encryption using a block cipher (NOTE: GCM mode doesn't require padding)
    ret = mbedtls_gcm_crypt_and_tag(&gcm,                // GCM context
                                     MBEDTLS_GCM_ENCRYPT, // mode
                                     plain_len,           // length of input data
                                     iv,                  // initialization vector
                                     IV_BYTES,            // length of IV
                                     add_data,            // additional data
                                     ADD_DATA_BYTES,      // length of additional data
                                     model,               // buffer holding the input data
                                     output,              // buffer for holding the output data
                                     TAG_BYTES,           // length of the tag to generate
                                     tag);                // buffer for holding the tag
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_crypt_and_tag failed to encrypt the data - returned -0x%04x\n", -ret);
        goto exit;
    }

    mbedtls_printf("ciphertext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", output[i]);
    }
    mbedtls_printf("\n");

    mbedtls_gcm_free(&gcm);

    mbedtls_gcm_init(&gcm);

    // Initialize the GCM context with our key and desired cipher
    ret = mbedtls_gcm_setkey(&gcm,                      // GCM context to be initialized
                             MBEDTLS_CIPHER_ID_AES,     // cipher to use (a 128-bit block cipher)
                             key,                       // encryption key
                             KEY_BITS);                 // key bits
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Uncomment this line to corrupt buffers so that GCM will fail to authenticate on decryption
    // memset(add_data, 0, ADD_DATA_BYTES);
    // memset(tag, 0, TAG_BYTES);
    // memset(iv, 0, IV_BYTES);
    // memset(key, 0, KEY_BYTES); // key must be changes before calling mbedtls_gcm_setkey

    // GCM buffer authenticated decryption using a block cipher
    ret = mbedtls_gcm_auth_decrypt(&gcm,                // GCM context
                                    plain_len,          // length of the input ciphertext data (always same as plain)
                                    iv,                 // initialization vector
                                    IV_BYTES,           // length of IV
                                    add_data,           // additional data
                                    ADD_DATA_BYTES,     // length of additional data
                                    tag,                // buffer holding the tag
                                    TAG_BYTES,          // length of the tag
                                    output,             // buffer holding the input ciphertext data
                                    decrypted);         // buffer for holding the output decrypted data
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_auth_decrypt failed to decrypt the ciphertext - tag doesn't match\n");
        goto exit;
    }

    mbedtls_printf("plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", decrypted[i]);
    }
    mbedtls_printf("\n");
    
    if (memcmp(decrypted, model, plain_len) != 0) {
        ret = 1;
        goto exit;
    } else {
        mbedtls_printf("SUCCESS\n");
    }

    ret = 0;

exit:
    if (ret != 0) {
        fprintf(stderr, "FAILURE\n");
    }

    free(model);
    free(output);
    free(decrypted);

    mbedtls_entropy_free(&entropy);
    mbedtls_ctr_drbg_free(&ctr_drbg);
    mbedtls_gcm_free(&gcm);

    return ret;
}