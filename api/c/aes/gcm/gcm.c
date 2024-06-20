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
    unsigned char tag_encr[TAG_BYTES];
    unsigned char tag_decr[TAG_BYTES];
    size_t olen;
    int ret;
    
    if (!output || !decrypted) {
        ret = 1;
        fprintf(stderr, "Memory allocation failed\n");
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
    memset(tag_encr, 0, TAG_BYTES);
    memset(tag_decr, 0, TAG_BYTES);

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

    // Generate random bytes for the add_data (64 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, add_data, ADD_DATA_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract add_data - returned -0x%04x\n", -ret);
        goto exit;
    }
    mbedtls_printf("aad: ");
    for (int i = 0; i < ADD_DATA_BYTES; i++) {
        mbedtls_printf("%02x", add_data[i]);
    }
    mbedtls_printf("\n");

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
        fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher in encryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Start the GCM encryption process
    ret = mbedtls_gcm_starts(&gcm,                 // GCM context
                            MBEDTLS_GCM_ENCRYPT,   // mode
                            iv,                    // initialization vector
                            IV_BYTES);             // length of IV
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_starts failed to start the encryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Set additional authenticated data (AAD)
    ret = mbedtls_gcm_update_ad(&gcm,              // GCM context
                                add_data,          // additional data
                                ADD_DATA_BYTES);   // length of AAD
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_starts failed to set the AAD in the encryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    if (plain_len > 32) {
        size_t rest_len = plain_len - 32;

        // Encrypt the first 32 bytes
        ret = mbedtls_gcm_update(&gcm,      // GCM context
                                model,      // input data
                                32,         // length of first 32 bytes of input data
                                output,     // output of encryption process for the first 32 bytes
                                plain_len,  // length of input data
                                &olen);     // length of output data (expected 32)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to encrypt the first 32 bytes of input data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != 32) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in encryption process, expected 32 - returned -0x%04x\n", -ret);
            goto exit;
        }

        // Encrypt the rest of the data
        ret = mbedtls_gcm_update(&gcm,            // GCM context
                                model + 32,       // input data for the rest data
                                rest_len,         // length of the rest data
                                output + 32,      // output of encryption process for the rest data
                                plain_len - 32,   // length of the rest data
                                &olen);           // length of output data (expected rest_len)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to encrypt the rest of the data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != rest_len) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in encryption process, expected %ld - returned -0x%04x\n", rest_len, -ret);
            goto exit;
        }
    } else {
        // Encrypt the whole model
        ret = mbedtls_gcm_update(&gcm,       // GCM context
                                model,       // input data
                                plain_len,   // length of input data
                                output,      // output of encryption process
                                plain_len,   // length of input data
                                &olen);      // length of output data (expected plain_len)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to encrypt the whole data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != (size_t)(plain_len)) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in the encryption process, expected %ld - returned -0x%04x\n", plain_len, -ret);
            goto exit;
        }
    }

    // Finish the GCM encryption process and generate the tag in encryption process
    ret = mbedtls_gcm_finish(&gcm,           // GCM context
                            NULL,            // input data, here NULL
                            0,               // length of input data, here 0
                            &olen,           // length of output data, here olen
                            tag_encr,        // buffer for holding the tag
                            TAG_BYTES);      // length of the tag
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_finish failed to finish the encryption process and generate the tag - returned -0x%04x\n", -ret);
        goto exit;
    }

    mbedtls_printf("ciphertxt in hex: ");
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
        fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher for the decryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Start the GCM decryption process
    ret = mbedtls_gcm_starts(&gcm,                 // GCM context
                            MBEDTLS_GCM_DECRYPT,   // mode
                            iv,                    // initialization vector
                            IV_BYTES);             // length of IV
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_starts failed to start the decryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    // Set additional authenticated data (AAD)
    ret = mbedtls_gcm_update_ad(&gcm,              // GCM context
                                add_data,          // additional data
                                ADD_DATA_BYTES);   // length of AAD
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_starts failed to set the AAD in the decryption process - returned -0x%04x\n", -ret);
        goto exit;
    }

    if (plain_len > 32) {
        size_t rest_len = plain_len - 32;

        // Decrypt the first 32 bytes
        ret = mbedtls_gcm_update(&gcm,       // GCM context
                                output,      // output of encryption data
                                32,          // length of first 32 bytes of output data
                                decrypted,   // output of decryption process for the first 32 bytes
                                plain_len,   // length of output data
                                &olen);      // length of decrypted data (expected 32)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to decrypt the first 32 bytes of output data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != 32) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in decryption process, expected 32 - returned -0x%04x\n", -ret);
            goto exit;
        }

        // Decrypt the rest of the data
        ret = mbedtls_gcm_update(&gcm,             // GCM context
                                output + 32,       // output data for the rest data
                                rest_len,          // length of the rest data
                                decrypted + 32,    // output of decryption process for the rest data
                                plain_len - 32,    // length of the rest data
                                &olen);            // length of decrypted data (expected rest_len)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to decrypt the rest of the output data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != rest_len) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in decryption process, expected %ld - returned -0x%04x\n", rest_len, -ret);
            goto exit;
        }
    } else {
        // Decrypt the whole model
        ret = mbedtls_gcm_update(&gcm,        // GCM context
                                output,       // output data
                                plain_len,    // length of output data
                                decrypted,    // output of decryption process
                                plain_len,    // length of output data
                                &olen);       // length of decrypted data (expected plain_len)
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_update failed to decrypt the whole data - returned -0x%04x\n", -ret);
            goto exit;
        }
        if (olen != (size_t)(plain_len)) {
            fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in the decryption process, expected %ld - returned -0x%04x\n", plain_len, -ret);
            goto exit;
        }
    }

    // Finish the GCM decryption process and generate the tag in decryption process
    ret = mbedtls_gcm_finish(&gcm,           // GCM context
                            NULL,            // input data, here NULL
                            0,               // length of input data, here 0
                            &olen,           // length of output data, here olen
                            tag_decr,        // buffer for holding the tag
                            TAG_BYTES);      // length of the tag
    if (ret != 0) {
        fprintf(stderr, "mbedtls_gcm_finish failed to finish the decryption process and generate the tag - returned -0x%04x\n", -ret);
        goto exit;
    }

    mbedtls_printf("plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", decrypted[i]);
    }
    mbedtls_printf("\n");

    // Uncomment this line to corrupt buffers so that GCM will fail to authenticate on decryption
    // memset(add_data, 0, ADD_DATA_BYTES);
    // memset(tag, 0, TAG_BYTES);
    // memset(iv, 0, IV_BYTES);
    // memset(key, 0, KEY_BYTES); // key must be changes before calling mbedtls_gcm_setkey

    if (memcmp(decrypted, model, plain_len) != 0 || memcmp(tag_encr, tag_decr, TAG_BYTES) != 0) {
        ret = 1;
        goto exit;
    } else {
        mbedtls_printf("SUCCESS\n");
    }

exit:
    if (ret != 0) {
        fprintf(stderr, "FAILURE\n");
    }

    free(model);
    free(output);
    free(decrypted);
    mbedtls_gcm_free(&gcm);

    return ret;
}