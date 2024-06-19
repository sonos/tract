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
#define IV_BYTES 12

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

    unsigned char tag_buf[16];
    unsigned char *output = malloc(plain_len + 1);
    unsigned char *decrypted = malloc(plain_len + 1);

    unsigned char key[KEY_BYTES];
    unsigned char iv[IV_BYTES];
    
    mbedtls_entropy_init(&entropy);
    mbedtls_ctr_drbg_init(&ctr_drbg);
    mbedtls_gcm_init(&gcm);

    char *pers = "aes generate key";
    int ret;
    size_t olen;

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

    if (!output || !decrypted) {
        fprintf(stderr, "Memory allocation failed\n");
        goto exit;
    }
    
    mbedtls_printf("AES-GCM-%3d (%s):\n", 256, "enc");

    mbedtls_printf("plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", model[i]);
    }
    mbedtls_printf("\n");

    ret = mbedtls_gcm_setkey(&gcm,
                            MBEDTLS_CIPHER_ID_AES,
                            key,
                            256);
    if (ret != 0) {
        goto exit;
    }


    ret = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT,
                                    plain_len,
                                    iv,
                                    IV_BYTES,
                                    NULL,
                                    0,
                                    model,
                                    output, 16, tag_buf);

    if (ret != 0) {
        goto exit;
    }

    mbedtls_printf("output in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", output[i]);
    }
    mbedtls_printf("\n");

    mbedtls_gcm_free(&gcm);

    mbedtls_gcm_init(&gcm);

    mbedtls_printf("  AES-GCM-%3d (%s): ",
                        256, "dec");

    ret = mbedtls_gcm_setkey(&gcm,
                            MBEDTLS_CIPHER_ID_AES,
                            key,
                            256);
    if (ret != 0) {
        goto exit;
    }

    ret = mbedtls_gcm_crypt_and_tag(&gcm, 
                                    MBEDTLS_GCM_DECRYPT,
                                    plain_len,
                                    iv,
                                    IV_BYTES,
                                    NULL,
                                    0,
                                    output,
                                    decrypted,
                                    16,
                                    tag_buf);

    if (ret != 0) {
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
        mbedtls_printf("passed without splitting\n");
    }

    mbedtls_gcm_free(&gcm);

    memset(output, 0, plain_len);
    if (output) {
        mbedtls_printf("output: %s\n", output);
    }
    memset(decrypted, 0, plain_len);

    mbedtls_gcm_init(&gcm);

    mbedtls_printf("AES-GCM-%3d split (%s): ",
                        256, "enc");

    ret = mbedtls_gcm_setkey(&gcm,
                            MBEDTLS_CIPHER_ID_AES,
                            key,
                            256);
    if (ret != 0) {
        goto exit;
    }

    mbedtls_printf("plaintext in hex: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", model[i]);
    }
    mbedtls_printf("\n");

    ret = mbedtls_gcm_starts(&gcm,
                            MBEDTLS_GCM_ENCRYPT,
                            iv,
                            IV_BYTES);
    if (ret != 0) {
        goto exit;
    }

    ret = mbedtls_gcm_update_ad(&gcm,
                                NULL,
                                0);
    if (ret != 0) {
        goto exit;
    }

    if (plain_len > 32) {
        size_t rest_len = plain_len - 32;
        ret = mbedtls_gcm_update(&gcm,
                                model,
                                32,
                                output,
                                plain_len,
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != 32) {
            goto exit;
        }

        ret = mbedtls_gcm_update(&gcm,
                                model + 32,
                                rest_len,
                                output + 32, 
                                plain_len - 32, 
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != rest_len) {
            goto exit;
        }
    } else {
        ret = mbedtls_gcm_update(&gcm,
                                model,
                                plain_len,
                                output, 
                                plain_len, 
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != (size_t)(plain_len)) {
            goto exit;
        }
    }

    ret = mbedtls_gcm_finish(&gcm, 
                            NULL,
                            0,
                            &olen,
                            tag_buf,
                            16);
    if (ret != 0) {
        goto exit;
    }

    mbedtls_gcm_free(&gcm);

    mbedtls_printf("output: ");
    for (int i = 0; i < plain_len; i++) {
        mbedtls_printf("%02x", output[i]);
    }
    mbedtls_printf("\n");

    mbedtls_gcm_init(&gcm);

    mbedtls_printf("  AES-GCM-%3d split (%s): ",
                        256, "dec");

    ret = mbedtls_gcm_setkey(&gcm,
                            MBEDTLS_CIPHER_ID_AES,
                            key,
                            256);
    if (ret != 0) {
        goto exit;
    }

    ret = mbedtls_gcm_starts(&gcm,
                            MBEDTLS_GCM_DECRYPT,
                            iv,
                            IV_BYTES);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_gcm_update_ad(&gcm,
                                NULL,
                                0);
    if (ret != 0) {
        goto exit;
    }

    if (plain_len > 32) {
        size_t rest_len = plain_len - 32;
        ret = mbedtls_gcm_update(&gcm,
                                output,
                                32,
                                decrypted,
                                plain_len,
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != 32) {
            goto exit;
        }

        ret = mbedtls_gcm_update(&gcm,
                                output + 32,
                                rest_len,
                                decrypted + 32,
                                plain_len - 32, 
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != rest_len) {
            goto exit;
        }
    } else {
        ret = mbedtls_gcm_update(&gcm,
                                output,
                                plain_len,
                                decrypted,
                                plain_len,
                                &olen);
        if (ret != 0) {
            goto exit;
        }
        if (olen != (size_t)(plain_len)) {
            goto exit;
        }
    }

    ret = mbedtls_gcm_finish(&gcm, 
                            NULL,
                            0,
                            &olen,
                            tag_buf, 
                            16);
    if (ret != 0) {
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
        mbedtls_printf("passed with splitting\n");
    }

    mbedtls_gcm_free(&gcm);

    mbedtls_printf("\n");

    ret = 0;

exit:
    if (ret != 0) {
        mbedtls_printf("failed\n");
    }

    free(model);
    free(output);
    free(decrypted);
    mbedtls_gcm_free(&gcm);

    return ret;
}