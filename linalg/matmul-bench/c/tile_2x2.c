
#include <stddef.h>

void c_tile_2x2(size_t m, size_t k, size_t n, float *a, float *b, float *c) {
    for(size_t row = 0 ; row < m / 2 ; row++) {
        for(size_t col = 0 ; col < n / 2 ; col++) {
            float  sum00 = 0.0;
            float  sum01 = 0.0;
            float  sum10 = 0.0;
            float  sum11 = 0.0;
            for(size_t i = 0; i < k ; i++) {
                float a0 = a[2 * row * k + i];
                float a1 = a[(2 * row + 1) * k + i];
                float b0 = b[i * n + 2 * col];
                float b1 = b[i * n + 2 * col + 1];
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
            }
            c[(2 * row + 0) * n + 2 * col] = sum00;
            c[(2 * row + 0) * n + 2 * col + 1] = sum01;
            c[(2 * row + 1) * n + 2 * col] = sum10;
            c[(2 * row + 1) * n + 2 * col + 1] = sum11;
        }
    }
}
