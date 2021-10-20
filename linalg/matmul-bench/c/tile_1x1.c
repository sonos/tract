
#include <stddef.h>

void c_tile_1x1(size_t m, size_t k, size_t n, float *a, float *b, float *c) {
    for(size_t row = 0 ; row < m ; row++) {
        for(size_t col = 0 ; col < n ; col++) {
            float  sum00 = 0.0;
            for(size_t i = 0; i < k ; i++) {
                float a0 = a[row * k + i];
                float b0 = b[i * n + col];
                sum00 += a0 * b0;
            }
            c[row * n + col] = sum00;
        }
    }
}
