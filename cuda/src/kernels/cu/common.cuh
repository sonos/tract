#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

#define CUDA_CC_TURING 750
#define CUDA_CC_AMPERE 800

#define FLT_MAX 3.40282347e+38F

#define MAX_THREADS 1024
#define WARP_SIZE 32

#define QK8_1 32
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QK4_0 32
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QK8_0 32
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

typedef struct {
  half d;                // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

typedef struct {
  half2 ds;
  int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(half) + QK8_1,
              "wrong q8_1 block size/padding");

struct block_q8_1_mmq {
  // The y float data is converted to a data layout that can simply be copied to
  // shared memory as a contiguous block. The y float data is first grouped as
  // blocks of 128 values. These blocks are then treated as individual data
  // values and transposed.
  //
  // To avoid shared memory bank conflicts each block is padded with 16 bytes.
  // This padding is also used to store block scales/partial sums.
  // The scales multiplied with the quantized data are equal to the unquantized
  // values. The partial sums are obtained by summing up a subgroup of the
  // contained values (prior to quantization)
  //     and are only needed for performance reasons.
  half2 ds4[4]; // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored
                // as d0,s0,d1,s1,d2,s2,d3,s3
  int8_t qs[4 * QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4 * QK8_1 + 4 * sizeof(half2),
              "Unexpected block_q8_1_mmq size");

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_sum(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_max(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmax(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_min(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fminf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_min(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmin(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_prod(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x *= __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_prod(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmul(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
  return __all_sync(0xffffffff, x);
}

namespace cuda_mma {

template <int I_, int J_, typename T> struct tile {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / WARP_SIZE;
  T x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && (J == 4 || J == 8)) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return (l / 2) * 8 + threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 16) {
      return ((l / 2) % 2) * 8 + threadIdx.x / 4;
    } else {
      static_assert(I == -1 && J == -1,
                    "template specialization not implemented");
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x % 4;
    } else if constexpr (I == 8 && J == 8) {
      return 4 * l + threadIdx.x % 4;
    } else if constexpr (I == 16 && J == 8) {
      return 2 * (threadIdx.x % 4) + l % 2;
    } else if constexpr (I == 16 && J == 16) {
      return 8 * (l / 4) + 2 * (threadIdx.x % 4) + l % 2;
    } else {
      static_assert(I == -1 && J == -1,
                    "template specialization not implemented");
    }
  }
};

template <int I_, int J_> struct tile<I_, J_, half2> {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / WARP_SIZE;
  half2 x[ne] = {{0.0f, 0.0f}};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 4) {
      return l * 8 + threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return (l % 2) * 8 + threadIdx.x / 4;
    } else {
      static_assert(I == -1 && J == -1,
                    "template specialization not implemented");
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 8) {
      return l * 4 + threadIdx.x % 4;
    } else if constexpr (I == 16 && J == 4) {
      return threadIdx.x % 4;
    } else if constexpr (I == 16 && J == 8) {
      return (l / 2) * 4 + threadIdx.x % 4;
    } else {
      static_assert(I == -1 && J == -1,
                    "template specialization not implemented");
    }
  }
};

template <int I, int J>
static __device__ __forceinline__ tile<I, J / 2, half2>
get_half2(const tile<I, J, float> &tile_float) {
  tile<I, J / 2, half2> ret;
#pragma unroll
  for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
    ret.x[l0 / 2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
  }
  return ret;
}

template <int I, int J, typename T>
static __device__ __forceinline__ void
load_generic(tile<I, J, T> &t, const T *__restrict__ xs0, const int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

template <typename T>
static __device__ __forceinline__ void
load_ldmatrix(tile<8, 8, T> &t, const T *__restrict__ xs0, const int stride) {
  int *xi = (int *)t.x;
  const int *xs = (const int *)xs0 + (threadIdx.x % t.I) * stride +
                  ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
               : "=r"(xi[0]), "=r"(xi[1])
               : "l"(xs));
}

template <typename T>
static __device__ __forceinline__ void
load_ldmatrix(tile<16, 4, T> &t, const T *__restrict__ xs0, const int stride) {
  int *xi = (int *)t.x;
  const int *xs = (const int *)xs0 + (threadIdx.x % t.I) * stride;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
               : "=r"(xi[0]), "=r"(xi[1])
               : "l"(xs));
}

template <typename T>
static __device__ __forceinline__ void
load_ldmatrix(tile<16, 8, T> &t, const T *__restrict__ xs0, const int stride) {
  int *xi = (int *)t.x;
  const int *xs = (const int *)xs0 + (threadIdx.x % t.I) * stride +
                  (threadIdx.x / t.I) * (t.J / 2);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
}

template <typename T>
static __device__ __forceinline__ void
load_ldmatrix_trans(tile<16, 8, T> &t, const T *__restrict__ xs0,
                    const int stride) {
  int *xi = (int *)t.x;
  const int *xs = (const int *)xs0 + (threadIdx.x % t.I) * stride +
                  (threadIdx.x / t.I) * (t.J / 2);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
               : "l"(xs));
}

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE

    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[2]), "r"(B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[3]), "r"(B.x[1]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 4, half2> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, float> & A, const tile<8, 8, float> & B) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 16, float> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE
    }
} // namespace cuda_mma

#if CUDART_VERSION >= 11080

static __device__ __forceinline__ int cuda_movmatrix(const int x) {
    int ret = 0;

    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(ret) : "r"(x));
    return ret;
}

#else

static __device__ __forceinline__ int cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11080

static __device__ __forceinline__ half2 cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = cuda_movmatrix(*((const int *) &x));
    return ret;
}


// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};