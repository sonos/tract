#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

#define CAT2(a,b) a##b
#define CAT(a,b)  CAT2(a,b)
#define CAT3(a,b,c) CAT(CAT(a,b),c)
#define CAT4(a,b,c,d) CAT(CAT3(a,b,c),d)
#define CAT5(a,b,c,d,e) CAT(CAT(CAT3(a,b,c),d),e)

#define CUDA_CC_TURING 750
#define CUDA_CC_AMPERE 800

#define FLT_MAX 3.40282347e+38F

#define MAX_THREADS 1024
#define WARP_SIZE 32

// Dummy type for aligning extern __shared__ declarations.
// Always use alignment_dummy instead of raw types for extern __shared__
// to ensure 16-byte alignment for cp.async / ldmatrix / mma operations.
struct alignment_dummy {
    alignas(16) int dummy;
};

// Type-safe shared memory allocator for dynamic shared memory.
// Inspired by ThunderKittens (github.com/HazyResearch/ThunderKittens)
// include/common/util.cuh shared_allocator.
//
// Usage:
//   extern __shared__ alignment_dummy __shm[];
//   shared_allocator al((int*)&__shm[0]);
//   auto& my_struct  = al.allocate<my_type>();        // single instance
//   auto& my_array   = al.allocate<float, 256>();     // float[256]
//   auto& my_matrix  = al.allocate<half, 8, 32>();    // half[8][32]
//
// Each allocate<>() call advances an internal pointer with proper alignment,
// eliminating manual offset arithmetic and alignment bugs.
template<int default_alignment = 16>
struct shared_allocator {
    int *ptr;

private:
    // Recursive helper to generate N-dimensional array type from trailing
    // size_t dimensions: allocate<T, 8, 32>() -> T&[8][32]
    template<typename A, size_t... dims>
    struct variadic_array;
    template<typename A, size_t first_dim, size_t... rest_dims>
    struct variadic_array<A, first_dim, rest_dims...> {
        using type = typename variadic_array<A, rest_dims...>::type[first_dim];
    };
    template<typename A>
    struct variadic_array<A> {
        using type = A;
    };
    template<typename A, size_t... dims>
    using variadic_array_t = typename variadic_array<A, dims...>::type;

    template<int alignment>
    __device__ __forceinline__ void align_ptr() {
        // No-op: alignment_dummy has alignas(16), guaranteeing the
        // shared memory block is 16-byte aligned at entry. All
        // allocate() sizes in this codebase are multiples of 16
        // bytes, so ptr stays aligned after each call.
    }

public:
    __device__ shared_allocator(int *_ptr) : ptr(_ptr) {}

    // Allocate a single instance or N-dimensional array of type A.
    // Returns a reference to the allocated object.
    template<typename A, size_t... dims>
    __device__ __forceinline__ variadic_array_t<A, dims...> &allocate() {
        align_ptr<default_alignment>();
        using at = variadic_array_t<A, dims...>;
        at *p = reinterpret_cast<at *>(ptr);
        // Ceiling division so sub-4-byte types (e.g. half) are handled correctly.
        ptr += (sizeof(at) + sizeof(int) - 1) / sizeof(int);
        return *p;
    }

    // Allocate with a custom alignment override.
    template<int alignment, typename A, size_t... dims>
    __device__ __forceinline__ variadic_array_t<A, dims...> &allocate() {
        align_ptr<alignment>();
        using at = variadic_array_t<A, dims...>;
        at *p = reinterpret_cast<at *>(ptr);
        ptr += (sizeof(at) + sizeof(int) - 1) / sizeof(int);
        return *p;
    }
};

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

// ============================================================================
// WGMMA intrinsics (SM90+ / Hopper, SM100+ / Blackwell)
//
// WGMMA (Warp General Matrix Multiply) operates on 4-warpgroup tiles and can
// compute far more MACs per instruction than traditional mma.sync (e.g.
// m64n32k16 = 32768 MACs vs 2048 for m16n8k16).  This section provides thin
// wrappers around the raw PTX, modelled on ThunderKittens' educational Level-05
// examples and the matrix-descriptor encoding from TK's `prototype/mma/`.
//
// On SM90 the accumulator for m64n32k16.f32.f16.f16 holds 16 f32 per thread
// (8 × float2) across 4 warps (128 threads).  The row-major layout is:
//   thread tid → row = tid/2, col_base = (tid%2)*16
//   D[i].x → (row, col_base + i*2)
//   D[i].y → (row, col_base + i*2 + 1)
// ============================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

/// cp.async 16-byte copy from global memory to shared memory (cache-at-shared).
static __device__ __forceinline__ void cp_async_ca_16B(uint32_t dst, const void *src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(src) : "memory");
}

/// Predicated cp.async 16-byte copy with zero-fill on miss.
static __device__ __forceinline__ void cp_async_ca_16B_pred(uint32_t dst, const void *src, bool pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 z;\n\t"
        "mov.b32 z, 0;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p   cp.async.ca.shared.global [%0], [%1], 16;\n\t"
        "@!p  st.shared.v4.b32 [%0], {z, z, z, z};\n\t"
        "}\n\t"
        :: "r"(dst), "l"(src), "r"((int)pred) : "memory");
}

/// Commit all pending cp.async copies.
static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

/// Wait until all pending cp.async copies have completed.
static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

/// Predicated cp.async 4-byte copy with zero-fill on miss.
static __device__ __forceinline__ void cp_async_ca_4B_pred(uint32_t dst, const void *src, bool pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 z;\n\t"
        "mov.b32 z, 0;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p   cp.async.ca.shared.global [%0], [%1], 4;\n\t"
        "@!p  st.shared.b32 [%0], z;\n\t"
        "}\n\t"
        :: "r"(dst), "l"(src), "r"((int)pred) : "memory");
}

namespace cuda_wgmma {

/// Fence the WGMMA pipeline so subsequent cp.async loads are ordered
/// before the next mma_async issue.
static __device__ __forceinline__ void fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

/// Commit the current group of issued WGMMA operations to the pipeline.
static __device__ __forceinline__ void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

/// Wait until at least `n` committed WGMMA groups have completed.
static __device__ __forceinline__ void wait_group(int n) {
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"r"(n) : "memory");
}

/// Construct a 64-bit WGMMA shared-memory descriptor for matrix B
/// (the K×N operand).  `smem_addr` is a raw shared-memory address (e.g.
/// from `__cvta_generic_to_shared`); it must be 16-byte aligned.
///
/// Layout fields (SM90, descriptor is 64-bit):
///   [13:0]   = addr bits [17:4]
///   [29:16]  = (leading_dim / 16) & 0x3FFF
///   [63:62]  = swizzle_mode (0=none, 1=16B, 2=32B, 3=64B)
///
/// On SM100+ (Blackwell, __CUDA_ARCH__ >= 1000) bit-46 is set as well.
static __device__ __forceinline__ uint64_t
make_b_desc(uint32_t smem_addr, uint32_t leading_dim_bytes, uint32_t swizzle_mode) {
    uint64_t desc = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    desc |= ((((uint64_t)leading_dim_bytes >> 4) & 0x3FFFULL) << 16);
    desc |= ((uint64_t)(swizzle_mode & 0x3) << 62);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    desc |= (1ULL << 46); // SM100+ requires this bit
#endif
    return desc;
}

/// WGMMA MMA: m64n32k16, fp16 (A,B) → fp32 accumulation.
///
/// Tile shapes (per 4-warpgroup / 128-thread call):
///   A: 64×16 fp16, row-major in shared memory
///   B: 16×32 fp16, in shared memory (descriptor-encoded layout)
///   D: 64×32 fp32  (16 f32 = 8 float2 per thread, accumulated)
///
/// Parameters:
///   D           – 8 float2 accumulators (in/out)
///   A_smem_addr – raw shared-memory address of A tile
///   A_stride    – byte stride between rows of A
///   B_desc      – 64-bit B descriptor from make_b_desc()
///   pred_en     – non-zero enables the WGMMA (predicate control)
///   neg_b       – 0 = no negation, 1 = negate B
///   trans_b     – 0 = row-major B, 1 = col-major B
static __device__ __forceinline__ void
mma_m64n32k16_fp16_fp32(float2 *D, uint32_t A_smem_addr, uint32_t A_stride,
                        uint64_t B_desc, int pred_en, int neg_b, int trans_b) {
    uint32_t a_desc[4] = {
        A_smem_addr,
        0u,
        A_stride,
        0u, // reserved
    };
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %21, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, "
        "%20, "
        "p, 1, %23, %22;\n\t"
        "}\n\t"
        : "+f"(D[0].x), "+f"(D[0].y), "+f"(D[1].x), "+f"(D[1].y),
          "+f"(D[2].x), "+f"(D[2].y), "+f"(D[3].x), "+f"(D[3].y),
          "+f"(D[4].x), "+f"(D[4].y), "+f"(D[5].x), "+f"(D[5].y),
          "+f"(D[6].x), "+f"(D[6].y), "+f"(D[7].x), "+f"(D[7].y)
        : "r"(a_desc[0]), "r"(a_desc[1]), "r"(a_desc[2]), "r"(a_desc[3]),
          "l"(B_desc), "r"(pred_en), "r"(trans_b), "r"(neg_b)
        : "memory");
}

} // namespace cuda_wgmma

#endif // __CUDA_ARCH__ >= 900

// ============================================================================
// TMA (Tensor Memory Accelerator) helpers for SM90+ (Hopper, Blackwell, etc.)
//
// `cp.async.bulk.tensor` uses a CUtensorMap descriptor to describe the
// global→shared layout, letting the hardware manage swizzling, coalescing
// and border handling.  Each bulk copy also carries an mbarrier arrival,
// so we supplement with explicit mbarrier init / arrive.expect_tx / wait
// calls for group-level synchronization.
// ============================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// 128-byte opaque CUDA tensor-map descriptor (matches CUtensorMap_st).
struct cuda_tensor_map {
    uint64_t opaque[16];
};
static_assert(sizeof(cuda_tensor_map) == 128, "CUtensorMap must be 128 bytes");

// 8-byte mbarrier in shared memory (matches kittens::semaphore).
struct cuda_mbar {
    uint64_t value;
};

// Issue a TMA bulk-tensor 4D shared load.  coord c/r/d/b select the tile.
static __device__ __forceinline__ void
cp_async_bulk_tensor_4d(uint32_t dst_smem, const cuda_tensor_map *tma_desc,
                        uint32_t c, uint32_t r, uint32_t d, uint32_t b,
                        uint32_t mbar_smem) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
        ::"r"(dst_smem), "l"(tma_desc), "r"(c), "r"(r), "d"(d), "b"(b),
        "r"(mbar_smem)
        : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

static __device__ __forceinline__ void cp_async_bulk_wait_group(int N) {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

static __device__ __forceinline__ void mbarrier_init(cuda_mbar &bar, uint32_t count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr), "r"(count));
}

static __device__ __forceinline__ void mbarrier_arrive_expect_tx(cuda_mbar &bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(bar_ptr), "r"(bytes)
                 : "memory");
}

static __device__ __forceinline__ void mbarrier_wait_parity(cuda_mbar &bar, uint32_t parity) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
        " @!p bra -;\n\t"
        "}"
        ::"r"(bar_ptr), "r"(parity)
        : "memory");
}

static __device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

#endif // __CUDA_ARCH__ >= 900

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
