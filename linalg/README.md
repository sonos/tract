# tract-linalg

linalg stands for "linear algebra". This is a misnamer. This crates contains
low-level, architecture dependant optimisations used by tract-core.

# Functions

* MatMatMul: Extended matrix*matrix product:
    * inspired by Gotoblass and BLIS micro kernel approach
    * extended for convolution friendly addressing (fused img2col)
    * fused output pipeline (min, max, and a few more simple, fast ops)
    * f32*f32 -> f32 (à la sgemm)
    * i8*i8 -> i32 accumulator -> i32 storage
    * i8*i8 -> i32 accumulator -> i8 (with channel zeropoint and scale, and re-quantization pipeline)
* f32 sigmoid and f32 tanh: at f32 precision, by a rationale function (no exponentiation)
* byte-to-byte lookup table

# Implementations

|                   |  generic fallback  |   armv6, vfp  |     armv7 neon    |    armv8 simd     |     x64 FMA
|-------------------|--------------------|---------------|-------------------|-------------------|-----------------
| MatMatMul f32     |                    |      4x4      |         8x4       |       8x8         |       16x6
| MatMatMul i8->i8  |                    |               |         8x4       |                   |        8x8
| MatMatMul i8->i32 |                    |               |                   |                   |        8x8
| sigmoid f32       |                    |               |         4n        |        4n         |
| tanh f32          |                    |               |         4n        |        4n         |
| byte lookup       |                    |               |                   |                   |
