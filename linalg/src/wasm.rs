/// Wasm SIMD implementation of `MatMatMulKer<f32>`
///
/// To run test, you need to install `wasmtime`
/// and export the following environment variables:
/// ```
/// > export RUSTFLAGS='-C target-feature=+simd128'
/// > export CARGO_TARGET_WASM32_WASI_RUNNER=wasmtime
/// > cargo test --target=wasm32-wasi
/// ```
use crate::mmm::FusedKerSpec;
use crate::mmm::ImplementationQuality;
use crate::{Ops, Scaler};

#[cfg(target_feature = "relaxed-simd")]
use crate::frame::element_wise::ElementWiseKer;

// f32x4 mul+add → relaxed FMA when the build has +relaxed-simd, else explicit
// mul+add. Lets the MMM kernels emit f32x4.relaxed_madd without duplicating
// kernel source. Per PR #2199: LLVM does not auto-emit relaxed_madd from
// f32x4_add(f32x4_mul(...)) even with +relaxed-simd — hand emission is needed.
//
// Caller must have `use std::arch::wasm32::*;` in scope (every kernel does).
// Args are passed (acc, a, b); evaluation order differs between the two arms
// (acc-first in baseline, acc-last in FMA), so callers must pass simple
// variable names rather than expressions with side effects.
#[cfg(target_feature = "relaxed-simd")]
macro_rules! madd_f32x4 {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_relaxed_madd($a, $b, $acc)
    };
}

#[cfg(not(target_feature = "relaxed-simd"))]
macro_rules! madd_f32x4 {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_add($acc, f32x4_mul($a, $b))
    };
}

// Always-non-fused madd. Used by kernels with ≤4 SIMD accumulators per K-step
// (wasm_f32_4x1, _8x1, _16x1, _4x4), where the destructive `fmla.4s`
// emitted by +relaxed-simd creates a 4-cycle accumulator RAW recurrence
// that throttles throughput to 1 FMA/cycle even though Apple-class ARM64
// pipes can do 4. The separate `fmul.4s; fadd.4s` form gives each multiply
// a fresh destination register, letting the OoO renamer overlap the next
// iteration's multiply with the in-flight add. Measured: under
// +simd128,+relaxed-simd these kernels are 19-28% slower than under
// +simd128 when using the fused form on Apple M1 — both wasmtime
// (Cranelift) and Node 20 (V8) reproduce identically. Wider kernels
// (wasm_f32_32x1 with 8 accs, wasm_f32_8x8 with 16) keep the fused form
// because their pipe is saturated and FMA's 1-instruction-per-madd wins.
//
// Cross-check: XNNPACK only ships wasmrelaxedsimd-fma GEMM kernels at
// NR=8 (i.e. ≥8 accumulator-equivalents), independently arriving at the
// same threshold without writing it down.
macro_rules! madd_f32x4_nofma {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_add($acc, f32x4_mul($a, $b))
    };
}

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(wasm_f32_4x4.mmm());
    ops.mmm_impls.push(wasm_f32_4x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x1.mmm());
    ops.mmm_impls.push(wasm_f32_16x1.mmm());
    ops.mmm_impls.push(wasm_f32_32x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x8.mmm());
    // int8 -> i32 matmul: SIMD kernel (was generic scalar). ManuallyOptimized so
    // strategize's retain() keeps it over generic_i32_4x4 for i8 packing.
    ops.mmm_impls.push(wasm_i32_4x4.mmm());
    ops.qmmm_i32 = Box::new(|_, _, _| wasm_i32_4x4.mmm());
    // Selection paths. Both rely on kernel_selection::strategize honouring
    // the mmm_f32 / mmv_f32 callback, which it only does when the callback's
    // kernel is tagged ManuallyOptimized. Otherwise strategize falls through
    // to list_impls, whose retain() keeps only the top ImplementationQuality
    // and drops every TargetOptimized kernel.
    //   - N>1 (GEMM): mmm_f32 returns 8x8, so 8x8 MUST be ManuallyOptimized.
    //     If it were TargetOptimized it would be dropped by retain(), and the
    //     N>1 branch's max(nr*mr) over the surviving (ManuallyOptimized) GEMV
    //     kernels would pick wasm_f32_32x1 — a matrix×vector kernel — for
    //     every GEMM.
    //   - N=1 (GEMV): mmv_f32 routes by M-band to the kernel whose MR fits.
    //     The four GEMV kernels are ManuallyOptimized for the same reason —
    //     without the tag strategize discards the callback and picks
    //     max(mr)=32x1 for every M, leaving up to ~37% on the table for
    //     small-M GEMV.
    ops.mmm_f32 = Box::new(|_m, _k, _n| wasm_f32_8x8.mmm());
    // Bands derived from microbench_dispatch_gemv. At each band edge, using
    // the next-larger kernel beats halving outer iterations of the smaller
    // one (1 outer with ILP-absorbed padding > 2 outer with kernel preamble
    // doubled). M=4/8/16 are exact tile fits at the lower edges; M=17/9/5
    // are the first values where the next-larger kernel wins.
    ops.mmv_f32 = Box::new(|m, _k| match m.unwrap_or(0) {
        0..=4 => wasm_f32_4x1.mmm(),
        5..=8 => wasm_f32_8x1.mmm(),
        9..=16 => wasm_f32_16x1.mmm(),
        _ => wasm_f32_32x1.mmm(),
    });
    // Relaxed-SIMD activation kernels (FMA path). Only installed when the
    // build has `+relaxed-simd`; otherwise the slots stay at the generic
    // scalar polynomial.
    #[cfg(target_feature = "relaxed-simd")]
    {
        ops.sigmoid_f32 = Box::new(|| WasmSigmoid4Relaxed::ew());
        ops.tanh_f32 = Box::new(|| WasmTanh4Relaxed::ew());
    }
}

unsafe fn kernel_f32_4x4(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // Each of these variables stores a row of the matrix,
        // consisting of four packed `f32` numbers.
        let mut ab0 = f32x4_splat(0.0);
        let mut ab1 = f32x4_splat(0.0);
        let mut ab2 = f32x4_splat(0.0);
        let mut ab3 = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    let a = f32x4_splat(0.0);
                    ab0 = a;
                    ab1 = a;
                    ab2 = a;
                    ab3 = a;
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    let rows = rows as *const v128;
                    ab0 = *rows;
                    ab1 = *rows.add(1);
                    ab2 = *rows.add(2);
                    ab3 = *rows.add(3);
                }
                FusedKerSpec::ScalarMin(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_min(a, ab0);
                    ab1 = f32x4_min(a, ab1);
                    ab2 = f32x4_min(a, ab2);
                    ab3 = f32x4_min(a, ab3);
                }
                FusedKerSpec::ScalarMax(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_max(a, ab0);
                    ab1 = f32x4_max(a, ab1);
                    ab2 = f32x4_max(a, ab2);
                    ab3 = f32x4_max(a, ab3);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_add(a, ab0);
                    ab1 = f32x4_add(a, ab1);
                    ab2 = f32x4_add(a, ab2);
                    ab3 = f32x4_add(a, ab3);
                }
                FusedKerSpec::ScalarMul(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_mul(a, ab0);
                    ab1 = f32x4_mul(a, ab1);
                    ab2 = f32x4_mul(a, ab2);
                    ab3 = f32x4_mul(a, ab3);
                }
                FusedKerSpec::ScalarSub(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_sub(a, ab0);
                    ab1 = f32x4_sub(a, ab1);
                    ab2 = f32x4_sub(a, ab2);
                    ab3 = f32x4_sub(a, ab3);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let a = f32x4_splat(a);
                    ab0 = f32x4_sub(ab0, a);
                    ab1 = f32x4_sub(ab1, a);
                    ab2 = f32x4_sub(ab2, a);
                    ab3 = f32x4_sub(ab3, a);
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let a = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);

                    let mask0 = f32x4_gt(ab0, zero);
                    ab0 = v128_bitselect(ab0, f32x4_mul(a, ab0), mask0);

                    let mask1 = f32x4_gt(ab1, zero);
                    ab1 = v128_bitselect(ab1, f32x4_mul(a, ab1), mask1);

                    let mask2 = f32x4_gt(ab2, zero);
                    ab2 = v128_bitselect(ab2, f32x4_mul(a, ab2), mask2);

                    let mask3 = f32x4_gt(ab3, zero);
                    ab3 = v128_bitselect(ab3, f32x4_mul(a, ab3), mask3);
                }
                FusedKerSpec::PerRowMin(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_min(f32x4_splat(row[0]), ab0);
                    ab1 = f32x4_min(f32x4_splat(row[1]), ab1);
                    ab2 = f32x4_min(f32x4_splat(row[2]), ab2);
                    ab3 = f32x4_min(f32x4_splat(row[3]), ab3);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_max(f32x4_splat(row[0]), ab0);
                    ab1 = f32x4_max(f32x4_splat(row[1]), ab1);
                    ab2 = f32x4_max(f32x4_splat(row[2]), ab2);
                    ab3 = f32x4_max(f32x4_splat(row[3]), ab3);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_add(f32x4_splat(row[0]), ab0);
                    ab1 = f32x4_add(f32x4_splat(row[1]), ab1);
                    ab2 = f32x4_add(f32x4_splat(row[2]), ab2);
                    ab3 = f32x4_add(f32x4_splat(row[3]), ab3);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_mul(f32x4_splat(row[0]), ab0);
                    ab1 = f32x4_mul(f32x4_splat(row[1]), ab1);
                    ab2 = f32x4_mul(f32x4_splat(row[2]), ab2);
                    ab3 = f32x4_mul(f32x4_splat(row[3]), ab3);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_sub(f32x4_splat(row[0]), ab0);
                    ab1 = f32x4_sub(f32x4_splat(row[1]), ab1);
                    ab2 = f32x4_sub(f32x4_splat(row[2]), ab2);
                    ab3 = f32x4_sub(f32x4_splat(row[3]), ab3);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let row = std::slice::from_raw_parts(row, 4);
                    ab0 = f32x4_sub(ab0, f32x4_splat(row[0]));
                    ab1 = f32x4_sub(ab1, f32x4_splat(row[1]));
                    ab2 = f32x4_sub(ab2, f32x4_splat(row[2]));
                    ab3 = f32x4_sub(ab3, f32x4_splat(row[3]));
                }
                FusedKerSpec::PerColMin(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_min(cols, ab0);
                    ab1 = f32x4_min(cols, ab1);
                    ab2 = f32x4_min(cols, ab2);
                    ab3 = f32x4_min(cols, ab3);
                }
                FusedKerSpec::PerColMax(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_max(cols, ab0);
                    ab1 = f32x4_max(cols, ab1);
                    ab2 = f32x4_max(cols, ab2);
                    ab3 = f32x4_max(cols, ab3);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_add(cols, ab0);
                    ab1 = f32x4_add(cols, ab1);
                    ab2 = f32x4_add(cols, ab2);
                    ab3 = f32x4_add(cols, ab3);
                }
                FusedKerSpec::PerColMul(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_mul(cols, ab0);
                    ab1 = f32x4_mul(cols, ab1);
                    ab2 = f32x4_mul(cols, ab2);
                    ab3 = f32x4_mul(cols, ab3);
                }
                FusedKerSpec::PerColSub(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_sub(cols, ab0);
                    ab1 = f32x4_sub(cols, ab1);
                    ab2 = f32x4_sub(cols, ab2);
                    ab3 = f32x4_sub(cols, ab3);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = f32x4_sub(ab0, cols);
                    ab1 = f32x4_sub(ab1, cols);
                    ab2 = f32x4_sub(ab2, cols);
                    ab3 = f32x4_sub(ab3, cols);
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let scale = f32x4_splat(scaler.scale);
                    ab0 = f32x4_mul(scale, ab0);
                    ab1 = f32x4_mul(scale, ab1);
                    ab2 = f32x4_mul(scale, ab2);
                    ab3 = f32x4_mul(scale, ab3);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let shift = f32x4_splat(2f32.powi(-(shift as i32)));
                    ab0 = f32x4_mul(shift, ab0);
                    ab1 = f32x4_mul(shift, ab1);
                    ab2 = f32x4_mul(shift, ab2);
                    ab3 = f32x4_mul(shift, ab3);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let shift = f32x4_splat(2f32.powi(shift as i32));
                    ab0 = f32x4_mul(shift, ab0);
                    ab1 = f32x4_mul(shift, ab1);
                    ab2 = f32x4_mul(shift, ab2);
                    ab3 = f32x4_mul(shift, ab3);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    let mut ptr: *const u8 = tile.ptr;

                    let m0 = *(ptr as *const f32);
                    let m1 = *(ptr.offset(tile.col_byte_stride) as *const f32);
                    let m2 = *(ptr.offset(tile.col_byte_stride * 2) as *const f32);
                    let m3 = *(ptr.offset(tile.col_byte_stride * 3) as *const f32);
                    ab0 = f32x4_add(ab0, f32x4(m0, m1, m2, m3));
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    let m0 = *(ptr as *const f32);
                    let m1 = *(ptr.offset(tile.col_byte_stride) as *const f32);
                    let m2 = *(ptr.offset(tile.col_byte_stride * 2) as *const f32);
                    let m3 = *(ptr.offset(tile.col_byte_stride * 3) as *const f32);
                    ab1 = f32x4_add(ab1, f32x4(m0, m1, m2, m3));
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    let m0 = *(ptr as *const f32);
                    let m1 = *(ptr.offset(tile.col_byte_stride) as *const f32);
                    let m2 = *(ptr.offset(tile.col_byte_stride * 2) as *const f32);
                    let m3 = *(ptr.offset(tile.col_byte_stride * 3) as *const f32);
                    ab2 = f32x4_add(ab2, f32x4(m0, m1, m2, m3));
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    let m0 = *(ptr as *const f32);
                    let m1 = *(ptr.offset(tile.col_byte_stride) as *const f32);
                    let m2 = *(ptr.offset(tile.col_byte_stride * 2) as *const f32);
                    let m3 = *(ptr.offset(tile.col_byte_stride * 3) as *const f32);
                    ab3 = f32x4_add(ab3, f32x4(m0, m1, m2, m3));
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let cols = v128_load(cols as *const v128);
                    ab0 = madd_f32x4_nofma!(ab0, f32x4_splat(*rows.add(0)), cols);
                    ab1 = madd_f32x4_nofma!(ab1, f32x4_splat(*rows.add(1)), cols);
                    ab2 = madd_f32x4_nofma!(ab2, f32x4_splat(*rows.add(2)), cols);
                    ab3 = madd_f32x4_nofma!(ab3, f32x4_splat(*rows.add(3)), cols);
                }
                FusedKerSpec::Store(tile) => {
                    let mut ptr: *mut u8 = tile.ptr;

                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab0);
                    *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab0);
                    *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) =
                        f32x4_extract_lane::<2>(ab0);
                    *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) =
                        f32x4_extract_lane::<3>(ab0);
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab1);
                    *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab1);
                    *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) =
                        f32x4_extract_lane::<2>(ab1);
                    *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) =
                        f32x4_extract_lane::<3>(ab1);
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab2);
                    *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab2);
                    *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) =
                        f32x4_extract_lane::<2>(ab2);
                    *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) =
                        f32x4_extract_lane::<3>(ab2);
                    ptr = ptr.add(tile.row_byte_stride as usize);

                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab3);
                    *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab3);
                    *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) =
                        f32x4_extract_lane::<2>(ab3);
                    *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) =
                        f32x4_extract_lane::<3>(ab3);
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    let a = pa as *const f32;
                    let b = pb as *const v128;
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let b = v128_load(b.offset(i as isize));
                        ab0 = madd_f32x4_nofma!(ab0, f32x4_splat(a[0]), b);
                        ab1 = madd_f32x4_nofma!(ab1, f32x4_splat(a[1]), b);
                        ab2 = madd_f32x4_nofma!(ab2, f32x4_splat(a[2]), b);
                        ab3 = madd_f32x4_nofma!(ab3, f32x4_splat(a[3]), b);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

MMMRustKernel!(kernel_f32_4x4 => wasm_f32_4x4<f32>(4,4)@(4,4) quality(ImplementationQuality::TargetOptimized));

/// WASM SIMD f32 4x1 kernel — GEMV-shaped variant for matrix-vector products
/// (single-column outputs, e.g., streaming-RNN inference where each frame's
/// activation is a single column). Mirrors the 4x4 kernel's FusedKerSpec
/// match arms but collapses the column dimension from 4 to 1: a single
/// f32x4 accumulator holds 4 output rows × 1 output column packed as
/// [ab[0], ab[1], ab[2], ab[3]].
///
/// Selection: tract-core's einsum kernel_selection::strategize() prefers
/// kernels with nr() == 1 when op.n.is_one(), so this kernel is
/// automatically picked for N=1 cases once registered.
unsafe fn kernel_f32_4x1(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // Single accumulator: 4 rows × 1 col, packed into one f32x4.
        // lane[i] holds ab[i] = the output value for row i (col 0).
        let mut ab = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    ab = f32x4_splat(0.0);
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    // Tile is 4 rows × 1 col = 4 contiguous f32s = 1 v128
                    ab = v128_load(rows as *const v128);
                }
                FusedKerSpec::ScalarMin(a) => {
                    ab = f32x4_min(f32x4_splat(a), ab);
                }
                FusedKerSpec::ScalarMax(a) => {
                    ab = f32x4_max(f32x4_splat(a), ab);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    ab = f32x4_add(f32x4_splat(a), ab);
                }
                FusedKerSpec::ScalarMul(a) => {
                    ab = f32x4_mul(f32x4_splat(a), ab);
                }
                FusedKerSpec::ScalarSub(a) => {
                    ab = f32x4_sub(f32x4_splat(a), ab);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    ab = f32x4_sub(ab, f32x4_splat(a));
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let zero = f32x4_splat(0.0);
                    let mask = f32x4_gt(ab, zero);
                    ab = v128_bitselect(ab, f32x4_mul(f32x4_splat(a), ab), mask);
                }
                FusedKerSpec::PerRowMin(row) => {
                    // 4 row values, applied to ab's 4 lanes in order
                    let r = v128_load(row as *const v128);
                    ab = f32x4_min(r, ab);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let r = v128_load(row as *const v128);
                    ab = f32x4_max(r, ab);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let r = v128_load(row as *const v128);
                    ab = f32x4_add(r, ab);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let r = v128_load(row as *const v128);
                    ab = f32x4_mul(r, ab);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let r = v128_load(row as *const v128);
                    ab = f32x4_sub(r, ab);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let r = v128_load(row as *const v128);
                    ab = f32x4_sub(ab, r);
                }
                FusedKerSpec::PerColMin(cols) => {
                    // Single col value broadcast to all 4 rows
                    ab = f32x4_min(f32x4_splat(*cols), ab);
                }
                FusedKerSpec::PerColMax(cols) => {
                    ab = f32x4_max(f32x4_splat(*cols), ab);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    ab = f32x4_add(f32x4_splat(*cols), ab);
                }
                FusedKerSpec::PerColMul(cols) => {
                    ab = f32x4_mul(f32x4_splat(*cols), ab);
                }
                FusedKerSpec::PerColSub(cols) => {
                    ab = f32x4_sub(f32x4_splat(*cols), ab);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    ab = f32x4_sub(ab, f32x4_splat(*cols));
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    ab = f32x4_mul(f32x4_splat(scaler.scale), ab);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    ab = f32x4_mul(s, ab);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    ab = f32x4_mul(s, ab);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    // 4 rows × 1 col, with row_byte_stride between rows (col_stride irrelevant for N=1)
                    let mut ptr: *const u8 = tile.ptr;
                    let m0 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m1 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m2 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m3 = *(ptr as *const f32);
                    ab = f32x4_add(ab, f32x4(m0, m1, m2, m3));
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    // ab[i] += rows[i] * cols[0]  (cols[0] is the single col)
                    let r = v128_load(rows as *const v128);
                    let c = f32x4_splat(*cols);
                    ab = madd_f32x4_nofma!(ab, r, c);
                }
                FusedKerSpec::Store(tile) => {
                    // 4 rows × 1 col, write each lane to a separate row
                    let mut ptr: *mut u8 = tile.ptr;
                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<1>(ab);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<2>(ab);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<3>(ab);
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A is packed [k][MR=4]: each k iter loads 4 contiguous f32s = 1 v128.
                    // B is packed [k][NR=1]: each k iter loads 1 scalar f32, broadcast.
                    // ab[i] += a[i] * b for all i in 0..4 → SIMD: ab += a_vec * b_splat
                    let a = pa as *const v128;
                    let b = pb as *const f32;
                    for i in 0..k {
                        let a_vec = v128_load(a.offset(i as isize));
                        let b_splat = f32x4_splat(*b.offset(i as isize));
                        ab = madd_f32x4_nofma!(ab, a_vec, b_splat);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

// ManuallyOptimized so kernel_selection::strategize honours the M-band
// dispatch in mmv_f32 below. See module-level comment on plug().
MMMRustKernel!(kernel_f32_4x1 => wasm_f32_4x1<f32>(4,1)@(4,1) quality(ImplementationQuality::ManuallyOptimized));

/// WASM SIMD f32 8x1 kernel — wider GEMV variant for matrix-vector products
/// on large M. Uses TWO independent f32x4 accumulators (rows 0-3 in ab_top,
/// rows 4-7 in ab_bot), enabling 2-way ILP within each k-iteration:
/// the inner loop issues two independent f32x4_add(f32x4_mul(...)) ops per
/// k-step, breaking the data-dependency chain depth from K to ~K/2 at the
/// hardware pipeline level.
///
/// Compared to wasm_f32_4x1 (1 accumulator, k-serial dep chain), this is
/// targeted at GEMV ops where M is a multiple of 8 (or close to it). For
/// M=256 GRU gate matmuls (the dominant GEMV in DFN3), this should yield
/// ~2x speedup on the inner loop on hardware where SIMD FMLA throughput
/// exceeds 1 op/cycle.
///
/// Selection: `kernel_selection::strategize()` prefers max mr() for n=1
/// cases, so this kernel automatically wins over wasm_f32_4x1 for all N=1
/// ops once registered (including small-M cases where it slightly wastes
/// rows — for M=1 lsnr_fc-style ops, that's 7-of-8 row waste, but those
/// ops are <1% of frame so the regression is noise).
unsafe fn kernel_f32_8x1(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // Two accumulators: 8 rows × 1 col packed as [ab_top, ab_bot]
        // ab_top.lane[i] holds row i (i in 0..4); ab_bot.lane[i] holds row i+4
        let mut ab_top = f32x4_splat(0.0);
        let mut ab_bot = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    ab_top = f32x4_splat(0.0);
                    ab_bot = f32x4_splat(0.0);
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    // 8 rows × 1 col = 8 contiguous f32 = 2 v128
                    let p = rows as *const v128;
                    ab_top = *p;
                    ab_bot = *p.add(1);
                }
                FusedKerSpec::ScalarMin(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_min(s, ab_top);
                    ab_bot = f32x4_min(s, ab_bot);
                }
                FusedKerSpec::ScalarMax(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_max(s, ab_top);
                    ab_bot = f32x4_max(s, ab_bot);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_add(s, ab_top);
                    ab_bot = f32x4_add(s, ab_bot);
                }
                FusedKerSpec::ScalarMul(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_mul(s, ab_top);
                    ab_bot = f32x4_mul(s, ab_bot);
                }
                FusedKerSpec::ScalarSub(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_sub(s, ab_top);
                    ab_bot = f32x4_sub(s, ab_bot);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let s = f32x4_splat(a);
                    ab_top = f32x4_sub(ab_top, s);
                    ab_bot = f32x4_sub(ab_bot, s);
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let s = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);
                    let mask_t = f32x4_gt(ab_top, zero);
                    let mask_b = f32x4_gt(ab_bot, zero);
                    ab_top = v128_bitselect(ab_top, f32x4_mul(s, ab_top), mask_t);
                    ab_bot = v128_bitselect(ab_bot, f32x4_mul(s, ab_bot), mask_b);
                }
                FusedKerSpec::PerRowMin(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_min(r_t, ab_top);
                    ab_bot = f32x4_min(r_b, ab_bot);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_max(r_t, ab_top);
                    ab_bot = f32x4_max(r_b, ab_bot);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_add(r_t, ab_top);
                    ab_bot = f32x4_add(r_b, ab_bot);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_mul(r_t, ab_top);
                    ab_bot = f32x4_mul(r_b, ab_bot);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_sub(r_t, ab_top);
                    ab_bot = f32x4_sub(r_b, ab_bot);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let p = row as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    ab_top = f32x4_sub(ab_top, r_t);
                    ab_bot = f32x4_sub(ab_bot, r_b);
                }
                FusedKerSpec::PerColMin(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_min(c, ab_top);
                    ab_bot = f32x4_min(c, ab_bot);
                }
                FusedKerSpec::PerColMax(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_max(c, ab_top);
                    ab_bot = f32x4_max(c, ab_bot);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_add(c, ab_top);
                    ab_bot = f32x4_add(c, ab_bot);
                }
                FusedKerSpec::PerColMul(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_mul(c, ab_top);
                    ab_bot = f32x4_mul(c, ab_bot);
                }
                FusedKerSpec::PerColSub(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_sub(c, ab_top);
                    ab_bot = f32x4_sub(c, ab_bot);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_top = f32x4_sub(ab_top, c);
                    ab_bot = f32x4_sub(ab_bot, c);
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let s = f32x4_splat(scaler.scale);
                    ab_top = f32x4_mul(s, ab_top);
                    ab_bot = f32x4_mul(s, ab_bot);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    ab_top = f32x4_mul(s, ab_top);
                    ab_bot = f32x4_mul(s, ab_bot);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    ab_top = f32x4_mul(s, ab_top);
                    ab_bot = f32x4_mul(s, ab_bot);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    // 8 rows × 1 col, stride is row_byte_stride between rows
                    let mut ptr: *const u8 = tile.ptr;
                    let m0 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m1 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m2 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m3 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m4 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m5 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m6 = *(ptr as *const f32);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    let m7 = *(ptr as *const f32);
                    ab_top = f32x4_add(ab_top, f32x4(m0, m1, m2, m3));
                    ab_bot = f32x4_add(ab_bot, f32x4(m4, m5, m6, m7));
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let p = rows as *const v128;
                    let r_t = v128_load(p);
                    let r_b = v128_load(p.add(1));
                    let c = f32x4_splat(*cols);
                    ab_top = madd_f32x4_nofma!(ab_top, r_t, c);
                    ab_bot = madd_f32x4_nofma!(ab_bot, r_b, c);
                }
                FusedKerSpec::Store(tile) => {
                    // 8 rows × 1 col, write each lane to a separate row
                    let mut ptr: *mut u8 = tile.ptr;
                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab_top);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<1>(ab_top);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<2>(ab_top);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<3>(ab_top);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab_bot);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<1>(ab_bot);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<2>(ab_bot);
                    ptr = ptr.add(tile.row_byte_stride as usize);
                    *(ptr as *mut f32) = f32x4_extract_lane::<3>(ab_bot);
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A: packed [k][MR=8] = each k iter loads 8 f32 = 2 v128
                    // B: packed [k][NR=1] = each k iter loads 1 scalar f32, broadcast
                    // The two fmadd ops on (ab_top, ab_bot) are independent — 2-way ILP per iter.
                    let a = pa as *const v128;
                    let b = pb as *const f32;
                    for i in 0..k {
                        let a_t = v128_load(a.offset((2 * i) as isize));
                        let a_b = v128_load(a.offset((2 * i + 1) as isize));
                        let b_splat = f32x4_splat(*b.offset(i as isize));
                        ab_top = madd_f32x4_nofma!(ab_top, a_t, b_splat);
                        ab_bot = madd_f32x4_nofma!(ab_bot, a_b, b_splat);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

MMMRustKernel!(kernel_f32_8x1 => wasm_f32_8x1<f32>(8,1)@(8,1) quality(ImplementationQuality::ManuallyOptimized));

/// WASM SIMD f32 16x1 kernel — wider GEMV variant for matrix-vector products
/// on very large M. Uses FOUR independent f32x4 accumulators (rows 0-3,
/// 4-7, 8-11, 12-15), enabling 4-way ILP within each k-iteration.
///
/// Compared to wasm_f32_8x1 (2 accumulators, 2-way ILP), this exposes more
/// parallel work to the SIMD pipelines, beneficial on hardware with 3+
/// SIMD execution units (most modern ARM and x86).
unsafe fn kernel_f32_16x1(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // Four accumulators: 16 rows × 1 col packed as [ab_q0, ab_q1, ab_q2, ab_q3]
        // ab_q0 = rows 0-3, ab_q1 = rows 4-7, ab_q2 = rows 8-11, ab_q3 = rows 12-15
        let mut ab_q0 = f32x4_splat(0.0);
        let mut ab_q1 = f32x4_splat(0.0);
        let mut ab_q2 = f32x4_splat(0.0);
        let mut ab_q3 = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    let z = f32x4_splat(0.0);
                    ab_q0 = z;
                    ab_q1 = z;
                    ab_q2 = z;
                    ab_q3 = z;
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    let p = rows as *const v128;
                    ab_q0 = *p;
                    ab_q1 = *p.add(1);
                    ab_q2 = *p.add(2);
                    ab_q3 = *p.add(3);
                }
                FusedKerSpec::ScalarMin(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_min(s, ab_q0);
                    ab_q1 = f32x4_min(s, ab_q1);
                    ab_q2 = f32x4_min(s, ab_q2);
                    ab_q3 = f32x4_min(s, ab_q3);
                }
                FusedKerSpec::ScalarMax(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_max(s, ab_q0);
                    ab_q1 = f32x4_max(s, ab_q1);
                    ab_q2 = f32x4_max(s, ab_q2);
                    ab_q3 = f32x4_max(s, ab_q3);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_add(s, ab_q0);
                    ab_q1 = f32x4_add(s, ab_q1);
                    ab_q2 = f32x4_add(s, ab_q2);
                    ab_q3 = f32x4_add(s, ab_q3);
                }
                FusedKerSpec::ScalarMul(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                }
                FusedKerSpec::ScalarSub(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_sub(s, ab_q0);
                    ab_q1 = f32x4_sub(s, ab_q1);
                    ab_q2 = f32x4_sub(s, ab_q2);
                    ab_q3 = f32x4_sub(s, ab_q3);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_sub(ab_q0, s);
                    ab_q1 = f32x4_sub(ab_q1, s);
                    ab_q2 = f32x4_sub(ab_q2, s);
                    ab_q3 = f32x4_sub(ab_q3, s);
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let s = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);
                    let m0 = f32x4_gt(ab_q0, zero);
                    ab_q0 = v128_bitselect(ab_q0, f32x4_mul(s, ab_q0), m0);
                    let m1 = f32x4_gt(ab_q1, zero);
                    ab_q1 = v128_bitselect(ab_q1, f32x4_mul(s, ab_q1), m1);
                    let m2 = f32x4_gt(ab_q2, zero);
                    ab_q2 = v128_bitselect(ab_q2, f32x4_mul(s, ab_q2), m2);
                    let m3 = f32x4_gt(ab_q3, zero);
                    ab_q3 = v128_bitselect(ab_q3, f32x4_mul(s, ab_q3), m3);
                }
                FusedKerSpec::PerRowMin(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_min(v128_load(p), ab_q0);
                    ab_q1 = f32x4_min(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_min(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_min(v128_load(p.add(3)), ab_q3);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_max(v128_load(p), ab_q0);
                    ab_q1 = f32x4_max(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_max(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_max(v128_load(p.add(3)), ab_q3);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_add(v128_load(p), ab_q0);
                    ab_q1 = f32x4_add(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_add(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_add(v128_load(p.add(3)), ab_q3);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_mul(v128_load(p), ab_q0);
                    ab_q1 = f32x4_mul(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_mul(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_mul(v128_load(p.add(3)), ab_q3);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_sub(v128_load(p), ab_q0);
                    ab_q1 = f32x4_sub(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_sub(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_sub(v128_load(p.add(3)), ab_q3);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_sub(ab_q0, v128_load(p));
                    ab_q1 = f32x4_sub(ab_q1, v128_load(p.add(1)));
                    ab_q2 = f32x4_sub(ab_q2, v128_load(p.add(2)));
                    ab_q3 = f32x4_sub(ab_q3, v128_load(p.add(3)));
                }
                FusedKerSpec::PerColMin(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_min(c, ab_q0);
                    ab_q1 = f32x4_min(c, ab_q1);
                    ab_q2 = f32x4_min(c, ab_q2);
                    ab_q3 = f32x4_min(c, ab_q3);
                }
                FusedKerSpec::PerColMax(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_max(c, ab_q0);
                    ab_q1 = f32x4_max(c, ab_q1);
                    ab_q2 = f32x4_max(c, ab_q2);
                    ab_q3 = f32x4_max(c, ab_q3);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_add(c, ab_q0);
                    ab_q1 = f32x4_add(c, ab_q1);
                    ab_q2 = f32x4_add(c, ab_q2);
                    ab_q3 = f32x4_add(c, ab_q3);
                }
                FusedKerSpec::PerColMul(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_mul(c, ab_q0);
                    ab_q1 = f32x4_mul(c, ab_q1);
                    ab_q2 = f32x4_mul(c, ab_q2);
                    ab_q3 = f32x4_mul(c, ab_q3);
                }
                FusedKerSpec::PerColSub(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_sub(c, ab_q0);
                    ab_q1 = f32x4_sub(c, ab_q1);
                    ab_q2 = f32x4_sub(c, ab_q2);
                    ab_q3 = f32x4_sub(c, ab_q3);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_sub(ab_q0, c);
                    ab_q1 = f32x4_sub(ab_q1, c);
                    ab_q2 = f32x4_sub(ab_q2, c);
                    ab_q3 = f32x4_sub(ab_q3, c);
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let s = f32x4_splat(scaler.scale);
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    // 16 rows × 1 col, with row_byte_stride between rows
                    let mut ptr: *const u8 = tile.ptr;
                    let mut ms = [0f32; 16];
                    for i in 0..16 {
                        ms[i] = *(ptr as *const f32);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                    ab_q0 = f32x4_add(ab_q0, f32x4(ms[0], ms[1], ms[2], ms[3]));
                    ab_q1 = f32x4_add(ab_q1, f32x4(ms[4], ms[5], ms[6], ms[7]));
                    ab_q2 = f32x4_add(ab_q2, f32x4(ms[8], ms[9], ms[10], ms[11]));
                    ab_q3 = f32x4_add(ab_q3, f32x4(ms[12], ms[13], ms[14], ms[15]));
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let p = rows as *const v128;
                    let c = f32x4_splat(*cols);
                    ab_q0 = madd_f32x4_nofma!(ab_q0, v128_load(p), c);
                    ab_q1 = madd_f32x4_nofma!(ab_q1, v128_load(p.add(1)), c);
                    ab_q2 = madd_f32x4_nofma!(ab_q2, v128_load(p.add(2)), c);
                    ab_q3 = madd_f32x4_nofma!(ab_q3, v128_load(p.add(3)), c);
                }
                FusedKerSpec::Store(tile) => {
                    // 16 rows × 1 col, write each lane to a separate row
                    let mut ptr: *mut u8 = tile.ptr;
                    for ab in [ab_q0, ab_q1, ab_q2, ab_q3].iter() {
                        *(ptr as *mut f32) = f32x4_extract_lane::<0>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<1>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<2>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<3>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A: packed [k][MR=16] = each k iter loads 16 f32 = 4 v128
                    // B: packed [k][NR=1] = each k iter loads 1 scalar f32, broadcast
                    // 4 INDEPENDENT fmadds per k-iter — 4-way ILP
                    let a = pa as *const v128;
                    let b = pb as *const f32;
                    for i in 0..k {
                        let a0 = v128_load(a.offset((4 * i) as isize));
                        let a1 = v128_load(a.offset((4 * i + 1) as isize));
                        let a2 = v128_load(a.offset((4 * i + 2) as isize));
                        let a3 = v128_load(a.offset((4 * i + 3) as isize));
                        let bs = f32x4_splat(*b.offset(i as isize));
                        ab_q0 = madd_f32x4_nofma!(ab_q0, a0, bs);
                        ab_q1 = madd_f32x4_nofma!(ab_q1, a1, bs);
                        ab_q2 = madd_f32x4_nofma!(ab_q2, a2, bs);
                        ab_q3 = madd_f32x4_nofma!(ab_q3, a3, bs);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

MMMRustKernel!(kernel_f32_16x1 => wasm_f32_16x1<f32>(16,1)@(16,1) quality(ImplementationQuality::ManuallyOptimized));

/// WASM SIMD f32 32x1 kernel — widest GEMV variant for matrix-vector products
/// on very large M. Uses EIGHT independent f32x4 accumulators (rows 0-3, 4-7,
/// 8-11, 12-15, 16-19, 20-23, 24-27, 28-31), enabling 8-way ILP within each
/// k-iteration.
///
/// Compared to wasm_f32_16x1 (4 accumulators, 4-way ILP), this halves the
/// per-call dispatch overhead for M=256 GRU gates (8 calls instead of 16),
/// and exposes 8 independent fmadd dependency chains. On hosts with 16+
/// physical SIMD registers (x86_64 has 16 xmm, ARM64 has 32 NEON), the 8
/// accumulators fit without spilling. Mirrors `apple_amx_mmm_f32_32x1` MR.
///
/// Selection: `kernel_selection::strategize()` prefers max mr() for n=1
/// cases, so this kernel automatically wins over wasm_f32_16x1 for M >= 32.
unsafe fn kernel_f32_32x1(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // Eight accumulators: 32 rows × 1 col packed as [ab_q0..ab_q7]
        // ab_q0 = rows 0-3, ab_q1 = rows 4-7, ..., ab_q7 = rows 28-31
        let mut ab_q0 = f32x4_splat(0.0);
        let mut ab_q1 = f32x4_splat(0.0);
        let mut ab_q2 = f32x4_splat(0.0);
        let mut ab_q3 = f32x4_splat(0.0);
        let mut ab_q4 = f32x4_splat(0.0);
        let mut ab_q5 = f32x4_splat(0.0);
        let mut ab_q6 = f32x4_splat(0.0);
        let mut ab_q7 = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    let z = f32x4_splat(0.0);
                    ab_q0 = z;
                    ab_q1 = z;
                    ab_q2 = z;
                    ab_q3 = z;
                    ab_q4 = z;
                    ab_q5 = z;
                    ab_q6 = z;
                    ab_q7 = z;
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    let p = rows as *const v128;
                    ab_q0 = *p;
                    ab_q1 = *p.add(1);
                    ab_q2 = *p.add(2);
                    ab_q3 = *p.add(3);
                    ab_q4 = *p.add(4);
                    ab_q5 = *p.add(5);
                    ab_q6 = *p.add(6);
                    ab_q7 = *p.add(7);
                }
                FusedKerSpec::ScalarMin(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_min(s, ab_q0);
                    ab_q1 = f32x4_min(s, ab_q1);
                    ab_q2 = f32x4_min(s, ab_q2);
                    ab_q3 = f32x4_min(s, ab_q3);
                    ab_q4 = f32x4_min(s, ab_q4);
                    ab_q5 = f32x4_min(s, ab_q5);
                    ab_q6 = f32x4_min(s, ab_q6);
                    ab_q7 = f32x4_min(s, ab_q7);
                }
                FusedKerSpec::ScalarMax(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_max(s, ab_q0);
                    ab_q1 = f32x4_max(s, ab_q1);
                    ab_q2 = f32x4_max(s, ab_q2);
                    ab_q3 = f32x4_max(s, ab_q3);
                    ab_q4 = f32x4_max(s, ab_q4);
                    ab_q5 = f32x4_max(s, ab_q5);
                    ab_q6 = f32x4_max(s, ab_q6);
                    ab_q7 = f32x4_max(s, ab_q7);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_add(s, ab_q0);
                    ab_q1 = f32x4_add(s, ab_q1);
                    ab_q2 = f32x4_add(s, ab_q2);
                    ab_q3 = f32x4_add(s, ab_q3);
                    ab_q4 = f32x4_add(s, ab_q4);
                    ab_q5 = f32x4_add(s, ab_q5);
                    ab_q6 = f32x4_add(s, ab_q6);
                    ab_q7 = f32x4_add(s, ab_q7);
                }
                FusedKerSpec::ScalarMul(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                    ab_q4 = f32x4_mul(s, ab_q4);
                    ab_q5 = f32x4_mul(s, ab_q5);
                    ab_q6 = f32x4_mul(s, ab_q6);
                    ab_q7 = f32x4_mul(s, ab_q7);
                }
                FusedKerSpec::ScalarSub(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_sub(s, ab_q0);
                    ab_q1 = f32x4_sub(s, ab_q1);
                    ab_q2 = f32x4_sub(s, ab_q2);
                    ab_q3 = f32x4_sub(s, ab_q3);
                    ab_q4 = f32x4_sub(s, ab_q4);
                    ab_q5 = f32x4_sub(s, ab_q5);
                    ab_q6 = f32x4_sub(s, ab_q6);
                    ab_q7 = f32x4_sub(s, ab_q7);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let s = f32x4_splat(a);
                    ab_q0 = f32x4_sub(ab_q0, s);
                    ab_q1 = f32x4_sub(ab_q1, s);
                    ab_q2 = f32x4_sub(ab_q2, s);
                    ab_q3 = f32x4_sub(ab_q3, s);
                    ab_q4 = f32x4_sub(ab_q4, s);
                    ab_q5 = f32x4_sub(ab_q5, s);
                    ab_q6 = f32x4_sub(ab_q6, s);
                    ab_q7 = f32x4_sub(ab_q7, s);
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let s = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);
                    let m0 = f32x4_gt(ab_q0, zero);
                    ab_q0 = v128_bitselect(ab_q0, f32x4_mul(s, ab_q0), m0);
                    let m1 = f32x4_gt(ab_q1, zero);
                    ab_q1 = v128_bitselect(ab_q1, f32x4_mul(s, ab_q1), m1);
                    let m2 = f32x4_gt(ab_q2, zero);
                    ab_q2 = v128_bitselect(ab_q2, f32x4_mul(s, ab_q2), m2);
                    let m3 = f32x4_gt(ab_q3, zero);
                    ab_q3 = v128_bitselect(ab_q3, f32x4_mul(s, ab_q3), m3);
                    let m4 = f32x4_gt(ab_q4, zero);
                    ab_q4 = v128_bitselect(ab_q4, f32x4_mul(s, ab_q4), m4);
                    let m5 = f32x4_gt(ab_q5, zero);
                    ab_q5 = v128_bitselect(ab_q5, f32x4_mul(s, ab_q5), m5);
                    let m6 = f32x4_gt(ab_q6, zero);
                    ab_q6 = v128_bitselect(ab_q6, f32x4_mul(s, ab_q6), m6);
                    let m7 = f32x4_gt(ab_q7, zero);
                    ab_q7 = v128_bitselect(ab_q7, f32x4_mul(s, ab_q7), m7);
                }
                FusedKerSpec::PerRowMin(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_min(v128_load(p), ab_q0);
                    ab_q1 = f32x4_min(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_min(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_min(v128_load(p.add(3)), ab_q3);
                    ab_q4 = f32x4_min(v128_load(p.add(4)), ab_q4);
                    ab_q5 = f32x4_min(v128_load(p.add(5)), ab_q5);
                    ab_q6 = f32x4_min(v128_load(p.add(6)), ab_q6);
                    ab_q7 = f32x4_min(v128_load(p.add(7)), ab_q7);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_max(v128_load(p), ab_q0);
                    ab_q1 = f32x4_max(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_max(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_max(v128_load(p.add(3)), ab_q3);
                    ab_q4 = f32x4_max(v128_load(p.add(4)), ab_q4);
                    ab_q5 = f32x4_max(v128_load(p.add(5)), ab_q5);
                    ab_q6 = f32x4_max(v128_load(p.add(6)), ab_q6);
                    ab_q7 = f32x4_max(v128_load(p.add(7)), ab_q7);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_add(v128_load(p), ab_q0);
                    ab_q1 = f32x4_add(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_add(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_add(v128_load(p.add(3)), ab_q3);
                    ab_q4 = f32x4_add(v128_load(p.add(4)), ab_q4);
                    ab_q5 = f32x4_add(v128_load(p.add(5)), ab_q5);
                    ab_q6 = f32x4_add(v128_load(p.add(6)), ab_q6);
                    ab_q7 = f32x4_add(v128_load(p.add(7)), ab_q7);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_mul(v128_load(p), ab_q0);
                    ab_q1 = f32x4_mul(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_mul(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_mul(v128_load(p.add(3)), ab_q3);
                    ab_q4 = f32x4_mul(v128_load(p.add(4)), ab_q4);
                    ab_q5 = f32x4_mul(v128_load(p.add(5)), ab_q5);
                    ab_q6 = f32x4_mul(v128_load(p.add(6)), ab_q6);
                    ab_q7 = f32x4_mul(v128_load(p.add(7)), ab_q7);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_sub(v128_load(p), ab_q0);
                    ab_q1 = f32x4_sub(v128_load(p.add(1)), ab_q1);
                    ab_q2 = f32x4_sub(v128_load(p.add(2)), ab_q2);
                    ab_q3 = f32x4_sub(v128_load(p.add(3)), ab_q3);
                    ab_q4 = f32x4_sub(v128_load(p.add(4)), ab_q4);
                    ab_q5 = f32x4_sub(v128_load(p.add(5)), ab_q5);
                    ab_q6 = f32x4_sub(v128_load(p.add(6)), ab_q6);
                    ab_q7 = f32x4_sub(v128_load(p.add(7)), ab_q7);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let p = row as *const v128;
                    ab_q0 = f32x4_sub(ab_q0, v128_load(p));
                    ab_q1 = f32x4_sub(ab_q1, v128_load(p.add(1)));
                    ab_q2 = f32x4_sub(ab_q2, v128_load(p.add(2)));
                    ab_q3 = f32x4_sub(ab_q3, v128_load(p.add(3)));
                    ab_q4 = f32x4_sub(ab_q4, v128_load(p.add(4)));
                    ab_q5 = f32x4_sub(ab_q5, v128_load(p.add(5)));
                    ab_q6 = f32x4_sub(ab_q6, v128_load(p.add(6)));
                    ab_q7 = f32x4_sub(ab_q7, v128_load(p.add(7)));
                }
                FusedKerSpec::PerColMin(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_min(c, ab_q0);
                    ab_q1 = f32x4_min(c, ab_q1);
                    ab_q2 = f32x4_min(c, ab_q2);
                    ab_q3 = f32x4_min(c, ab_q3);
                    ab_q4 = f32x4_min(c, ab_q4);
                    ab_q5 = f32x4_min(c, ab_q5);
                    ab_q6 = f32x4_min(c, ab_q6);
                    ab_q7 = f32x4_min(c, ab_q7);
                }
                FusedKerSpec::PerColMax(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_max(c, ab_q0);
                    ab_q1 = f32x4_max(c, ab_q1);
                    ab_q2 = f32x4_max(c, ab_q2);
                    ab_q3 = f32x4_max(c, ab_q3);
                    ab_q4 = f32x4_max(c, ab_q4);
                    ab_q5 = f32x4_max(c, ab_q5);
                    ab_q6 = f32x4_max(c, ab_q6);
                    ab_q7 = f32x4_max(c, ab_q7);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_add(c, ab_q0);
                    ab_q1 = f32x4_add(c, ab_q1);
                    ab_q2 = f32x4_add(c, ab_q2);
                    ab_q3 = f32x4_add(c, ab_q3);
                    ab_q4 = f32x4_add(c, ab_q4);
                    ab_q5 = f32x4_add(c, ab_q5);
                    ab_q6 = f32x4_add(c, ab_q6);
                    ab_q7 = f32x4_add(c, ab_q7);
                }
                FusedKerSpec::PerColMul(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_mul(c, ab_q0);
                    ab_q1 = f32x4_mul(c, ab_q1);
                    ab_q2 = f32x4_mul(c, ab_q2);
                    ab_q3 = f32x4_mul(c, ab_q3);
                    ab_q4 = f32x4_mul(c, ab_q4);
                    ab_q5 = f32x4_mul(c, ab_q5);
                    ab_q6 = f32x4_mul(c, ab_q6);
                    ab_q7 = f32x4_mul(c, ab_q7);
                }
                FusedKerSpec::PerColSub(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_sub(c, ab_q0);
                    ab_q1 = f32x4_sub(c, ab_q1);
                    ab_q2 = f32x4_sub(c, ab_q2);
                    ab_q3 = f32x4_sub(c, ab_q3);
                    ab_q4 = f32x4_sub(c, ab_q4);
                    ab_q5 = f32x4_sub(c, ab_q5);
                    ab_q6 = f32x4_sub(c, ab_q6);
                    ab_q7 = f32x4_sub(c, ab_q7);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let c = f32x4_splat(*cols);
                    ab_q0 = f32x4_sub(ab_q0, c);
                    ab_q1 = f32x4_sub(ab_q1, c);
                    ab_q2 = f32x4_sub(ab_q2, c);
                    ab_q3 = f32x4_sub(ab_q3, c);
                    ab_q4 = f32x4_sub(ab_q4, c);
                    ab_q5 = f32x4_sub(ab_q5, c);
                    ab_q6 = f32x4_sub(ab_q6, c);
                    ab_q7 = f32x4_sub(ab_q7, c);
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let s = f32x4_splat(scaler.scale);
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                    ab_q4 = f32x4_mul(s, ab_q4);
                    ab_q5 = f32x4_mul(s, ab_q5);
                    ab_q6 = f32x4_mul(s, ab_q6);
                    ab_q7 = f32x4_mul(s, ab_q7);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                    ab_q4 = f32x4_mul(s, ab_q4);
                    ab_q5 = f32x4_mul(s, ab_q5);
                    ab_q6 = f32x4_mul(s, ab_q6);
                    ab_q7 = f32x4_mul(s, ab_q7);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    ab_q0 = f32x4_mul(s, ab_q0);
                    ab_q1 = f32x4_mul(s, ab_q1);
                    ab_q2 = f32x4_mul(s, ab_q2);
                    ab_q3 = f32x4_mul(s, ab_q3);
                    ab_q4 = f32x4_mul(s, ab_q4);
                    ab_q5 = f32x4_mul(s, ab_q5);
                    ab_q6 = f32x4_mul(s, ab_q6);
                    ab_q7 = f32x4_mul(s, ab_q7);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    // 32 rows × 1 col, with row_byte_stride between rows
                    let mut ptr: *const u8 = tile.ptr;
                    let mut ms = [0f32; 32];
                    for i in 0..32 {
                        ms[i] = *(ptr as *const f32);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                    ab_q0 = f32x4_add(ab_q0, f32x4(ms[0], ms[1], ms[2], ms[3]));
                    ab_q1 = f32x4_add(ab_q1, f32x4(ms[4], ms[5], ms[6], ms[7]));
                    ab_q2 = f32x4_add(ab_q2, f32x4(ms[8], ms[9], ms[10], ms[11]));
                    ab_q3 = f32x4_add(ab_q3, f32x4(ms[12], ms[13], ms[14], ms[15]));
                    ab_q4 = f32x4_add(ab_q4, f32x4(ms[16], ms[17], ms[18], ms[19]));
                    ab_q5 = f32x4_add(ab_q5, f32x4(ms[20], ms[21], ms[22], ms[23]));
                    ab_q6 = f32x4_add(ab_q6, f32x4(ms[24], ms[25], ms[26], ms[27]));
                    ab_q7 = f32x4_add(ab_q7, f32x4(ms[28], ms[29], ms[30], ms[31]));
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let p = rows as *const v128;
                    let c = f32x4_splat(*cols);
                    ab_q0 = madd_f32x4!(ab_q0, v128_load(p), c);
                    ab_q1 = madd_f32x4!(ab_q1, v128_load(p.add(1)), c);
                    ab_q2 = madd_f32x4!(ab_q2, v128_load(p.add(2)), c);
                    ab_q3 = madd_f32x4!(ab_q3, v128_load(p.add(3)), c);
                    ab_q4 = madd_f32x4!(ab_q4, v128_load(p.add(4)), c);
                    ab_q5 = madd_f32x4!(ab_q5, v128_load(p.add(5)), c);
                    ab_q6 = madd_f32x4!(ab_q6, v128_load(p.add(6)), c);
                    ab_q7 = madd_f32x4!(ab_q7, v128_load(p.add(7)), c);
                }
                FusedKerSpec::Store(tile) => {
                    // 32 rows × 1 col, write each lane to a separate row
                    let mut ptr: *mut u8 = tile.ptr;
                    for ab in [ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7].iter() {
                        *(ptr as *mut f32) = f32x4_extract_lane::<0>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<1>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<2>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                        *(ptr as *mut f32) = f32x4_extract_lane::<3>(*ab);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A: packed [k][MR=32] = each k iter loads 32 f32 = 8 v128
                    // B: packed [k][NR=1] = each k iter loads 1 scalar f32, broadcast
                    // 8 INDEPENDENT fmadds per k-iter — 8-way ILP
                    let a = pa as *const v128;
                    let b = pb as *const f32;
                    for i in 0..k {
                        let a0 = v128_load(a.offset((8 * i) as isize));
                        let a1 = v128_load(a.offset((8 * i + 1) as isize));
                        let a2 = v128_load(a.offset((8 * i + 2) as isize));
                        let a3 = v128_load(a.offset((8 * i + 3) as isize));
                        let a4 = v128_load(a.offset((8 * i + 4) as isize));
                        let a5 = v128_load(a.offset((8 * i + 5) as isize));
                        let a6 = v128_load(a.offset((8 * i + 6) as isize));
                        let a7 = v128_load(a.offset((8 * i + 7) as isize));
                        let bs = f32x4_splat(*b.offset(i as isize));
                        ab_q0 = madd_f32x4!(ab_q0, a0, bs);
                        ab_q1 = madd_f32x4!(ab_q1, a1, bs);
                        ab_q2 = madd_f32x4!(ab_q2, a2, bs);
                        ab_q3 = madd_f32x4!(ab_q3, a3, bs);
                        ab_q4 = madd_f32x4!(ab_q4, a4, bs);
                        ab_q5 = madd_f32x4!(ab_q5, a5, bs);
                        ab_q6 = madd_f32x4!(ab_q6, a6, bs);
                        ab_q7 = madd_f32x4!(ab_q7, a7, bs);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

MMMRustKernel!(kernel_f32_32x1 => wasm_f32_32x1<f32>(32,1)@(32,1) quality(ImplementationQuality::ManuallyOptimized));

/// WASM SIMD f32 8x8 kernel — wide MM tile (8 rows × 8 cols, 16 v128 accumulators).
/// Each row uses 2 v128: cols 0-3 in `_lo`, cols 4-7 in `_hi`. 16 accumulators
/// is at the limit of WASM's 16 logical SIMD register slots; this tests the
/// register-pressure boundary. For DFN3 ops, all M and N are multiples of 8,
/// so 8x8 fits cleanly with no padding waste.
unsafe fn kernel_f32_8x8(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        // 8 rows × 8 cols = 16 f32x4 accumulators (cols 0-3 in _lo, cols 4-7 in _hi)
        let mut a0lo = f32x4_splat(0.0);
        let mut a0hi = f32x4_splat(0.0);
        let mut a1lo = f32x4_splat(0.0);
        let mut a1hi = f32x4_splat(0.0);
        let mut a2lo = f32x4_splat(0.0);
        let mut a2hi = f32x4_splat(0.0);
        let mut a3lo = f32x4_splat(0.0);
        let mut a3hi = f32x4_splat(0.0);
        let mut a4lo = f32x4_splat(0.0);
        let mut a4hi = f32x4_splat(0.0);
        let mut a5lo = f32x4_splat(0.0);
        let mut a5hi = f32x4_splat(0.0);
        let mut a6lo = f32x4_splat(0.0);
        let mut a6hi = f32x4_splat(0.0);
        let mut a7lo = f32x4_splat(0.0);
        let mut a7hi = f32x4_splat(0.0);

        while !pnl.is_null() {
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => {
                    let z = f32x4_splat(0.0);
                    a0lo = z;
                    a0hi = z;
                    a1lo = z;
                    a1hi = z;
                    a2lo = z;
                    a2hi = z;
                    a3lo = z;
                    a3hi = z;
                    a4lo = z;
                    a4hi = z;
                    a5lo = z;
                    a5hi = z;
                    a6lo = z;
                    a6hi = z;
                    a7lo = z;
                    a7hi = z;
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    // 8 rows × 8 cols = 16 v128 (2 per row, contiguous lo+hi)
                    let p = rows as *const v128;
                    a0lo = *p.add(0);
                    a0hi = *p.add(1);
                    a1lo = *p.add(2);
                    a1hi = *p.add(3);
                    a2lo = *p.add(4);
                    a2hi = *p.add(5);
                    a3lo = *p.add(6);
                    a3hi = *p.add(7);
                    a4lo = *p.add(8);
                    a4hi = *p.add(9);
                    a5lo = *p.add(10);
                    a5hi = *p.add(11);
                    a6lo = *p.add(12);
                    a6hi = *p.add(13);
                    a7lo = *p.add(14);
                    a7hi = *p.add(15);
                }
                FusedKerSpec::ScalarMin(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_min(s, a0lo);
                    a0hi = f32x4_min(s, a0hi);
                    a1lo = f32x4_min(s, a1lo);
                    a1hi = f32x4_min(s, a1hi);
                    a2lo = f32x4_min(s, a2lo);
                    a2hi = f32x4_min(s, a2hi);
                    a3lo = f32x4_min(s, a3lo);
                    a3hi = f32x4_min(s, a3hi);
                    a4lo = f32x4_min(s, a4lo);
                    a4hi = f32x4_min(s, a4hi);
                    a5lo = f32x4_min(s, a5lo);
                    a5hi = f32x4_min(s, a5hi);
                    a6lo = f32x4_min(s, a6lo);
                    a6hi = f32x4_min(s, a6hi);
                    a7lo = f32x4_min(s, a7lo);
                    a7hi = f32x4_min(s, a7hi);
                }
                FusedKerSpec::ScalarMax(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_max(s, a0lo);
                    a0hi = f32x4_max(s, a0hi);
                    a1lo = f32x4_max(s, a1lo);
                    a1hi = f32x4_max(s, a1hi);
                    a2lo = f32x4_max(s, a2lo);
                    a2hi = f32x4_max(s, a2hi);
                    a3lo = f32x4_max(s, a3lo);
                    a3hi = f32x4_max(s, a3hi);
                    a4lo = f32x4_max(s, a4lo);
                    a4hi = f32x4_max(s, a4hi);
                    a5lo = f32x4_max(s, a5lo);
                    a5hi = f32x4_max(s, a5hi);
                    a6lo = f32x4_max(s, a6lo);
                    a6hi = f32x4_max(s, a6hi);
                    a7lo = f32x4_max(s, a7lo);
                    a7hi = f32x4_max(s, a7hi);
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_add(s, a0lo);
                    a0hi = f32x4_add(s, a0hi);
                    a1lo = f32x4_add(s, a1lo);
                    a1hi = f32x4_add(s, a1hi);
                    a2lo = f32x4_add(s, a2lo);
                    a2hi = f32x4_add(s, a2hi);
                    a3lo = f32x4_add(s, a3lo);
                    a3hi = f32x4_add(s, a3hi);
                    a4lo = f32x4_add(s, a4lo);
                    a4hi = f32x4_add(s, a4hi);
                    a5lo = f32x4_add(s, a5lo);
                    a5hi = f32x4_add(s, a5hi);
                    a6lo = f32x4_add(s, a6lo);
                    a6hi = f32x4_add(s, a6hi);
                    a7lo = f32x4_add(s, a7lo);
                    a7hi = f32x4_add(s, a7hi);
                }
                FusedKerSpec::ScalarMul(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_mul(s, a0lo);
                    a0hi = f32x4_mul(s, a0hi);
                    a1lo = f32x4_mul(s, a1lo);
                    a1hi = f32x4_mul(s, a1hi);
                    a2lo = f32x4_mul(s, a2lo);
                    a2hi = f32x4_mul(s, a2hi);
                    a3lo = f32x4_mul(s, a3lo);
                    a3hi = f32x4_mul(s, a3hi);
                    a4lo = f32x4_mul(s, a4lo);
                    a4hi = f32x4_mul(s, a4hi);
                    a5lo = f32x4_mul(s, a5lo);
                    a5hi = f32x4_mul(s, a5hi);
                    a6lo = f32x4_mul(s, a6lo);
                    a6hi = f32x4_mul(s, a6hi);
                    a7lo = f32x4_mul(s, a7lo);
                    a7hi = f32x4_mul(s, a7hi);
                }
                FusedKerSpec::ScalarSub(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_sub(s, a0lo);
                    a0hi = f32x4_sub(s, a0hi);
                    a1lo = f32x4_sub(s, a1lo);
                    a1hi = f32x4_sub(s, a1hi);
                    a2lo = f32x4_sub(s, a2lo);
                    a2hi = f32x4_sub(s, a2hi);
                    a3lo = f32x4_sub(s, a3lo);
                    a3hi = f32x4_sub(s, a3hi);
                    a4lo = f32x4_sub(s, a4lo);
                    a4hi = f32x4_sub(s, a4hi);
                    a5lo = f32x4_sub(s, a5lo);
                    a5hi = f32x4_sub(s, a5hi);
                    a6lo = f32x4_sub(s, a6lo);
                    a6hi = f32x4_sub(s, a6hi);
                    a7lo = f32x4_sub(s, a7lo);
                    a7hi = f32x4_sub(s, a7hi);
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let s = f32x4_splat(a);
                    a0lo = f32x4_sub(a0lo, s);
                    a0hi = f32x4_sub(a0hi, s);
                    a1lo = f32x4_sub(a1lo, s);
                    a1hi = f32x4_sub(a1hi, s);
                    a2lo = f32x4_sub(a2lo, s);
                    a2hi = f32x4_sub(a2hi, s);
                    a3lo = f32x4_sub(a3lo, s);
                    a3hi = f32x4_sub(a3hi, s);
                    a4lo = f32x4_sub(a4lo, s);
                    a4hi = f32x4_sub(a4hi, s);
                    a5lo = f32x4_sub(a5lo, s);
                    a5hi = f32x4_sub(a5hi, s);
                    a6lo = f32x4_sub(a6lo, s);
                    a6hi = f32x4_sub(a6hi, s);
                    a7lo = f32x4_sub(a7lo, s);
                    a7hi = f32x4_sub(a7hi, s);
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let s = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);
                    let m0a = f32x4_gt(a0lo, zero);
                    a0lo = v128_bitselect(a0lo, f32x4_mul(s, a0lo), m0a);
                    let m0b = f32x4_gt(a0hi, zero);
                    a0hi = v128_bitselect(a0hi, f32x4_mul(s, a0hi), m0b);
                    let m1a = f32x4_gt(a1lo, zero);
                    a1lo = v128_bitselect(a1lo, f32x4_mul(s, a1lo), m1a);
                    let m1b = f32x4_gt(a1hi, zero);
                    a1hi = v128_bitselect(a1hi, f32x4_mul(s, a1hi), m1b);
                    let m2a = f32x4_gt(a2lo, zero);
                    a2lo = v128_bitselect(a2lo, f32x4_mul(s, a2lo), m2a);
                    let m2b = f32x4_gt(a2hi, zero);
                    a2hi = v128_bitselect(a2hi, f32x4_mul(s, a2hi), m2b);
                    let m3a = f32x4_gt(a3lo, zero);
                    a3lo = v128_bitselect(a3lo, f32x4_mul(s, a3lo), m3a);
                    let m3b = f32x4_gt(a3hi, zero);
                    a3hi = v128_bitselect(a3hi, f32x4_mul(s, a3hi), m3b);
                    let m4a = f32x4_gt(a4lo, zero);
                    a4lo = v128_bitselect(a4lo, f32x4_mul(s, a4lo), m4a);
                    let m4b = f32x4_gt(a4hi, zero);
                    a4hi = v128_bitselect(a4hi, f32x4_mul(s, a4hi), m4b);
                    let m5a = f32x4_gt(a5lo, zero);
                    a5lo = v128_bitselect(a5lo, f32x4_mul(s, a5lo), m5a);
                    let m5b = f32x4_gt(a5hi, zero);
                    a5hi = v128_bitselect(a5hi, f32x4_mul(s, a5hi), m5b);
                    let m6a = f32x4_gt(a6lo, zero);
                    a6lo = v128_bitselect(a6lo, f32x4_mul(s, a6lo), m6a);
                    let m6b = f32x4_gt(a6hi, zero);
                    a6hi = v128_bitselect(a6hi, f32x4_mul(s, a6hi), m6b);
                    let m7a = f32x4_gt(a7lo, zero);
                    a7lo = v128_bitselect(a7lo, f32x4_mul(s, a7lo), m7a);
                    let m7b = f32x4_gt(a7hi, zero);
                    a7hi = v128_bitselect(a7hi, f32x4_mul(s, a7hi), m7b);
                }
                FusedKerSpec::PerRowMin(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_min(r0, a0lo);
                    a0hi = f32x4_min(r0, a0hi);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_min(r1, a1lo);
                    a1hi = f32x4_min(r1, a1hi);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_min(r2, a2lo);
                    a2hi = f32x4_min(r2, a2hi);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_min(r3, a3lo);
                    a3hi = f32x4_min(r3, a3hi);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_min(r4, a4lo);
                    a4hi = f32x4_min(r4, a4hi);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_min(r5, a5lo);
                    a5hi = f32x4_min(r5, a5hi);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_min(r6, a6lo);
                    a6hi = f32x4_min(r6, a6hi);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_min(r7, a7lo);
                    a7hi = f32x4_min(r7, a7hi);
                }
                FusedKerSpec::PerRowMax(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_max(r0, a0lo);
                    a0hi = f32x4_max(r0, a0hi);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_max(r1, a1lo);
                    a1hi = f32x4_max(r1, a1hi);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_max(r2, a2lo);
                    a2hi = f32x4_max(r2, a2hi);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_max(r3, a3lo);
                    a3hi = f32x4_max(r3, a3hi);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_max(r4, a4lo);
                    a4hi = f32x4_max(r4, a4hi);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_max(r5, a5lo);
                    a5hi = f32x4_max(r5, a5hi);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_max(r6, a6lo);
                    a6hi = f32x4_max(r6, a6hi);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_max(r7, a7lo);
                    a7hi = f32x4_max(r7, a7hi);
                }
                FusedKerSpec::PerRowAdd(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_add(r0, a0lo);
                    a0hi = f32x4_add(r0, a0hi);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_add(r1, a1lo);
                    a1hi = f32x4_add(r1, a1hi);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_add(r2, a2lo);
                    a2hi = f32x4_add(r2, a2hi);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_add(r3, a3lo);
                    a3hi = f32x4_add(r3, a3hi);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_add(r4, a4lo);
                    a4hi = f32x4_add(r4, a4hi);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_add(r5, a5lo);
                    a5hi = f32x4_add(r5, a5hi);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_add(r6, a6lo);
                    a6hi = f32x4_add(r6, a6hi);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_add(r7, a7lo);
                    a7hi = f32x4_add(r7, a7hi);
                }
                FusedKerSpec::PerRowMul(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_mul(r0, a0lo);
                    a0hi = f32x4_mul(r0, a0hi);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_mul(r1, a1lo);
                    a1hi = f32x4_mul(r1, a1hi);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_mul(r2, a2lo);
                    a2hi = f32x4_mul(r2, a2hi);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_mul(r3, a3lo);
                    a3hi = f32x4_mul(r3, a3hi);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_mul(r4, a4lo);
                    a4hi = f32x4_mul(r4, a4hi);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_mul(r5, a5lo);
                    a5hi = f32x4_mul(r5, a5hi);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_mul(r6, a6lo);
                    a6hi = f32x4_mul(r6, a6hi);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_mul(r7, a7lo);
                    a7hi = f32x4_mul(r7, a7hi);
                }
                FusedKerSpec::PerRowSub(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_sub(r0, a0lo);
                    a0hi = f32x4_sub(r0, a0hi);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_sub(r1, a1lo);
                    a1hi = f32x4_sub(r1, a1hi);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_sub(r2, a2lo);
                    a2hi = f32x4_sub(r2, a2hi);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_sub(r3, a3lo);
                    a3hi = f32x4_sub(r3, a3hi);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_sub(r4, a4lo);
                    a4hi = f32x4_sub(r4, a4hi);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_sub(r5, a5lo);
                    a5hi = f32x4_sub(r5, a5hi);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_sub(r6, a6lo);
                    a6hi = f32x4_sub(r6, a6hi);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_sub(r7, a7lo);
                    a7hi = f32x4_sub(r7, a7hi);
                }
                FusedKerSpec::PerRowSubF(row) => {
                    let r = std::slice::from_raw_parts(row, 8);
                    let r0 = f32x4_splat(r[0]);
                    a0lo = f32x4_sub(a0lo, r0);
                    a0hi = f32x4_sub(a0hi, r0);
                    let r1 = f32x4_splat(r[1]);
                    a1lo = f32x4_sub(a1lo, r1);
                    a1hi = f32x4_sub(a1hi, r1);
                    let r2 = f32x4_splat(r[2]);
                    a2lo = f32x4_sub(a2lo, r2);
                    a2hi = f32x4_sub(a2hi, r2);
                    let r3 = f32x4_splat(r[3]);
                    a3lo = f32x4_sub(a3lo, r3);
                    a3hi = f32x4_sub(a3hi, r3);
                    let r4 = f32x4_splat(r[4]);
                    a4lo = f32x4_sub(a4lo, r4);
                    a4hi = f32x4_sub(a4hi, r4);
                    let r5 = f32x4_splat(r[5]);
                    a5lo = f32x4_sub(a5lo, r5);
                    a5hi = f32x4_sub(a5hi, r5);
                    let r6 = f32x4_splat(r[6]);
                    a6lo = f32x4_sub(a6lo, r6);
                    a6hi = f32x4_sub(a6hi, r6);
                    let r7 = f32x4_splat(r[7]);
                    a7lo = f32x4_sub(a7lo, r7);
                    a7hi = f32x4_sub(a7hi, r7);
                }
                FusedKerSpec::PerColMin(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_min(clo, a0lo);
                    a0hi = f32x4_min(chi, a0hi);
                    a1lo = f32x4_min(clo, a1lo);
                    a1hi = f32x4_min(chi, a1hi);
                    a2lo = f32x4_min(clo, a2lo);
                    a2hi = f32x4_min(chi, a2hi);
                    a3lo = f32x4_min(clo, a3lo);
                    a3hi = f32x4_min(chi, a3hi);
                    a4lo = f32x4_min(clo, a4lo);
                    a4hi = f32x4_min(chi, a4hi);
                    a5lo = f32x4_min(clo, a5lo);
                    a5hi = f32x4_min(chi, a5hi);
                    a6lo = f32x4_min(clo, a6lo);
                    a6hi = f32x4_min(chi, a6hi);
                    a7lo = f32x4_min(clo, a7lo);
                    a7hi = f32x4_min(chi, a7hi);
                }
                FusedKerSpec::PerColMax(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_max(clo, a0lo);
                    a0hi = f32x4_max(chi, a0hi);
                    a1lo = f32x4_max(clo, a1lo);
                    a1hi = f32x4_max(chi, a1hi);
                    a2lo = f32x4_max(clo, a2lo);
                    a2hi = f32x4_max(chi, a2hi);
                    a3lo = f32x4_max(clo, a3lo);
                    a3hi = f32x4_max(chi, a3hi);
                    a4lo = f32x4_max(clo, a4lo);
                    a4hi = f32x4_max(chi, a4hi);
                    a5lo = f32x4_max(clo, a5lo);
                    a5hi = f32x4_max(chi, a5hi);
                    a6lo = f32x4_max(clo, a6lo);
                    a6hi = f32x4_max(chi, a6hi);
                    a7lo = f32x4_max(clo, a7lo);
                    a7hi = f32x4_max(chi, a7hi);
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_add(clo, a0lo);
                    a0hi = f32x4_add(chi, a0hi);
                    a1lo = f32x4_add(clo, a1lo);
                    a1hi = f32x4_add(chi, a1hi);
                    a2lo = f32x4_add(clo, a2lo);
                    a2hi = f32x4_add(chi, a2hi);
                    a3lo = f32x4_add(clo, a3lo);
                    a3hi = f32x4_add(chi, a3hi);
                    a4lo = f32x4_add(clo, a4lo);
                    a4hi = f32x4_add(chi, a4hi);
                    a5lo = f32x4_add(clo, a5lo);
                    a5hi = f32x4_add(chi, a5hi);
                    a6lo = f32x4_add(clo, a6lo);
                    a6hi = f32x4_add(chi, a6hi);
                    a7lo = f32x4_add(clo, a7lo);
                    a7hi = f32x4_add(chi, a7hi);
                }
                FusedKerSpec::PerColMul(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_mul(clo, a0lo);
                    a0hi = f32x4_mul(chi, a0hi);
                    a1lo = f32x4_mul(clo, a1lo);
                    a1hi = f32x4_mul(chi, a1hi);
                    a2lo = f32x4_mul(clo, a2lo);
                    a2hi = f32x4_mul(chi, a2hi);
                    a3lo = f32x4_mul(clo, a3lo);
                    a3hi = f32x4_mul(chi, a3hi);
                    a4lo = f32x4_mul(clo, a4lo);
                    a4hi = f32x4_mul(chi, a4hi);
                    a5lo = f32x4_mul(clo, a5lo);
                    a5hi = f32x4_mul(chi, a5hi);
                    a6lo = f32x4_mul(clo, a6lo);
                    a6hi = f32x4_mul(chi, a6hi);
                    a7lo = f32x4_mul(clo, a7lo);
                    a7hi = f32x4_mul(chi, a7hi);
                }
                FusedKerSpec::PerColSub(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_sub(clo, a0lo);
                    a0hi = f32x4_sub(chi, a0hi);
                    a1lo = f32x4_sub(clo, a1lo);
                    a1hi = f32x4_sub(chi, a1hi);
                    a2lo = f32x4_sub(clo, a2lo);
                    a2hi = f32x4_sub(chi, a2hi);
                    a3lo = f32x4_sub(clo, a3lo);
                    a3hi = f32x4_sub(chi, a3hi);
                    a4lo = f32x4_sub(clo, a4lo);
                    a4hi = f32x4_sub(chi, a4hi);
                    a5lo = f32x4_sub(clo, a5lo);
                    a5hi = f32x4_sub(chi, a5hi);
                    a6lo = f32x4_sub(clo, a6lo);
                    a6hi = f32x4_sub(chi, a6hi);
                    a7lo = f32x4_sub(clo, a7lo);
                    a7hi = f32x4_sub(chi, a7hi);
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    a0lo = f32x4_sub(a0lo, clo);
                    a0hi = f32x4_sub(a0hi, chi);
                    a1lo = f32x4_sub(a1lo, clo);
                    a1hi = f32x4_sub(a1hi, chi);
                    a2lo = f32x4_sub(a2lo, clo);
                    a2hi = f32x4_sub(a2hi, chi);
                    a3lo = f32x4_sub(a3lo, clo);
                    a3hi = f32x4_sub(a3hi, chi);
                    a4lo = f32x4_sub(a4lo, clo);
                    a4hi = f32x4_sub(a4hi, chi);
                    a5lo = f32x4_sub(a5lo, clo);
                    a5hi = f32x4_sub(a5hi, chi);
                    a6lo = f32x4_sub(a6lo, clo);
                    a6hi = f32x4_sub(a6hi, chi);
                    a7lo = f32x4_sub(a7lo, clo);
                    a7hi = f32x4_sub(a7hi, chi);
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let s = f32x4_splat(scaler.scale);
                    a0lo = f32x4_mul(s, a0lo);
                    a0hi = f32x4_mul(s, a0hi);
                    a1lo = f32x4_mul(s, a1lo);
                    a1hi = f32x4_mul(s, a1hi);
                    a2lo = f32x4_mul(s, a2lo);
                    a2hi = f32x4_mul(s, a2hi);
                    a3lo = f32x4_mul(s, a3lo);
                    a3hi = f32x4_mul(s, a3hi);
                    a4lo = f32x4_mul(s, a4lo);
                    a4hi = f32x4_mul(s, a4hi);
                    a5lo = f32x4_mul(s, a5lo);
                    a5hi = f32x4_mul(s, a5hi);
                    a6lo = f32x4_mul(s, a6lo);
                    a6hi = f32x4_mul(s, a6hi);
                    a7lo = f32x4_mul(s, a7lo);
                    a7hi = f32x4_mul(s, a7hi);
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    a0lo = f32x4_mul(s, a0lo);
                    a0hi = f32x4_mul(s, a0hi);
                    a1lo = f32x4_mul(s, a1lo);
                    a1hi = f32x4_mul(s, a1hi);
                    a2lo = f32x4_mul(s, a2lo);
                    a2hi = f32x4_mul(s, a2hi);
                    a3lo = f32x4_mul(s, a3lo);
                    a3hi = f32x4_mul(s, a3hi);
                    a4lo = f32x4_mul(s, a4lo);
                    a4hi = f32x4_mul(s, a4hi);
                    a5lo = f32x4_mul(s, a5lo);
                    a5hi = f32x4_mul(s, a5hi);
                    a6lo = f32x4_mul(s, a6lo);
                    a6hi = f32x4_mul(s, a6hi);
                    a7lo = f32x4_mul(s, a7lo);
                    a7hi = f32x4_mul(s, a7hi);
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    a0lo = f32x4_mul(s, a0lo);
                    a0hi = f32x4_mul(s, a0hi);
                    a1lo = f32x4_mul(s, a1lo);
                    a1hi = f32x4_mul(s, a1hi);
                    a2lo = f32x4_mul(s, a2lo);
                    a2hi = f32x4_mul(s, a2hi);
                    a3lo = f32x4_mul(s, a3lo);
                    a3hi = f32x4_mul(s, a3hi);
                    a4lo = f32x4_mul(s, a4lo);
                    a4hi = f32x4_mul(s, a4hi);
                    a5lo = f32x4_mul(s, a5lo);
                    a5hi = f32x4_mul(s, a5hi);
                    a6lo = f32x4_mul(s, a6lo);
                    a6hi = f32x4_mul(s, a6hi);
                    a7lo = f32x4_mul(s, a7lo);
                    a7hi = f32x4_mul(s, a7hi);
                }
                FusedKerSpec::AddUnicast(tile) => {
                    // 8 rows × 8 cols, each row laid out per col_byte_stride
                    let mut ptr: *const u8 = tile.ptr;
                    for ab_pair in [
                        (&mut a0lo, &mut a0hi),
                        (&mut a1lo, &mut a1hi),
                        (&mut a2lo, &mut a2hi),
                        (&mut a3lo, &mut a3hi),
                        (&mut a4lo, &mut a4hi),
                        (&mut a5lo, &mut a5hi),
                        (&mut a6lo, &mut a6hi),
                        (&mut a7lo, &mut a7hi),
                    ]
                    .iter_mut()
                    {
                        let m0 = *(ptr as *const f32);
                        let m1 = *(ptr.offset(tile.col_byte_stride) as *const f32);
                        let m2 = *(ptr.offset(tile.col_byte_stride * 2) as *const f32);
                        let m3 = *(ptr.offset(tile.col_byte_stride * 3) as *const f32);
                        let m4 = *(ptr.offset(tile.col_byte_stride * 4) as *const f32);
                        let m5 = *(ptr.offset(tile.col_byte_stride * 5) as *const f32);
                        let m6 = *(ptr.offset(tile.col_byte_stride * 6) as *const f32);
                        let m7 = *(ptr.offset(tile.col_byte_stride * 7) as *const f32);
                        let (lo, hi) = ab_pair;
                        **lo = f32x4_add(**lo, f32x4(m0, m1, m2, m3));
                        **hi = f32x4_add(**hi, f32x4(m4, m5, m6, m7));
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let p = cols as *const v128;
                    let clo = v128_load(p);
                    let chi = v128_load(p.add(1));
                    let r0 = f32x4_splat(*rows.add(0));
                    a0lo = madd_f32x4!(a0lo, r0, clo);
                    a0hi = madd_f32x4!(a0hi, r0, chi);
                    let r1 = f32x4_splat(*rows.add(1));
                    a1lo = madd_f32x4!(a1lo, r1, clo);
                    a1hi = madd_f32x4!(a1hi, r1, chi);
                    let r2 = f32x4_splat(*rows.add(2));
                    a2lo = madd_f32x4!(a2lo, r2, clo);
                    a2hi = madd_f32x4!(a2hi, r2, chi);
                    let r3 = f32x4_splat(*rows.add(3));
                    a3lo = madd_f32x4!(a3lo, r3, clo);
                    a3hi = madd_f32x4!(a3hi, r3, chi);
                    let r4 = f32x4_splat(*rows.add(4));
                    a4lo = madd_f32x4!(a4lo, r4, clo);
                    a4hi = madd_f32x4!(a4hi, r4, chi);
                    let r5 = f32x4_splat(*rows.add(5));
                    a5lo = madd_f32x4!(a5lo, r5, clo);
                    a5hi = madd_f32x4!(a5hi, r5, chi);
                    let r6 = f32x4_splat(*rows.add(6));
                    a6lo = madd_f32x4!(a6lo, r6, clo);
                    a6hi = madd_f32x4!(a6hi, r6, chi);
                    let r7 = f32x4_splat(*rows.add(7));
                    a7lo = madd_f32x4!(a7lo, r7, clo);
                    a7hi = madd_f32x4!(a7hi, r7, chi);
                }
                FusedKerSpec::Store(tile) => {
                    // 8 rows × 8 cols stores
                    let mut ptr: *mut u8 = tile.ptr;
                    for (lo, hi) in [
                        (a0lo, a0hi),
                        (a1lo, a1hi),
                        (a2lo, a2hi),
                        (a3lo, a3hi),
                        (a4lo, a4hi),
                        (a5lo, a5hi),
                        (a6lo, a6hi),
                        (a7lo, a7hi),
                    ]
                    .iter()
                    {
                        *(ptr as *mut f32) = f32x4_extract_lane::<0>(*lo);
                        *(ptr.offset(tile.col_byte_stride) as *mut f32) =
                            f32x4_extract_lane::<1>(*lo);
                        *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) =
                            f32x4_extract_lane::<2>(*lo);
                        *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) =
                            f32x4_extract_lane::<3>(*lo);
                        *(ptr.offset(tile.col_byte_stride * 4) as *mut f32) =
                            f32x4_extract_lane::<0>(*hi);
                        *(ptr.offset(tile.col_byte_stride * 5) as *mut f32) =
                            f32x4_extract_lane::<1>(*hi);
                        *(ptr.offset(tile.col_byte_stride * 6) as *mut f32) =
                            f32x4_extract_lane::<2>(*hi);
                        *(ptr.offset(tile.col_byte_stride * 7) as *mut f32) =
                            f32x4_extract_lane::<3>(*hi);
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A: packed [k][MR=8] = each k iter loads 8 row values
                    // B: packed [k][NR=8] = each k iter loads 8 col values as 2 v128
                    let a = pa as *const f32;
                    let b = pb as *const v128;
                    for i in 0..k {
                        let arow = std::slice::from_raw_parts(a.offset(8 * i as isize), 8);
                        let blo = v128_load(b.offset((2 * i) as isize));
                        let bhi = v128_load(b.offset((2 * i + 1) as isize));
                        let s = f32x4_splat(arow[0]);
                        a0lo = madd_f32x4!(a0lo, s, blo);
                        a0hi = madd_f32x4!(a0hi, s, bhi);
                        let s = f32x4_splat(arow[1]);
                        a1lo = madd_f32x4!(a1lo, s, blo);
                        a1hi = madd_f32x4!(a1hi, s, bhi);
                        let s = f32x4_splat(arow[2]);
                        a2lo = madd_f32x4!(a2lo, s, blo);
                        a2hi = madd_f32x4!(a2hi, s, bhi);
                        let s = f32x4_splat(arow[3]);
                        a3lo = madd_f32x4!(a3lo, s, blo);
                        a3hi = madd_f32x4!(a3hi, s, bhi);
                        let s = f32x4_splat(arow[4]);
                        a4lo = madd_f32x4!(a4lo, s, blo);
                        a4hi = madd_f32x4!(a4hi, s, bhi);
                        let s = f32x4_splat(arow[5]);
                        a5lo = madd_f32x4!(a5lo, s, blo);
                        a5hi = madd_f32x4!(a5hi, s, bhi);
                        let s = f32x4_splat(arow[6]);
                        a6lo = madd_f32x4!(a6lo, s, blo);
                        a6hi = madd_f32x4!(a6hi, s, bhi);
                        let s = f32x4_splat(arow[7]);
                        a7lo = madd_f32x4!(a7lo, s, blo);
                        a7hi = madd_f32x4!(a7hi, s, bhi);
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

// ManuallyOptimized so kernel_selection::strategize honours the mmm_f32
// callback that returns it for N>1 GEMM (see the `plug` comment) — otherwise
// strategize drops it and routes every GEMM onto the 32x1 GEMV kernel.
MMMRustKernel!(kernel_f32_8x8 => wasm_f32_8x8<f32>(8,8)@(8,8) quality(ImplementationQuality::ManuallyOptimized));

// Wasm SIMD int8 -> i32 matmul kernel (4x4). WASM's only integer dot
// (i32x4.relaxed_dot_i8x16_i7x16) is non-deterministic for full i8 (its 2nd
// operand is i7), so for a bit-exact kernel the AddMatMul K-loop uses widening
// i8->i32 + i32x4 mul/add (an extmul/SMLAL-style outer product). The quant
// epilogue + fuse ops reuse the bit-exact scalar path (q_scale/q_shr/q_shl),
// which is O(MR*NR) and negligible vs the O(MR*NR*K) inner loop. Bit-identical
// to generic_i32_4x4; selected for i8 matmul via its ManuallyOptimized quality
// (WASM had no int8 matmul kernel — int8 fell back to the generic scalar one).
#[inline(never)]
unsafe fn kernel_i32_4x4(mut pnl: *const FusedKerSpec<i32>) -> isize {
    use crate::ScaleShiftAndRound;
    use std::arch::wasm32::*;
    unsafe {
        let mut ab = [[0i32; 4]; 4];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = [[0i32; 4]; 4],
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..4 {
                        for col in 0..4 {
                            ab[row][col] = *col_major.add(col * 4 + row);
                        }
                    }
                }
                FusedKerSpec::ScalarAdd(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] += a;
                        }
                    }
                }
                FusedKerSpec::ScalarMul(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] *= a;
                        }
                    }
                }
                FusedKerSpec::ScalarMin(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(m);
                        }
                    }
                }
                FusedKerSpec::ScalarMax(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(m);
                        }
                    }
                }
                FusedKerSpec::ScalarSub(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = m - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::ScalarSubF(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] -= m;
                        }
                    }
                }
                FusedKerSpec::LeakyRelu(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = if ab[i][j] > 0 { ab[i][j] } else { a * ab[i][j] };
                        }
                    }
                }
                FusedKerSpec::PerRowMin(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(v);
                        }
                    }
                }
                FusedKerSpec::PerRowMax(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(v);
                        }
                    }
                }
                FusedKerSpec::PerRowAdd(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] += v;
                        }
                    }
                }
                FusedKerSpec::PerRowMul(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] *= v;
                        }
                    }
                }
                FusedKerSpec::PerRowSub(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = v - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::PerRowSubF(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] -= v;
                        }
                    }
                }
                FusedKerSpec::PerColMin(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(c[j]);
                        }
                    }
                }
                FusedKerSpec::PerColMax(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(c[j]);
                        }
                    }
                }
                FusedKerSpec::PerColAdd(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] += c[j];
                        }
                    }
                }
                FusedKerSpec::PerColMul(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] *= c[j];
                        }
                    }
                }
                FusedKerSpec::PerColSub(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = c[j] - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::PerColSubF(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] -= c[j];
                        }
                    }
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..4 {
                        let r = *rows.add(i);
                        for j in 0..4 {
                            ab[i][j] += r * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            let p = other.ptr.offset(
                                other.row_byte_stride * i as isize
                                    + other.col_byte_stride * j as isize,
                            );
                            let v = match other.item_size {
                                1 => *(p as *const i8) as i32,
                                4 => *(p as *const i32),
                                _ => return 1,
                            };
                            ab[i][j] += v;
                        }
                    }
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_shl(shift);
                        }
                    }
                }
                FusedKerSpec::RoundingShiftRight(shift, rp) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_shr(shift, rp);
                        }
                    }
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let s = Scaler::from_fuse_params(shift, rp, mult);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_scale(s);
                        }
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    if packing == 1 {
                        let a = pa as *const i8;
                        let b = pb as *const i8;
                        let mut acc = [
                            v128_load(ab[0].as_ptr() as *const v128),
                            v128_load(ab[1].as_ptr() as *const v128),
                            v128_load(ab[2].as_ptr() as *const v128),
                            v128_load(ab[3].as_ptr() as *const v128),
                        ];
                        // PackedI8K4 (K=4-inner): per 4-K block, one B v128 load is
                        // shared across the 4 rows; each row broadcasts its 4 K bytes
                        // (a[kb*16 + m*4 ..]) and issues one relaxed_dot (16 MACs).
                        // b[kb*16 + n*4 + kr]. Tail K (k%4) is zero-padded by the packer.
                        #[cfg(target_feature = "relaxed-simd")]
                        for kb in 0..k.div_ceil(4) {
                            let b_all = v128_load(b.add(kb * 16) as *const v128);
                            for (m, acc_m) in acc.iter_mut().enumerate() {
                                let a4 = (a.add(kb * 16 + m * 4) as *const i32).read_unaligned();
                                *acc_m = i32x4_relaxed_dot_i8x16_i7x16_add(
                                    i32x4_splat(a4),
                                    b_all,
                                    *acc_m,
                                );
                            }
                        }
                        // Deterministic fallback (no relaxed-simd): standard PackedFormat
                        // K-major (4 i8 per k), widening outer-product accumulate.
                        #[cfg(not(target_feature = "relaxed-simd"))]
                        for ik in 0..k {
                            let bw = v128_load32_zero(b.add(4 * ik) as *const u32);
                            let bw = i16x8_extend_low_i8x16(bw);
                            let bw = i32x4_extend_low_i16x8(bw);
                            let ar = a.add(4 * ik);
                            acc[0] =
                                i32x4_add(acc[0], i32x4_mul(i32x4_splat(*ar.add(0) as i32), bw));
                            acc[1] =
                                i32x4_add(acc[1], i32x4_mul(i32x4_splat(*ar.add(1) as i32), bw));
                            acc[2] =
                                i32x4_add(acc[2], i32x4_mul(i32x4_splat(*ar.add(2) as i32), bw));
                            acc[3] =
                                i32x4_add(acc[3], i32x4_mul(i32x4_splat(*ar.add(3) as i32), bw));
                        }
                        v128_store(ab[0].as_mut_ptr() as *mut v128, acc[0]);
                        v128_store(ab[1].as_mut_ptr() as *mut v128, acc[1]);
                        v128_store(ab[2].as_mut_ptr() as *mut v128, acc[2]);
                        v128_store(ab[3].as_mut_ptr() as *mut v128, acc[3]);
                    } else if packing == 0 {
                        // i32 x i32, K-major (scalar; rare path).
                        let a = pa as *const i32;
                        let b = pb as *const i32;
                        for ik in 0..k {
                            for i in 0..4 {
                                let av = *a.add(4 * ik + i);
                                for j in 0..4 {
                                    ab[i][j] += av * *b.add(4 * ik + j);
                                }
                            }
                        }
                    } else {
                        return 1;
                    }
                }
                FusedKerSpec::Store(tile) => match tile.item_size {
                    1 => {
                        for i in 0..4 {
                            for j in 0..4 {
                                let loc = tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as *mut u8;
                                *loc = ab[i][j] as u8;
                            }
                        }
                    }
                    4 => {
                        for i in 0..4 {
                            for j in 0..4 {
                                let loc = tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as *mut i32;
                                *loc = ab[i][j];
                            }
                        }
                    }
                    _ => return 1,
                },
            };
            pnl = pnl.add(1);
        }
    }
    0
}

// i8i8 packing for wasm_i32_4x4. Under +relaxed-simd the kernel uses the
// `i32x4_relaxed_dot_i8x16_i7x16_add` SDOT-analog, which wants 4 contiguous K per
// mn-lane → PackedI8K4 (K=4-inner). Without relaxed-simd it uses the widening
// outer-product, which wants K-major → standard PackedFormat. Both are picked at
// compile time so the kernel's AddMatMul and the packer always agree.
#[cfg(target_feature = "relaxed-simd")]
fn wasm_i8_packing() -> impl crate::mmm::MMMInputFormat {
    crate::pack::PackedI8K4::new(4)
}
#[cfg(not(target_feature = "relaxed-simd"))]
fn wasm_i8_packing() -> impl crate::mmm::MMMInputFormat {
    use crate::pack::Packing;
    i8::packing(4)
}

MMMRustKernel!(kernel_i32_4x4 => wasm_i32_4x4<i32>(4,4)
    packing[1] = i8i8 => |k| k.with_packing(wasm_i8_packing(), wasm_i8_packing());
    quality(ImplementationQuality::ManuallyOptimized)
    store(i8)
);

#[cfg(test)]
mod dispatch_trace {
    fn trace_one(label: &str, m: Option<usize>, k: Option<usize>, n: Option<usize>) {
        let mut ops = crate::generic();
        super::plug(&mut ops);
        let mmm = ops.mmm(tract_data::prelude::DatumType::F32, m, k, n).unwrap();
        eprintln!(
            "DFN3 {} (m={:?} k={:?} n={:?}) => {}  [mr={}, nr={}]",
            label,
            m,
            k,
            n,
            mmm.name(),
            mmm.mr(),
            mmm.nr()
        );
    }

    #[test]
    fn dfn3_shapes() {
        // DFN3 N=1 GEMV ops (the dominant matrix-vector cases)
        trace_one("lsnr_fc-style m=1 k=512", Some(1), Some(512), Some(1));
        trace_one("small m=16 k=96", Some(16), Some(96), Some(1));
        trace_one("medium m=32 k=256", Some(32), Some(256), Some(1));
        trace_one("GRU m=256 k=256", Some(256), Some(256), Some(1));
        trace_one("post-rnn m=256 k=512", Some(256), Some(512), Some(1));
        trace_one("frame-encoder m=64 k=96", Some(64), Some(96), Some(1));
        // N>1 sanity: should hit 8x8
        trace_one("MM m=64 k=64 n=8", Some(64), Some(64), Some(8));
    }

    /// Exercise every M-band edge of mmv_f32 to lock in the dispatch.
    /// Lower edge of each band = perfect-tile size; upper edge = last
    /// M before crossover to the next kernel.
    #[test]
    fn band_edges() {
        // 4x1 band: M ∈ 0..=4
        trace_one("band 4x1 lo m=1", Some(1), Some(64), Some(1));
        trace_one("band 4x1 hi m=4", Some(4), Some(64), Some(1));
        // 8x1 band: M ∈ 5..=8
        trace_one("band 8x1 lo m=5", Some(5), Some(64), Some(1));
        trace_one("band 8x1 hi m=8", Some(8), Some(64), Some(1));
        // 16x1 band: M ∈ 9..=16
        trace_one("band 16x1 lo m=9", Some(9), Some(64), Some(1));
        trace_one("band 16x1 hi m=16", Some(16), Some(64), Some(1));
        // 32x1 band: M ≥ 17
        trace_one("band 32x1 lo m=17", Some(17), Some(64), Some(1));
        trace_one("band 32x1 hi m=512", Some(512), Some(64), Some(1));
    }

    /// Regression guard for the GEMM/GEMV dispatch.
    ///
    /// `kernel_selection::strategize` honours the `mmm_f32` / `mmv_f32`
    /// callback only when the returned kernel is `ManuallyOptimized`;
    /// otherwise it falls through to `list_impls`, whose `retain()` drops
    /// every `TargetOptimized` kernel, and for N>1 then picks `max(nr*mr)`
    /// over the surviving `ManuallyOptimized` GEMV kernels — i.e.
    /// `wasm_f32_32x1`, a matrix×vector kernel, for every GEMM. So every
    /// kernel reachable through the dispatch callbacks must be
    /// `ManuallyOptimized`.
    #[test]
    fn dispatch_kernels_are_manually_optimized() {
        use crate::mmm::ImplementationQuality::ManuallyOptimized;
        let mut ops = crate::generic();
        super::plug(&mut ops);
        for (label, m, k, n) in [
            ("GEMM m=64 k=64 n=8", 64, 64, 8),
            ("GEMM m=256 k=256 n=256", 256, 256, 256),
            ("GEMM m=1024 k=576 n=10", 1024, 576, 10),
            ("GEMV m=1 k=512 n=1", 1, 512, 1),
            ("GEMV m=256 k=256 n=1", 256, 256, 1),
        ] {
            let mmm =
                ops.mmm(tract_data::prelude::DatumType::F32, Some(m), Some(k), Some(n)).unwrap();
            assert_eq!(
                mmm.quality(),
                ManuallyOptimized,
                "{label}: dispatch returned {} tagged {:?} — strategize would \
                 discard it and reroute onto a GEMV kernel",
                mmm.name(),
                mmm.quality(),
            );
        }
    }
}

#[cfg(test)]
mod microbench_32x1 {
    //! Quick microbench: time per-call cost for the kernel kit's GEMV path
    //! on DFN3-shaped inputs. Compares 16x1 vs 32x1 head-to-head by
    //! dispatching the named kernel directly.
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_32x1::microbench -- --nocapture --ignored

    use crate::mmm::{AsInputValue, FusedSpec};
    use std::time::Instant;
    use tract_data::internal::*;
    use tract_data::prelude::*;

    fn run_one(kernel: &dyn crate::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        // Pack A (m,k) and B (k,1)
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        // Warmup
        for _ in 0..50 {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        // Timed
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9 // ns/call
    }

    fn pick(name: &str) -> Box<dyn crate::mmm::MatMatMul> {
        let mut ops = crate::generic();
        super::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, iters: usize) {
        let k16 = pick("wasm_f32_16x1");
        let k32 = pick("wasm_f32_32x1");
        let ns16 = run_one(&*k16, m, k, iters);
        let ns32 = run_one(&*k32, m, k, iters);
        let calls16 = m.div_ceil(16);
        let calls32 = m.div_ceil(32);
        let delta = (ns32 - ns16) / ns16 * 100.0;
        eprintln!(
            "{label} (m={m}, k={k}, iters={iters}): 16x1={ns16:.1} ns/call ({calls16} kernel calls); 32x1={ns32:.1} ns/call ({calls32} kernel calls); Δ={delta:+.2}% ; per-frame call ns: 16x1={n16:.1} 32x1={n32:.1} pf-Δ={dpf:+.2}%",
            n16 = ns16 * calls16 as f64,
            n32 = ns32 * calls32 as f64,
            dpf = (ns32 * calls32 as f64 - ns16 * calls16 as f64) / (ns16 * calls16 as f64) * 100.0,
        );
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== DFN3 GEMV microbench: 16x1 vs 32x1 ===");
        // DFN3 GRU gates (highest call count)
        bench_shape("GRU m=256 k=256", 256, 256, 5_000);
        // post-RNN
        bench_shape("post-rnn m=256 k=512", 256, 512, 3_000);
        // frame encoder
        bench_shape("frame-encoder m=64 k=96", 64, 96, 20_000);
        // perfect tile
        bench_shape("perfect-tile m=32 k=256", 32, 256, 20_000);
    }

    /// Numerical-equivalence sanity check between 16x1 and 32x1 kernels on a
    /// real-shape matmul with non-trivial inputs.
    ///
    /// Under `+simd128` (no relaxed-simd): both kernels emit
    /// `f32x4_add(f32x4_mul(...))` via `madd_f32x4!`, so the K-loop order is
    /// identical and outputs are bit-identical.
    ///
    /// Under `+simd128,+relaxed-simd`: 32x1 uses `f32x4.relaxed_madd` (fused
    /// FMA) via `madd_f32x4!`, while 16x1 uses separate `mul+add` via
    /// `madd_f32x4_nofma!` to avoid the destructive-accumulator recurrence
    /// that throttles ≤4-accumulator kernels (see header comment on
    /// `madd_f32x4_nofma`). Outputs drift by ≤1 ulp per K-step from the
    /// rounding difference between fused and separate ops. We accept that
    /// drift with a generous relative tolerance.
    #[test]
    fn numerical_consistency_16x1_vs_32x1() {
        let m = 256usize;
        let k = 256usize;
        let mut a_data = vec![0f32; m * k];
        for (i, x) in a_data.iter_mut().enumerate() {
            *x = ((i % 13) as f32 - 6.0) * 0.1 + ((i / 17) % 11) as f32 * 0.07;
        }
        let mut b_data = vec![0f32; k];
        for (i, x) in b_data.iter_mut().enumerate() {
            *x = (i as f32).sin() * 0.5;
        }
        let a = Tensor::from_shape(&[m, k], &a_data).unwrap();
        let b = Tensor::from_shape(&[k, 1], &b_data).unwrap();

        let run = |name: &str| -> Vec<f32> {
            let kernel = pick(name);
            let packing = &kernel.packings()[0];
            let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
            let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
            let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
            c.try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec()
        };

        let c16 = run("wasm_f32_16x1");
        let c32 = run("wasm_f32_32x1");

        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            for (i, (x16, x32)) in c16.iter().zip(c32.iter()).enumerate() {
                assert!(
                    x16.to_bits() == x32.to_bits(),
                    "row {i}: 16x1={x16} (bits 0x{:x}) != 32x1={x32} (bits 0x{:x})",
                    x16.to_bits(),
                    x32.to_bits()
                );
            }
            eprintln!("bit-identity OK over m={m} k={k} ({} rows)", m);
        }

        #[cfg(target_feature = "relaxed-simd")]
        {
            // K=256 accumulator drift on fp32 between FMA and separate mul+add
            // can grow up to roughly K × 0.5 ulp ≈ 128 ulp in the accumulator.
            // For small-magnitude outputs that translates to ~1e-4 relative.
            // We use 1e-4 as the tolerance — tight enough to catch real bugs
            // (typically 1e-2+ drift) but generous for legitimate FMA drift.
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (i, (x16, x32)) in c16.iter().zip(c32.iter()).enumerate() {
                let abs = (x16 - x32).abs();
                let scale = x16.abs().max(x32.abs()).max(1.0e-9);
                let rel = abs / scale;
                assert!(
                    rel < 1.0e-4,
                    "row {i}: relative drift {rel:e} too large; 16x1={x16} 32x1={x32}"
                );
                if abs > max_abs {
                    max_abs = abs;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!(
                "relaxed-simd consistency OK over m={m} k={k}: max abs={max_abs:.3e}, max rel={max_rel:.3e}"
            );
        }
    }
}

#[cfg(test)]
mod microbench_dispatch_gemv {
    //! Microbench: 4x1 vs 8x1 vs 16x1 vs 32x1 GEMV kernels across the M
    //! range. Drives the dispatch-fix decision — the M-band callback in
    //! plug() routes small-M to smaller kernels, but only takes effect
    //! once the kernels are tagged ManuallyOptimized (otherwise
    //! kernel_selection::strategize bypasses the callback and always
    //! picks max(mr) = 32x1).
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_dispatch_gemv::microbench -- --nocapture --ignored

    use crate::mmm::{AsInputValue, FusedSpec};
    use std::time::Instant;
    use tract_data::internal::*;
    use tract_data::prelude::*;

    fn run_one(kernel: &dyn crate::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        for _ in 0..50 {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn crate::mmm::MatMatMul> {
        let mut ops = crate::generic();
        super::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, iters: usize) {
        let k4 = pick("wasm_f32_4x1");
        let k8 = pick("wasm_f32_8x1");
        let k16 = pick("wasm_f32_16x1");
        let k32 = pick("wasm_f32_32x1");
        let n4 = run_one(&*k4, m, k, iters);
        let n8 = run_one(&*k8, m, k, iters);
        let n16 = run_one(&*k16, m, k, iters);
        let n32 = run_one(&*k32, m, k, iters);
        let entries = [("4x1", n4), ("8x1", n8), ("16x1", n16), ("32x1", n32)];
        let winner = entries.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let delta_vs_32 = (winner.1 - n32) / n32 * 100.0;
        eprintln!(
            "{label} (m={m} k={k}): 4x1={n4:.0} 8x1={n8:.0} 16x1={n16:.0} 32x1={n32:.0} ns; \
             winner={} ({:.0} ns, Δ vs 32x1: {delta_vs_32:+.1}%)",
            winner.0, winner.1
        );
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== WASM GEMV dispatch microbench: 4x1 vs 8x1 vs 16x1 vs 32x1 ===");
        // M ≤ 16 — small-M region; the M-band callback's choices win clearly.
        bench_shape("M=1   k=512", 1, 512, 50_000);
        bench_shape("M=8   k=64 ", 8, 64, 50_000);
        bench_shape("M=8   k=512", 8, 512, 20_000);
        bench_shape("M=12  k=256", 12, 256, 50_000);
        bench_shape("M=16  k=96 ", 16, 96, 50_000);
        bench_shape("M=16  k=256", 16, 256, 30_000);
        // M ≥ 17 — 32x1 wins (16x1 needs 2 outer iters, 32x1 single iter
        // with ILP absorbing the row padding).
        bench_shape("M=24  k=256", 24, 256, 30_000);
        bench_shape("M=32  k=256", 32, 256, 20_000);
        bench_shape("M=64  k=96 ", 64, 96, 20_000);
        bench_shape("M=100 k=256", 100, 256, 10_000);
        bench_shape("M=256 k=256", 256, 256, 5_000);
    }
}

// Relaxed-SIMD activation kernels (f32, FMA path).
//
// `f32x4_relaxed_madd(a, b, c)` computes `a * b + c`. On hosts with hardware
// FMA (all ARM64, x86_64 with FMA3) it lowers to a single fused, single-
// rounded instruction. On hosts without, it falls back to mul+add — hence
// "relaxed". The result is therefore not bit-deterministic across all hosts,
// but it is at least as accurate as the separate mul+add (FMA does fewer
// roundings).
//
// For sigmoid/tanh polynomial evaluation, the 14 muladds in the Horner chain
// fuse cleanly. Measured ~1.65x over the baseline-simd128 explicit kernel and
// over LLVM auto-vec'd scalar on V8.
//
// Gated on `target_feature = "relaxed-simd"` because `f32x4_relaxed_madd`
// requires the relaxed-simd proposal to be enabled at compile time.
// ---------------------------------------------------------------------------

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmSigmoid4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmSigmoid4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        // Coefficients match generic/sigmoid.rs::ssigmoid bit-for-bit.
        // Output may differ by ≤1 ulp from scalar on FMA hosts (more accurate).
        const LOW: f32 = -18.6;
        const HIGH: f32 = -LOW;

        const ALPHA_13: f32 = -4.433153405e-18;
        const ALPHA_11: f32 = 1.169974371e-14;
        const ALPHA_9: f32 = -1.875289645e-11;
        const ALPHA_7: f32 = 4.257889523e-8;
        const ALPHA_5: f32 = 0.00004811817576;
        const ALPHA_3: f32 = 0.008163842030;
        const ALPHA_1: f32 = 0.2499999971;

        const BETA_6: f32 = 3.922935744e-6;
        const BETA_4: f32 = 0.001524872358;
        const BETA_2: f32 = 0.1159886749;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let half = f32x4_splat(0.5);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                // Horner numerator with FMA: pn = x2 * pn + a_n
                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                // Horner denominator with FMA
                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let r = f32x4_add(f32x4_div(pn, qn), half);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmTanh4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmTanh4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const LOW: f32 = -8.9;
        const HIGH: f32 = 8.9;

        const ALPHA_13: f32 = -8.488492677e-14;
        const ALPHA_11: f32 = 5.277853000e-11;
        const ALPHA_9: f32 = -2.022500419e-8;
        const ALPHA_7: f32 = 0.00001115424833;
        const ALPHA_5: f32 = 0.003103950131;
        const ALPHA_3: f32 = 0.1308400453;
        const ALPHA_1: f32 = 0.9999999934;

        const BETA_6: f32 = 0.0002546136580;
        const BETA_4: f32 = 0.02449515379;
        const BETA_2: f32 = 0.4641733162;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let r = f32x4_div(pn, qn);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_sigmoid_relaxed {
    sigmoid_frame_tests!(true, f32, crate::wasm::WasmSigmoid4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_tanh_relaxed {
    tanh_frame_tests!(true, f32, crate::wasm::WasmTanh4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
mod microbench_activations {
    //! Microbench: WASM SIMD sigmoid/tanh vs the generic scalar fallback.
    //! Sizes mirror typical RNN/transformer hidden dims (256, 512, 1024).
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_activations::microbench -- --nocapture --ignored
    use crate::frame::element_wise::ElementWiseKer;
    use std::time::Instant;

    fn ns_per_call<K: ElementWiseKer<f32>>(buf: &mut [f32], iters: usize) -> f64 {
        // Warmup
        for _ in 0..50 {
            K::run(buf, ());
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            K::run(buf, ());
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn bench(label: &str, n: usize, iters: usize) {
        // Same input for both kernels — rebuild between to avoid post-clamp
        // saturation mucking up the measurement.
        let make = || (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.5).collect::<Vec<f32>>();

        let mut buf = make();
        let scalar_sig = ns_per_call::<crate::generic::sigmoid::SSigmoid4>(&mut buf, iters);
        let mut buf = make();
        let simd_sig = ns_per_call::<crate::wasm::WasmSigmoid4Relaxed>(&mut buf, iters);
        let mut buf = make();
        let scalar_tanh = ns_per_call::<crate::generic::tanh::STanh4>(&mut buf, iters);
        let mut buf = make();
        let simd_tanh = ns_per_call::<crate::wasm::WasmTanh4Relaxed>(&mut buf, iters);

        eprintln!(
            "{label} n={n} iters={iters}: \
             sigmoid scalar={scalar_sig:.0} ns simd={simd_sig:.0} ns ({:.2}x); \
             tanh scalar={scalar_tanh:.0} ns simd={simd_tanh:.0} ns ({:.2}x)",
            scalar_sig / simd_sig,
            scalar_tanh / simd_tanh,
        );
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== WASM SIMD activations: scalar vs simd ===");
        bench("hidden=256", 256, 5_000);
        bench("hidden=512", 512, 3_000);
        bench("hidden=1024", 1024, 2_000);
    }
}
