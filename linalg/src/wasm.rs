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

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(wasm_f32_4x4.mmm());
    ops.mmm_f32 = Box::new(|_m, _k, _n| wasm_f32_4x4.mmm());
}

unsafe fn kernel_f32_4x4(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

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
                ab0 = f32x4_add(ab0, f32x4_mul(f32x4_splat(*rows.add(0)), cols));
                ab1 = f32x4_add(ab1, f32x4_mul(f32x4_splat(*rows.add(1)), cols));
                ab2 = f32x4_add(ab2, f32x4_mul(f32x4_splat(*rows.add(2)), cols));
                ab3 = f32x4_add(ab3, f32x4_mul(f32x4_splat(*rows.add(3)), cols));
            }
            FusedKerSpec::Store(tile) => {
                let mut ptr: *mut u8 = tile.ptr;

                *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab0);
                *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab0);
                *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) = f32x4_extract_lane::<2>(ab0);
                *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) = f32x4_extract_lane::<3>(ab0);
                ptr = ptr.add(tile.row_byte_stride as usize);

                *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab1);
                *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab1);
                *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) = f32x4_extract_lane::<2>(ab1);
                *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) = f32x4_extract_lane::<3>(ab1);
                ptr = ptr.add(tile.row_byte_stride as usize);

                *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab2);
                *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab2);
                *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) = f32x4_extract_lane::<2>(ab2);
                *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) = f32x4_extract_lane::<3>(ab2);
                ptr = ptr.add(tile.row_byte_stride as usize);

                *(ptr as *mut f32) = f32x4_extract_lane::<0>(ab3);
                *(ptr.offset(tile.col_byte_stride) as *mut f32) = f32x4_extract_lane::<1>(ab3);
                *(ptr.offset(tile.col_byte_stride * 2) as *mut f32) = f32x4_extract_lane::<2>(ab3);
                *(ptr.offset(tile.col_byte_stride * 3) as *mut f32) = f32x4_extract_lane::<3>(ab3);
            }
            FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                let a = pa as *const f32;
                let b = pb as *const v128;
                for i in 0..k {
                    let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                    let b = v128_load(b.offset(i as isize));
                    ab0 = f32x4_add(ab0, f32x4_mul(f32x4_splat(a[0]), b));
                    ab1 = f32x4_add(ab1, f32x4_mul(f32x4_splat(a[1]), b));
                    ab2 = f32x4_add(ab2, f32x4_mul(f32x4_splat(a[2]), b));
                    ab3 = f32x4_add(ab3, f32x4_mul(f32x4_splat(a[3]), b));
                }
            }
        }
        pnl = pnl.add(1);
    }
    0
}

MMMRustKernel!(kernel_f32_4x4 => wasm_f32_4x4<f32>(4,4)@(4,4) quality(ImplementationQuality::TargetOptimized));
