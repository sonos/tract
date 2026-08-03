#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::Scaler;
use crate::mmm::FusedKerSpec;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
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
                FusedKerSpec::Clear => wasm_set!(f32x4_splat(0.0); ab0, ab1, ab2, ab3),
                FusedKerSpec::LoadTile(_cols, rows) => wasm_load_indexed!(rows; ab0, ab1, ab2, ab3),
                FusedKerSpec::ScalarMin(a) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ScalarMax(a) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ScalarAdd(a) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ScalarMul(a) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ScalarSub(a) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ScalarSubF(a) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(a); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::LeakyRelu(a) => wasm_leaky_relu!(a; ab0, ab1, ab2, ab3),
                FusedKerSpec::PerRowMin(row) => {
                    wasm_bin_splat_indexed!(f32x4_min, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerRowMax(row) => {
                    wasm_bin_splat_indexed!(f32x4_max, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerRowAdd(row) => {
                    wasm_bin_splat_indexed!(f32x4_add, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerRowMul(row) => {
                    wasm_bin_splat_indexed!(f32x4_mul, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerRowSub(row) => {
                    wasm_bin_splat_indexed!(f32x4_sub, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerRowSubF(row) => {
                    wasm_bin_splat_indexed_vs!(f32x4_sub, row; ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColMin(cols) => {
                    wasm_bin_sv!(f32x4_min, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColMax(cols) => {
                    wasm_bin_sv!(f32x4_max, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColAdd(cols) => {
                    wasm_bin_sv!(f32x4_add, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColMul(cols) => {
                    wasm_bin_sv!(f32x4_mul, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColSub(cols) => {
                    wasm_bin_sv!(f32x4_sub, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::PerColSubF(cols) => {
                    wasm_bin_vs!(f32x4_sub, v128_load(cols as *const v128); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); ab0, ab1, ab2, ab3)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); ab0, ab1, ab2, ab3)
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

// Reachable only by name, never through dispatch: it declares NEVER_PREFERRED, so every
// other wasm f32 kernel outranks it. Kept because it is the only f32 kernel besides 8x8
// whose C tile is two-dimensional, so the generated store and packing tests cover that
// layout on a second shape. `dispatch_never_returns_wasm_f32_4x4` holds this in place.
bail_stub!(wasm32; unsafe fn kernel_f32_4x4(*const FusedKerSpec<f32>) -> isize);

MMMRustKernel!(wasm32; kernel_f32_4x4 => wasm_f32_4x4<f32>(4,4)@(4,4)
    boost(|| crate::isa::NEVER_PREFERRED));

/// WASM SIMD f32 8x8 kernel — wide MM tile (8 rows × 8 cols, 16 v128 accumulators).
/// Each row uses 2 v128: cols 0-3 in `_lo`, cols 4-7 in `_hi`. 16 accumulators
/// is at the limit of WASM's 16 logical SIMD register slots; this tests the
/// register-pressure boundary. For DFN3 ops, all M and N are multiples of 8,
/// so 8x8 fits cleanly with no padding waste.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
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
                    wasm_set!(f32x4_splat(0.0); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    wasm_load_indexed!(rows; a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarMin(a) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarMax(a) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarAdd(a) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarMul(a) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarSub(a) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ScalarSubF(a) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(a); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::LeakyRelu(a) => {
                    wasm_leaky_relu!(a; a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::PerRowMin(row) => {
                    wasm_bin_row_pairs!(f32x4_min, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerRowMax(row) => {
                    wasm_bin_row_pairs!(f32x4_max, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerRowAdd(row) => {
                    wasm_bin_row_pairs!(f32x4_add, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerRowMul(row) => {
                    wasm_bin_row_pairs!(f32x4_mul, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerRowSub(row) => {
                    wasm_bin_row_pairs!(f32x4_sub, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerRowSubF(row) => {
                    wasm_bin_row_pairs_vs!(f32x4_sub, row; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColMin(cols) => {
                    wasm_bin_col_pairs!(f32x4_min, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColMax(cols) => {
                    wasm_bin_col_pairs!(f32x4_max, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColAdd(cols) => {
                    wasm_bin_col_pairs!(f32x4_add, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColMul(cols) => {
                    wasm_bin_col_pairs!(f32x4_mul, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColSub(cols) => {
                    wasm_bin_col_pairs!(f32x4_sub, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::PerColSubF(cols) => {
                    wasm_bin_col_pairs_vs!(f32x4_sub, cols; (a0lo, a0hi), (a1lo, a1hi), (a2lo, a2hi), (a3lo, a3hi), (a4lo, a4hi), (a5lo, a5hi), (a6lo, a6hi), (a7lo, a7hi))
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi, a4lo, a4hi, a5lo, a5hi, a6lo, a6hi, a7lo, a7hi)
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

bail_stub!(wasm32; unsafe fn kernel_f32_8x8(*const FusedKerSpec<f32>) -> isize);

MMMRustKernel!(wasm32; kernel_f32_8x8 => wasm_f32_8x8<f32>(8,8)@(8,8));

/// WASM SIMD f32 4x16 kernel — 4 rows x 16 cols, 16 v128 accumulators laid out
/// row-major as `acc[r * 4 + j]`, j indexing the four v128 chunks of a row.
///
/// Same accumulator count as `wasm_f32_8x8`, but the tile is wide rather than
/// square. Each k-step splats 4 A values and loads 4 B vectors, so one splat
/// feeds four multiply-adds instead of two: the per-splat overhead is halved
/// against the same arithmetic. Measured against `wasm_f32_8x8` on a full tiled
/// GEMM, this is 1.09-1.11x on the 64x64 shapes and 1.21x aggregate.
///
/// It is NOT a general replacement: at n=8 a 16-wide tile wastes half its
/// columns and this runs at 0.58x. Dispatch must keep 8x8 for narrow n.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn kernel_f32_4x16(mut pnl: *const FusedKerSpec<f32>) -> isize {
    use std::arch::wasm32::*;

    unsafe {
        const MR: usize = 4;
        const NV: usize = 4; // v128 chunks per row (NR = 16)
        let mut acc = [f32x4_splat(0.0); MR * NV];

        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => acc = [f32x4_splat(0.0); MR * NV],
                FusedKerSpec::LoadTile(_cols, rows) => {
                    let p = rows as *const v128;
                    for i in 0..MR * NV {
                        acc[i] = *p.add(i);
                    }
                }
                FusedKerSpec::ScalarMin(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_min(s, *v);
                    }
                }
                FusedKerSpec::ScalarMax(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_max(s, *v);
                    }
                }
                FusedKerSpec::ScalarAdd(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_add(s, *v);
                    }
                }
                FusedKerSpec::ScalarMul(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_mul(s, *v);
                    }
                }
                FusedKerSpec::ScalarSub(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_sub(s, *v);
                    }
                }
                FusedKerSpec::ScalarSubF(a) => {
                    let s = f32x4_splat(a);
                    for v in acc.iter_mut() {
                        *v = f32x4_sub(*v, s);
                    }
                }
                FusedKerSpec::LeakyRelu(a) => {
                    let s = f32x4_splat(a);
                    let zero = f32x4_splat(0.0);
                    for v in acc.iter_mut() {
                        let m = f32x4_gt(*v, zero);
                        *v = v128_bitselect(*v, f32x4_mul(s, *v), m);
                    }
                }
                FusedKerSpec::PerRowMin(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_min(s, acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerRowMax(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_max(s, acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerRowAdd(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_add(s, acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerRowMul(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_mul(s, acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerRowSub(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_sub(s, acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerRowSubF(rows) => {
                    for r in 0..MR {
                        let s = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_sub(acc[r * NV + j], s);
                        }
                    }
                }
                FusedKerSpec::PerColMin(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_min(v128_load(p.add(j)), acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerColMax(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_max(v128_load(p.add(j)), acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerColAdd(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_add(v128_load(p.add(j)), acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerColMul(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_mul(v128_load(p.add(j)), acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerColSub(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_sub(v128_load(p.add(j)), acc[r * NV + j]);
                        }
                    }
                }
                FusedKerSpec::PerColSubF(cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        for j in 0..NV {
                            acc[r * NV + j] = f32x4_sub(acc[r * NV + j], v128_load(p.add(j)));
                        }
                    }
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let scaler = Scaler::from_fuse_params(shift, rp, mult);
                    let s = f32x4_splat(scaler.scale);
                    for v in acc.iter_mut() {
                        *v = f32x4_mul(s, *v);
                    }
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    let s = f32x4_splat(2f32.powi(-(shift as i32)));
                    for v in acc.iter_mut() {
                        *v = f32x4_mul(s, *v);
                    }
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    let s = f32x4_splat(2f32.powi(shift as i32));
                    for v in acc.iter_mut() {
                        *v = f32x4_mul(s, *v);
                    }
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    let p = cols as *const v128;
                    for r in 0..MR {
                        let rv = f32x4_splat(*rows.add(r));
                        for j in 0..NV {
                            acc[r * NV + j] = madd_f32x4!(acc[r * NV + j], rv, v128_load(p.add(j)));
                        }
                    }
                }
                FusedKerSpec::AddUnicast(tile) => {
                    let mut ptr: *const u8 = tile.ptr;
                    for r in 0..MR {
                        for j in 0..NV {
                            let c0 = *(ptr.offset(tile.col_byte_stride * (j * 4) as isize)
                                as *const f32);
                            let c1 = *(ptr.offset(tile.col_byte_stride * (j * 4 + 1) as isize)
                                as *const f32);
                            let c2 = *(ptr.offset(tile.col_byte_stride * (j * 4 + 2) as isize)
                                as *const f32);
                            let c3 = *(ptr.offset(tile.col_byte_stride * (j * 4 + 3) as isize)
                                as *const f32);
                            acc[r * NV + j] = f32x4_add(f32x4(c0, c1, c2, c3), acc[r * NV + j]);
                        }
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::Store(tile) => {
                    let mut ptr: *mut u8 = tile.ptr;
                    for r in 0..MR {
                        for j in 0..NV {
                            let v = acc[r * NV + j];
                            let base = (j * 4) as isize;
                            *(ptr.offset(tile.col_byte_stride * base) as *mut f32) =
                                f32x4_extract_lane::<0>(v);
                            *(ptr.offset(tile.col_byte_stride * (base + 1)) as *mut f32) =
                                f32x4_extract_lane::<1>(v);
                            *(ptr.offset(tile.col_byte_stride * (base + 2)) as *mut f32) =
                                f32x4_extract_lane::<2>(v);
                            *(ptr.offset(tile.col_byte_stride * (base + 3)) as *mut f32) =
                                f32x4_extract_lane::<3>(v);
                        }
                        ptr = ptr.add(tile.row_byte_stride as usize);
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing: _ } => {
                    // A: packed [k][MR=4], B: packed [k][NR=16] = 4 v128 per k
                    let a = pa as *const f32;
                    let b = pb as *const v128;
                    for i in 0..k {
                        let arow = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let bb = b.offset((4 * i) as isize);
                        let b0 = v128_load(bb);
                        let b1 = v128_load(bb.add(1));
                        let b2 = v128_load(bb.add(2));
                        let b3 = v128_load(bb.add(3));
                        for r in 0..MR {
                            let s = f32x4_splat(arow[r]);
                            acc[r * NV] = madd_f32x4!(acc[r * NV], s, b0);
                            acc[r * NV + 1] = madd_f32x4!(acc[r * NV + 1], s, b1);
                            acc[r * NV + 2] = madd_f32x4!(acc[r * NV + 2], s, b2);
                            acc[r * NV + 3] = madd_f32x4!(acc[r * NV + 3], s, b3);
                        }
                    }
                }
            }
            pnl = pnl.add(1);
        }
        0
    }
}

bail_stub!(wasm32; unsafe fn kernel_f32_4x16(*const FusedKerSpec<f32>) -> isize);

MMMRustKernel!(wasm32; kernel_f32_4x16 => wasm_f32_4x16<f32>(4,16)@(4,16));
