use crate::Scaler;
use crate::mmm::FusedKerSpec;
use crate::mmm::ImplementationQuality;

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
                FusedKerSpec::Clear => wasm_set!(f32x4_splat(0.0); ab),
                FusedKerSpec::LoadTile(_cols, rows) => ab = v128_load(rows as *const v128),
                FusedKerSpec::ScalarMin(a) => wasm_bin_sv!(f32x4_min, f32x4_splat(a); ab),
                FusedKerSpec::ScalarMax(a) => wasm_bin_sv!(f32x4_max, f32x4_splat(a); ab),
                FusedKerSpec::ScalarAdd(a) => wasm_bin_sv!(f32x4_add, f32x4_splat(a); ab),
                FusedKerSpec::ScalarMul(a) => wasm_bin_sv!(f32x4_mul, f32x4_splat(a); ab),
                FusedKerSpec::ScalarSub(a) => wasm_bin_sv!(f32x4_sub, f32x4_splat(a); ab),
                FusedKerSpec::ScalarSubF(a) => wasm_bin_vs!(f32x4_sub, f32x4_splat(a); ab),
                FusedKerSpec::LeakyRelu(a) => wasm_leaky_relu!(a; ab),
                FusedKerSpec::PerRowMin(row) => wasm_bin_load_indexed!(f32x4_min, row; ab),
                FusedKerSpec::PerRowMax(row) => wasm_bin_load_indexed!(f32x4_max, row; ab),
                FusedKerSpec::PerRowAdd(row) => wasm_bin_load_indexed!(f32x4_add, row; ab),
                FusedKerSpec::PerRowMul(row) => wasm_bin_load_indexed!(f32x4_mul, row; ab),
                FusedKerSpec::PerRowSub(row) => wasm_bin_load_indexed!(f32x4_sub, row; ab),
                FusedKerSpec::PerRowSubF(row) => wasm_bin_load_indexed_vs!(f32x4_sub, row; ab),
                FusedKerSpec::PerColMin(cols) => wasm_bin_sv!(f32x4_min, f32x4_splat(*cols); ab),
                FusedKerSpec::PerColMax(cols) => wasm_bin_sv!(f32x4_max, f32x4_splat(*cols); ab),
                FusedKerSpec::PerColAdd(cols) => wasm_bin_sv!(f32x4_add, f32x4_splat(*cols); ab),
                FusedKerSpec::PerColMul(cols) => wasm_bin_sv!(f32x4_mul, f32x4_splat(*cols); ab),
                FusedKerSpec::PerColSub(cols) => wasm_bin_sv!(f32x4_sub, f32x4_splat(*cols); ab),
                FusedKerSpec::PerColSubF(cols) => wasm_bin_vs!(f32x4_sub, f32x4_splat(*cols); ab),
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); ab)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); ab)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); ab)
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
                FusedKerSpec::Clear => wasm_set!(f32x4_splat(0.0); ab_top, ab_bot),
                FusedKerSpec::LoadTile(_cols, rows) => wasm_load_indexed!(rows; ab_top, ab_bot),
                FusedKerSpec::ScalarMin(a) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::ScalarMax(a) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::ScalarAdd(a) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::ScalarMul(a) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::ScalarSub(a) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::ScalarSubF(a) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(a); ab_top, ab_bot)
                }
                FusedKerSpec::LeakyRelu(a) => wasm_leaky_relu!(a; ab_top, ab_bot),
                FusedKerSpec::PerRowMin(row) => {
                    wasm_bin_load_indexed!(f32x4_min, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerRowMax(row) => {
                    wasm_bin_load_indexed!(f32x4_max, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerRowAdd(row) => {
                    wasm_bin_load_indexed!(f32x4_add, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerRowMul(row) => {
                    wasm_bin_load_indexed!(f32x4_mul, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerRowSub(row) => {
                    wasm_bin_load_indexed!(f32x4_sub, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerRowSubF(row) => {
                    wasm_bin_load_indexed_vs!(f32x4_sub, row; ab_top, ab_bot)
                }
                FusedKerSpec::PerColMin(cols) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::PerColMax(cols) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::PerColAdd(cols) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::PerColMul(cols) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::PerColSub(cols) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::PerColSubF(cols) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(*cols); ab_top, ab_bot)
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); ab_top, ab_bot)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); ab_top, ab_bot)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); ab_top, ab_bot)
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
                FusedKerSpec::Clear => wasm_set!(f32x4_splat(0.0); ab_q0, ab_q1, ab_q2, ab_q3),
                FusedKerSpec::LoadTile(_cols, rows) => {
                    wasm_load_indexed!(rows; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarMin(a) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarMax(a) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarAdd(a) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarMul(a) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarSub(a) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ScalarSubF(a) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::LeakyRelu(a) => wasm_leaky_relu!(a; ab_q0, ab_q1, ab_q2, ab_q3),
                FusedKerSpec::PerRowMin(row) => {
                    wasm_bin_load_indexed!(f32x4_min, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerRowMax(row) => {
                    wasm_bin_load_indexed!(f32x4_max, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerRowAdd(row) => {
                    wasm_bin_load_indexed!(f32x4_add, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerRowMul(row) => {
                    wasm_bin_load_indexed!(f32x4_mul, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerRowSub(row) => {
                    wasm_bin_load_indexed!(f32x4_sub, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerRowSubF(row) => {
                    wasm_bin_load_indexed_vs!(f32x4_sub, row; ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColMin(cols) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColMax(cols) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColAdd(cols) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColMul(cols) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColSub(cols) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::PerColSubF(cols) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); ab_q0, ab_q1, ab_q2, ab_q3)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); ab_q0, ab_q1, ab_q2, ab_q3)
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
                    wasm_set!(f32x4_splat(0.0); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::LoadTile(_cols, rows) => {
                    wasm_load_indexed!(rows; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarMin(a) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarMax(a) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarAdd(a) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarMul(a) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarSub(a) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ScalarSubF(a) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(a); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::LeakyRelu(a) => {
                    wasm_leaky_relu!(a; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowMin(row) => {
                    wasm_bin_load_indexed!(f32x4_min, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowMax(row) => {
                    wasm_bin_load_indexed!(f32x4_max, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowAdd(row) => {
                    wasm_bin_load_indexed!(f32x4_add, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowMul(row) => {
                    wasm_bin_load_indexed!(f32x4_mul, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowSub(row) => {
                    wasm_bin_load_indexed!(f32x4_sub, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerRowSubF(row) => {
                    wasm_bin_load_indexed_vs!(f32x4_sub, row; ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColMin(cols) => {
                    wasm_bin_sv!(f32x4_min, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColMax(cols) => {
                    wasm_bin_sv!(f32x4_max, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColAdd(cols) => {
                    wasm_bin_sv!(f32x4_add, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColMul(cols) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColSub(cols) => {
                    wasm_bin_sv!(f32x4_sub, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::PerColSubF(cols) => {
                    wasm_bin_vs!(f32x4_sub, f32x4_splat(*cols); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(Scaler::from_fuse_params(shift, rp, mult).scale); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::RoundingShiftRight(shift, _rp) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(-(shift as i32))); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    wasm_bin_sv!(f32x4_mul, f32x4_splat(2f32.powi(shift as i32)); ab_q0, ab_q1, ab_q2, ab_q3, ab_q4, ab_q5, ab_q6, ab_q7)
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
