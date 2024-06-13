#![allow(clippy::needless_range_loop)]
use num_traits::AsPrimitive;

use tract_data::prelude::*;

use super::*;
use crate::frame::mmm::*;
use crate::LADatum;

macro_rules! scalar {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f($m, $ab[i][j])
            }
        }
    };
}

macro_rules! per_row {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f(*$m.add(i), $ab[i][j])
            }
        }
    };
}

macro_rules! per_col {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f(*$m.add(j), $ab[i][j])
            }
        }
    };
}

unsafe fn add_mat_mul<const MR: usize, const NR: usize, TI, TA, TB>(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[TI; NR]; MR],
) where
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TI: LADatum,
{
    let a = pa as *const TA;
    let b = pb as *const TB;
    for ik in 0..k {
        let a = std::slice::from_raw_parts(a.add(MR * ik), MR);
        let b = std::slice::from_raw_parts(b.add(NR * ik), NR);
        for i in 0..MR {
            for j in 0..NR {
                ab[i][j] += a[i].as_() * b[j].as_();
            }
        }
    }
}

unsafe fn store_t<const MR: usize, const NR: usize, TC, TI>(
    tile: &OutputStoreKer,
    ab: &[[TI; NR]; MR],
) where
    TC: Copy,
{
    for i in 0usize..MR {
        for j in 0usize..NR {
            let loc: *mut TC = tile
                .ptr
                .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                as _;
            let val: *const TC = (&ab[i].as_ref()[j]) as *const TI as _;
            *loc = *val
        }
    }
}

#[inline(never)]
unsafe fn kernel<TI, const MR: usize, const NR: usize>(mut pnl: *const FusedKerSpec<TI>) -> isize
where
    TI: LADatum + ScaleShiftAndRound + AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    unsafe {
        let mut ab = [[TI::zero(); NR]; MR];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = std::mem::zeroed(),
                FusedKerSpec::ScalarAdd(a) => scalar!(ab, a, |a, b| a + b),
                FusedKerSpec::ScalarMul(a) => scalar!(ab, a, |a, b| a * b),
                FusedKerSpec::ScalarMin(m) => scalar!(ab, m, |a, b| if a < b { a } else { b }),
                FusedKerSpec::ScalarMax(m) => scalar!(ab, m, |a, b| if a > b { a } else { b }),
                FusedKerSpec::ScalarSub(m) => scalar!(ab, m, |a, b| a - b),
                FusedKerSpec::ScalarSubF(m) => scalar!(ab, m, |a, b| b - a),
                FusedKerSpec::LeakyRelu(m) => {
                    scalar!(ab, m, |a, b| if b > TI::zero() { b } else { a * b })
                }
                FusedKerSpec::PerRowMin(m) => per_row!(ab, m, |a, b| if a < b { a } else { b }),
                FusedKerSpec::PerRowMax(m) => per_row!(ab, m, |a, b| if a > b { a } else { b }),
                FusedKerSpec::PerRowAdd(m) => per_row!(ab, m, |a, b| a + b),
                FusedKerSpec::PerRowMul(m) => per_row!(ab, m, |a, b| a * b),
                FusedKerSpec::PerRowSub(m) => per_row!(ab, m, |a, b| a - b),
                FusedKerSpec::PerRowSubF(m) => per_row!(ab, m, |a, b| b - a),
                FusedKerSpec::PerColMin(m) => per_col!(ab, m, |a, b| if a < b { a } else { b }),
                FusedKerSpec::PerColMax(m) => per_col!(ab, m, |a, b| if a > b { a } else { b }),
                FusedKerSpec::PerColAdd(m) => per_col!(ab, m, |a, b| a + b),
                FusedKerSpec::PerColMul(m) => per_col!(ab, m, |a, b| a * b),
                FusedKerSpec::PerColSub(m) => per_col!(ab, m, |a, b| a - b),
                FusedKerSpec::PerColSubF(m) => per_col!(ab, m, |a, b| b - a),
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] += *rows.add(i) * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(tile) => {
                    if tile.item_size == TI::datum_type().size_of() {
                        for i in 0usize..MR {
                            for j in 0usize..NR {
                                let value: *const TI = tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as _;
                                ab[i].as_mut()[j] += *value;
                            }
                        }
                    } else if TI::datum_type() == i32::datum_type() && tile.item_size == 1 {
                        for i in 0usize..MR {
                            for j in 0usize..NR {
                                let value: i8 = *(tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as *const i8);
                                let acc: *mut i32 = ab[i].as_mut().as_mut_ptr().add(j) as *mut i32;
                                *acc += value as i32;
                            }
                        }
                    } else {
                        unimplemented!("Missing AddUnicast type");
                    }
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] = ab[i][j].q_shl(shift);
                        }
                    }
                }
                FusedKerSpec::RoundingShiftRight(shift, rp) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] = ab[i][j].q_shr(shift, rp);
                        }
                    }
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] = ab[i][j].q_scale(Scaler::from_fuse_params(shift, rp, mult));
                        }
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    use std::mem::transmute;
                    if TI::datum_type().is_float() {
                        add_mat_mul::<MR, NR, TI, TI, TI>(pa, pb, k, &mut ab);
                    } else if TI::datum_type() == i32::datum_type() {
                        let ab = transmute::<&mut [[TI; NR]; MR], &mut [[i32; NR]; MR]>(&mut ab);
                        if packing == 0 {
                            add_mat_mul::<MR, NR, i32, i32, i32>(pa, pb, k, ab)
                        } else if packing == 1 {
                            add_mat_mul::<MR, NR, i32, i8, i8>(pa, pb, k, ab)
                        } else {
                            return 1;
                        }
                    } else {
                        return 1;
                    }
                }
                FusedKerSpec::Store(tile) => match tile.item_size {
                    1 => store_t::<MR, NR, u8, _>(&tile, &ab),
                    2 => store_t::<MR, NR, u16, _>(&tile, &ab),
                    4 => store_t::<MR, NR, u32, _>(&tile, &ab),
                    8 => store_t::<MR, NR, f64, _>(&tile, &ab),
                    _ => unimplemented!(),
                },
            };
            pnl = pnl.add(1);
        }
    }
    0
}

MMMKernelWrapper!(f16, generic_f16_4x4; kernel::<f16, 4, 4>; 4, 4; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(f16, generic_f16_4x1; kernel::<f16, 4, 1>; 4, 1; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(f32, generic_f32_4x4; kernel::<f32, 4, 4>; 4, 4; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(f32, generic_f32_4x1; kernel::<f32, 4, 1>; 4, 1; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(f64, generic_f64_4x4; kernel::<f64, 4, 4>; 4, 4; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(f64, generic_f64_4x1; kernel::<f64, 4, 1>; 4, 1; 4, 4; 0, 0; no_prefetch, true);
MMMKernelWrapper!(i32, generic_i32_4x4; kernel::<i32, 4, 4>; 4, 4; 4, 4; 0, 0; no_prefetch, true,
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 4, 4, 0);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 4, 4, 0);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, generic_i32_4x4, i8i8:1, i8, i8, i32, i32 }
);

MMMKernelWrapper!(i32, generic_i32_4x1; kernel::<i32, 4, 1>; 4, 1; 4, 4; 0, 0; no_prefetch, true,
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 4, 4, 0);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 1, 4, 0);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, generic_i32_4x1, i8i8:1, i8, i8, i32, i32 }
);

#[cfg(test)]
MMMKernelWrapper!(f32, generic_f32_3x2; kernel::<f32, 3, 2>; 3, 2; 4, 4; 0, 0; no_prefetch, true);

#[cfg(test)]
MMMKernelWrapper!(i32, generic_i32_3x2; kernel::<i32, 3, 2>; 3, 2; 4, 4; 0, 0; no_prefetch, true,
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 3, 4, 0);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 2, 4, 0);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, generic_i32_3x2, i8i8:1, i8, i8, i32, i32 }
);
