#![allow(clippy::needless_range_loop)]
use num_traits::AsPrimitive;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

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

#[derive(Copy, Clone, Debug, Default)]
pub struct GenericMmm<const MR: usize, const NR: usize, TA, TB, TI>(PhantomData<(TA, TB, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound;

unsafe impl<const MR: usize, const NR: usize, TA, TB, TI> Send for GenericMmm<MR, NR, TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

unsafe impl<const MR: usize, const NR: usize, TA, TB, TI> Sync for GenericMmm<MR, NR, TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

impl<const MR: usize, const NR: usize, TA, TB, TI> GenericMmm<MR, NR, TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound + AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    pub fn mmm(&self) -> Box<dyn MatMatMul> {
        Box::new(*self)
    }
}

impl<const MR: usize, const NR: usize, TA, TB, TI> MatMatMulKer for GenericMmm<MR, NR, TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound + AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    type Acc = TI;
    #[inline(always)]
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("generic_{:?}_{}x{}", TI::datum_type(), MR, NR).to_lowercase())
    }
    #[inline(always)]
    fn mr(&self) -> usize {
        MR
    }
    #[inline(always)]
    fn nr(&self) -> usize {
        NR
    }
    fn end_padding_packed_a(&self) -> usize {
        0
    }
    fn end_padding_packed_b(&self) -> usize {
        0
    }
    #[inline(always)]
    fn alignment_bytes_packed_a(&self) -> usize {
        std::mem::size_of::<TA>()
    }
    #[inline(always)]
    fn alignment_bytes_packed_b(&self) -> usize {
        std::mem::size_of::<TB>()
    }
    #[inline(never)]
    fn kernel(&self, spec: &[FusedKerSpec<TI>]) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); NR]; MR];
            let mut pnl = spec.as_ptr();
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
                                    )
                                        as _;
                                    ab[i].as_mut()[j] += *value;
                                }
                            }
                        } else if TI::datum_type() == i32::datum_type() && tile.item_size == 1 {
                            for i in 0usize..MR {
                                for j in 0usize..NR {
                                    let value: i8 = *(tile.ptr.offset(
                                        tile.row_byte_stride * i as isize
                                            + tile.col_byte_stride * j as isize,
                                    )
                                        as *const i8);
                                    let acc: *mut i32 =
                                        ab[i].as_mut().as_mut_ptr().add(j) as *mut i32;
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
                                ab[i][j] =
                                    ab[i][j].q_scale(Scaler::from_fuse_params(shift, rp, mult));
                            }
                        }
                    }
                    FusedKerSpec::AddMatMul { k, pa, pb, .. } => {
                        // FIXME type
                        if TI::datum_type().is_float() {
                            add_mat_mul::<MR, NR, TI, TI, TI>(pa, pb, k, &mut ab);
                        } else {
                            add_mat_mul::<MR, NR, i32, i8, i8>(pa, pb, k, std::mem::transmute(&mut ab))
                        }
                        /*
                         */
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
}

unsafe fn add_mat_mul<const MR: usize, const NR: usize, TI, TA, TB>(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[TI; NR]; MR],
) where
    TA: AsPrimitive<TI>,
    TB: AsPrimitive<TI>,
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

#[allow(non_upper_case_globals)]
pub const generic_f16_4x4: GenericMmm<4, 4, f16, f16, f16> = GenericMmm(PhantomData);
test_mmm_kernel_f16!(generic_f16_4x4, true);

#[allow(non_upper_case_globals)]
pub const generic_f16_4x1: GenericMmm<4, 1, f16, f16, f16> = GenericMmm(PhantomData);
test_mmm_kernel_f16!(generic_f16_4x1, true);

#[allow(non_upper_case_globals)]
pub const generic_f32_4x4: GenericMmm<4, 4, f32, f32, f32> = GenericMmm(PhantomData);
test_mmm_kernel_f32!(generic_f32_4x4, true);

#[allow(non_upper_case_globals)]
pub const generic_f64_4x4: GenericMmm<4, 4, f64, f64, f64> = GenericMmm(PhantomData);
test_mmm_kernel_f64!(generic_f64_4x4, true);

#[allow(non_upper_case_globals)]
pub const generic_i32_4x4: GenericMmm<4, 4, i8, i8, i32> = GenericMmm(PhantomData);
test_mmm_kernel_i32!(generic_i32_4x4, true);

#[allow(non_upper_case_globals)]
pub const generic_f32_4x1: GenericMmm<4, 1, f32, f32, f32> = GenericMmm(PhantomData);
test_mmm_kernel_f32!(generic_f32_4x1, true);

#[allow(non_upper_case_globals)]
pub const generic_f64_4x1: GenericMmm<4, 1, f64, f64, f64> = GenericMmm(PhantomData);
test_mmm_kernel_f64!(generic_f64_4x1, true);

#[allow(non_upper_case_globals)]
pub const generic_i32_4x1: GenericMmm<4, 1, i8, i8, i32> = GenericMmm(PhantomData);
test_mmm_kernel_i32!(generic_i32_4x1, true);

#[cfg(test)]
#[allow(non_upper_case_globals)]
const generic_f32_3x2: GenericMmm<3, 2, f32, f32, f32> = GenericMmm(PhantomData);
test_mmm_kernel_f32!(generic_f32_3x2, true);

#[cfg(test)]
#[allow(non_upper_case_globals)]
const generic_i32_3x2: GenericMmm<3, 2, i8, i8, i32> = GenericMmm(PhantomData);
test_mmm_kernel_i32!(generic_i32_3x2, true);
