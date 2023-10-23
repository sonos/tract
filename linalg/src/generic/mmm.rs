#![allow(clippy::needless_range_loop)]
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::{fmt, ops};

use tract_data::prelude::*;

use super::*;
use crate::frame::mmm::*;
use crate::LADatum;

#[derive(Copy, Clone, Debug)]
pub struct GenericMmm4x4<TA, TB, TI>(PhantomData<(TA, TB, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound;

unsafe impl<TA, TB, TI> Send for GenericMmm4x4<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

unsafe impl<TA, TB, TI> Sync for GenericMmm4x4<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

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

impl<TA, TB, TI> MatMatMulKer<TI> for GenericMmm4x4<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        match TI::datum_type() {
            DatumType::F16 => "generic_f16_4x4",
            DatumType::F32 => "generic_f32_4x4",
            DatumType::I32 => "generic_i32_4x4",
            DatumType::F64 => "generic_f64_4x4",
            _ => panic!(),
        }
    }
    #[inline(always)]
    fn mr() -> usize {
        4
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    fn end_padding_packed_a() -> usize {
        0
    }
    fn end_padding_packed_b() -> usize {
        0
    }
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize {
        std::mem::size_of::<TA>()
    }
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize {
        std::mem::size_of::<TB>()
    }
    #[inline(never)]
    fn kernel(spec: &[FusedKerSpec<TI>]) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); 4]; 4];
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
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] += *rows.add(i) * *cols.add(j);
                            }
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TI, _>(&tile, &mut ab),
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
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] =
                                    ab[i][j].q_scale(Scaler::from_fuse_params(shift, rp, mult));
                            }
                        }
                    }
                    FusedKerSpec::AddMatMul { k, pa, pb, .. } => {
                        let a = pa as *const TA;
                        let b = pb as *const TB;
                        for i in 0..k {
                            let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                            let b = std::slice::from_raw_parts(b.offset(4 * i as isize), 4);
                            ab[0][0] += a[0].as_() * b[0].as_();
                            ab[0][1] += a[0].as_() * b[1].as_();
                            ab[0][2] += a[0].as_() * b[2].as_();
                            ab[0][3] += a[0].as_() * b[3].as_();
                            ab[1][0] += a[1].as_() * b[0].as_();
                            ab[1][1] += a[1].as_() * b[1].as_();
                            ab[1][2] += a[1].as_() * b[2].as_();
                            ab[1][3] += a[1].as_() * b[3].as_();
                            ab[2][0] += a[2].as_() * b[0].as_();
                            ab[2][1] += a[2].as_() * b[1].as_();
                            ab[2][2] += a[2].as_() * b[2].as_();
                            ab[2][3] += a[2].as_() * b[3].as_();
                            ab[3][0] += a[3].as_() * b[0].as_();
                            ab[3][1] += a[3].as_() * b[1].as_();
                            ab[3][2] += a[3].as_() * b[2].as_();
                            ab[3][3] += a[3].as_() * b[3].as_();
                        }
                    }
                    FusedKerSpec::Store(tile) => store(&tile, &ab),
                };
                pnl = pnl.add(1);
            }
        }
        0
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GenericMmm4x1<TA, TB, TI>(PhantomData<(TA, TB, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound;

unsafe impl<TA, TB, TI> Send for GenericMmm4x1<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

unsafe impl<TA, TB, TI> Sync for GenericMmm4x1<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

impl<TA, TB, TI> MatMatMulKer<TI> for GenericMmm4x1<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        match TI::datum_type() {
            DatumType::F16 => "generic_f16_4x1",
            DatumType::F32 => "generic_f32_4x1",
            DatumType::I32 => "generic_i32_4x1",
            DatumType::F64 => "generic_f64_4x1",
            _ => panic!(),
        }
    }
    #[inline(always)]
    fn mr() -> usize {
        4
    }
    #[inline(always)]
    fn nr() -> usize {
        1
    }
    fn end_padding_packed_a() -> usize {
        0
    }
    fn end_padding_packed_b() -> usize {
        0
    }
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize {
        std::mem::size_of::<TA>()
    }
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize {
        std::mem::size_of::<TB>()
    }
    #[inline(never)]
    fn kernel(spec: &[FusedKerSpec<TI>]) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); 1]; 4];
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
                    FusedKerSpec::PerRowMul(m) => per_row!(ab, m, |a, b| a * b),
                    FusedKerSpec::PerRowMin(m) => per_row!(ab, m, |a, b| if a < b { a } else { b }),
                    FusedKerSpec::PerRowMax(m) => per_row!(ab, m, |a, b| if a > b { a } else { b }),
                    FusedKerSpec::PerRowAdd(m) => per_row!(ab, m, |a, b| a + b),
                    FusedKerSpec::PerRowSub(m) => per_row!(ab, m, |a, b| a - b),
                    FusedKerSpec::PerRowSubF(m) => per_row!(ab, m, |a, b| b - a),
                    FusedKerSpec::PerColMin(m) => per_col!(ab, m, |a, b| if a < b { a } else { b }),
                    FusedKerSpec::PerColMax(m) => per_col!(ab, m, |a, b| if a > b { a } else { b }),
                    FusedKerSpec::PerColAdd(m) => per_col!(ab, m, |a, b| a + b),
                    FusedKerSpec::PerColMul(m) => per_col!(ab, m, |a, b| a * b),
                    FusedKerSpec::PerColSub(m) => per_col!(ab, m, |a, b| a - b),
                    FusedKerSpec::PerColSubF(m) => per_col!(ab, m, |a, b| b - a),
                    FusedKerSpec::AddRowColProducts(rows, cols) => {
                        let col = *cols;
                        for i in 0..4 {
                            ab[i][0] += *rows.add(i) * col;
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TI, _>(
                        &tile,
                        &mut [
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(0) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(1) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(2) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(3) as _, 1),
                        ],
                    ),
                    FusedKerSpec::ShiftLeft(shift) => {
                        for i in 0..4 {
                            ab[i][0] = ab[i][0].q_shl(shift);
                        }
                    }
                    FusedKerSpec::RoundingShiftRight(shift, rp) => {
                        for i in 0..4 {
                            ab[i][0] = ab[i][0].q_shr(shift, rp);
                        }
                    }
                    FusedKerSpec::QScale(shift, rp, mult) => {
                        for i in 0..4 {
                            ab[i][0] = ab[i][0].q_scale(Scaler::from_fuse_params(shift, rp, mult));
                        }
                    }
                    FusedKerSpec::AddMatMul { k, pa, pb, .. } => {
                        let a = pa as *const TA;
                        let b = pb as *const TB;
                        for i in 0..k {
                            let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                            let b = *b.add(i);
                            ab[0][0] += a[0].as_() * b.as_();
                            ab[1][0] += a[1].as_() * b.as_();
                            ab[2][0] += a[2].as_() * b.as_();
                            ab[3][0] += a[3].as_() * b.as_();
                        }
                    }
                    FusedKerSpec::Store(tile) => store(
                        &tile,
                        &[
                            std::slice::from_raw_parts(ab.as_ptr().offset(0) as _, 1),
                            std::slice::from_raw_parts(ab.as_ptr().offset(1) as _, 1),
                            std::slice::from_raw_parts(ab.as_ptr().offset(2) as _, 1),
                            std::slice::from_raw_parts(ab.as_ptr().offset(3) as _, 1),
                        ],
                    ),
                }
                pnl = pnl.add(1);
            }
        }
        0
    }
}

#[cfg(test)]
#[derive(Copy, Clone, Debug)]
pub struct GenericMmmTest3x2<TA, TB, TI>(PhantomData<(TA, TB, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound;

#[cfg(test)]
unsafe impl<TA, TB, TI> Send for GenericMmmTest3x2<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

#[cfg(test)]
unsafe impl<TA, TB, TI> Sync for GenericMmmTest3x2<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
{
}

#[cfg(test)]
impl<TA, TB, TI> MatMatMulKer<TI> for GenericMmmTest3x2<TA, TB, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TI: LADatum + ScaleShiftAndRound,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        match TI::datum_type() {
            DatumType::F16 => "generic_f16_3x2",
            DatumType::F32 => "generic_f32_3x2",
            DatumType::I32 => "generic_i32_3x2",
            DatumType::F64 => "generic_f64_3x2",
            _ => panic!(),
        }
    }
    #[inline(always)]
    fn mr() -> usize {
        3
    }
    #[inline(always)]
    fn nr() -> usize {
        2
    }
    fn end_padding_packed_a() -> usize {
        0
    }
    fn end_padding_packed_b() -> usize {
        0
    }
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize {
        std::mem::size_of::<TA>()
    }
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize {
        std::mem::size_of::<TB>()
    }
    #[inline(never)]
    fn kernel(spec: &[FusedKerSpec<TI>]) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); 2]; 3];
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
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] += *rows.add(i) * *cols.add(j);
                            }
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TI, _>(&tile, &mut ab),
                    FusedKerSpec::ShiftLeft(shift) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = ab[i][j].q_shl(shift);
                            }
                        }
                    }
                    FusedKerSpec::RoundingShiftRight(shift, rp) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = ab[i][j].q_shr(shift, rp)
                            }
                        }
                    }
                    FusedKerSpec::QScale(shift, rp, mult) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] =
                                    ab[i][j].q_scale(Scaler::from_fuse_params(shift, rp, mult));
                            }
                        }
                    }
                    FusedKerSpec::AddMatMul { k, pa, pb, .. } => {
                        let a = pa as *const TA;
                        let b = pb as *const TB;
                        for i in 0..k {
                            let a = std::slice::from_raw_parts(a.offset(3 * i as isize), 3);
                            let b = std::slice::from_raw_parts(b.offset(2 * i as isize), 2);
                            ab[0][0] += a[0].as_() * b[0].as_();
                            ab[0][1] += a[0].as_() * b[1].as_();
                            ab[1][0] += a[1].as_() * b[0].as_();
                            ab[1][1] += a[1].as_() * b[1].as_();
                            ab[2][0] += a[2].as_() * b[0].as_();
                            ab[2][1] += a[2].as_() * b[1].as_();
                        }
                    }
                    FusedKerSpec::Store(tile) => store(&tile, &ab),
                }
                pnl = pnl.add(1);
            }
        }
        0
    }
}

unsafe fn store_t<TC, TI, AB>(tile: &OutputStoreKer, ab: &[AB])
where
    TC: Copy,
    AB: AsRef<[TI]> + fmt::Debug,
{
    for i in 0usize..ab.len() {
        for j in 0usize..ab[0].as_ref().len() {
            let loc: *mut TC = tile
                .ptr
                .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                as _;
            let val: *const TC = (&ab[i].as_ref()[j]) as *const TI as _;
            *loc = *val
        }
    }
}

unsafe fn store<TI, AB>(tile: &OutputStoreKer, ab: &[AB])
where
    AB: AsRef<[TI]> + fmt::Debug,
{
    match tile.item_size {
        1 => store_t::<u8, _, _>(tile, ab),
        2 => store_t::<u16, _, _>(tile, ab),
        4 => store_t::<u32, _, _>(tile, ab),
        8 => store_t::<f64, _, _>(tile, ab),
        _ => unimplemented!(),
    }
}

unsafe fn add_unicast<TI, AB>(tile: &OutputStoreKer, ab: &mut [AB])
where
    TI: LADatum + ops::AddAssign<TI>,
    AB: AsMut<[TI]> + fmt::Debug,
{
    if tile.item_size == TI::datum_type().size_of() {
        for i in 0usize..ab.len() {
            for j in 0usize..ab[0].as_mut().len() {
                let value: *const TI = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                ab[i].as_mut()[j] += *value;
            }
        }
    } else if TI::datum_type() == i32::datum_type() && tile.item_size == 1 {
        for i in 0usize..ab.len() {
            for j in 0usize..ab[0].as_mut().len() {
                let value: i8 = *(tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as *const i8);
                let acc: *mut i32 = ab[i].as_mut().as_mut_ptr().add(j) as *mut i32;
                *acc += value as i32;
            }
        }
    } else {
        unimplemented!("Missing AddUnicast type");
    }
}

#[allow(non_camel_case_types)]
pub type generic_f16_4x4 = GenericMmm4x4<f16, f16, f16>;
test_mmm_kernel_f16!(generic_f16_4x4, true);

#[allow(non_camel_case_types)]
pub type generic_f32_4x4 = GenericMmm4x4<f32, f32, f32>;
test_mmm_kernel_f32!(generic_f32_4x4, true);

#[allow(non_camel_case_types)]
pub type generic_f64_4x4 = GenericMmm4x4<f64, f64, f64>;
test_mmm_kernel_f64!(generic_f64_4x4, true);

#[allow(non_camel_case_types)]
pub type generic_i32_4x4 = GenericMmm4x4<i8, i8, i32>;
test_mmm_kernel_i32!(generic_i32_4x4, true);

#[allow(non_camel_case_types)]
pub type generic_f32_4x1 = GenericMmm4x1<f32, f32, f32>;
test_mmm_kernel_f32!(generic_f32_4x1, true);

#[allow(non_camel_case_types)]
pub type generic_f64_4x1 = GenericMmm4x1<f64, f64, f64>;
test_mmm_kernel_f64!(generic_f64_4x1, true);

#[allow(non_camel_case_types)]
pub type generic_i32_4x1 = GenericMmm4x1<i8, i8, i32>;
test_mmm_kernel_i32!(generic_i32_4x1, true);

#[cfg(test)]
#[allow(non_camel_case_types)]
type generic_f32_3x2 = GenericMmmTest3x2<f32, f32, f32>;
test_mmm_kernel_f32!(generic_f32_3x2, true);

#[cfg(test)]
#[allow(non_camel_case_types)]
type generic_i32_3x2 = GenericMmmTest3x2<i8, i8, i32>;
test_mmm_kernel_i32!(generic_i32_3x2, true);
