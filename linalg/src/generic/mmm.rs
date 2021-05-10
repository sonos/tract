use num_traits::{AsPrimitive, Bounded, Zero};
use std::marker::PhantomData;
use std::{fmt, ops};

use tract_data::prelude::*;

use crate::frame::mmm::LinearSpec::*;
use crate::frame::mmm::PanelStore::*;
use crate::frame::mmm::*;

use num_traits::sign::Signed;

pub trait PseudoRightShift {
    fn q_away(self, mult: Self, shift: usize) -> Self;
    fn q_even(self, mult: Self, shift: usize) -> Self;
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self;
}

impl PseudoRightShift for i32 {
    fn q_even(self, mult: Self, shift: usize) -> Self {
        let v = ((self as i64 * mult as i64) >> (30 + shift)) as i32;
        let truncated = v.abs();
        let nudge = ((truncated & 0x3) == 0x3) as usize as i32;
        let pos = (truncated + nudge) >> 1;
        if v.is_negative() {
            -pos
        } else {
            pos
        }
    }
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self {
        let v = ((self as i64 * mult as i64) >> (30 + shift)) as i32;
        (v + 1) >> 1
    }
    fn q_away(self, mult: Self, shift: usize) -> Self {
        let v = ((self.abs() as i64 * mult as i64) >> (30 + shift)) as i32;
        ((v + 1) >> 1) * self.signum()
    }
}

impl PseudoRightShift for f32 {
    fn q_even(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
    fn q_away(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GenericMmm4x4<TA, TB, TC, TI>(PhantomData<(TA, TB, TC, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static;

unsafe impl<TA, TB, TC, TI> Send for GenericMmm4x4<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

unsafe impl<TA, TB, TC, TI> Sync for GenericMmm4x4<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

impl<TA, TB, TC, TI> MatMatMulKer<TI> for GenericMmm4x4<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static + Bounded,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + Signed
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        "generic"
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
    fn kernel(spec: &MatMatMulKerSpec<TI>) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); 4]; 4];
            match (*spec.a, *spec.b, *spec.linear) {
                (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
                    let a = a as *const TA;
                    let b = b as *const TB;
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
                (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
                    let a = a as *const TA;
                    let col_ptrs = col_ptrs as *const *const TB;
                    let pb0 = *(col_ptrs.offset(0));
                    let pb1 = *(col_ptrs.offset(1));
                    let pb2 = *(col_ptrs.offset(2));
                    let pb3 = *(col_ptrs.offset(3));
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let offset = *row_byte_offsets.offset(i as isize)
                            / std::mem::size_of::<TB>() as isize;
                        let b0 = *(pb0.offset(offset));
                        let b1 = *(pb1.offset(offset));
                        let b2 = *(pb2.offset(offset));
                        let b3 = *(pb3.offset(offset));
                        ab[0][0] += a[0].as_() * b0.as_();
                        ab[0][1] += a[0].as_() * b1.as_();
                        ab[0][2] += a[0].as_() * b2.as_();
                        ab[0][3] += a[0].as_() * b3.as_();
                        ab[1][0] += a[1].as_() * b0.as_();
                        ab[1][1] += a[1].as_() * b1.as_();
                        ab[1][2] += a[1].as_() * b2.as_();
                        ab[1][3] += a[1].as_() * b3.as_();
                        ab[2][0] += a[2].as_() * b0.as_();
                        ab[2][1] += a[2].as_() * b1.as_();
                        ab[2][2] += a[2].as_() * b2.as_();
                        ab[2][3] += a[2].as_() * b3.as_();
                        ab[3][0] += a[3].as_() * b0.as_();
                        ab[3][1] += a[3].as_() * b1.as_();
                        ab[3][2] += a[3].as_() * b2.as_();
                        ab[3][3] += a[3].as_() * b3.as_();
                    }
                }
                _ => return 1,
            }
            let mut pnl = spec.non_linear;
            loop {
                if pnl.is_null() {
                    break;
                }
                match *pnl {
                    FusedKerSpec::Done => break,
                    FusedKerSpec::PerRowMul(bias) => {
                        for i in 0..4 {
                            ab[i][0] *= *bias.offset(i as isize);
                            ab[i][1] *= *bias.offset(i as isize);
                            ab[i][2] *= *bias.offset(i as isize);
                            ab[i][3] *= *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerRowAdd(bias) => {
                        for i in 0..4 {
                            ab[i][0] += *bias.offset(i as isize);
                            ab[i][1] += *bias.offset(i as isize);
                            ab[i][2] += *bias.offset(i as isize);
                            ab[i][3] += *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerColMul(bias) => {
                        for i in 0..4 {
                            ab[0][i] *= *bias.offset(i as isize);
                            ab[1][i] *= *bias.offset(i as isize);
                            ab[2][i] *= *bias.offset(i as isize);
                            ab[3][i] *= *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerColAdd(bias) => {
                        for i in 0..4 {
                            ab[0][i] += *bias.offset(i as isize);
                            ab[1][i] += *bias.offset(i as isize);
                            ab[2][i] += *bias.offset(i as isize);
                            ab[3][i] += *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::Min(m) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] = if m < ab[i][j] { m } else { ab[i][j] }
                            }
                        }
                    }
                    FusedKerSpec::Max(m) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] = if m > ab[i][j] { m } else { ab[i][j] }
                            }
                        }
                    }
                    FusedKerSpec::AddRowColProducts(rows, cols) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] += *rows.offset(i as isize) * *cols.offset(j as isize);
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
                    FusedKerSpec::QTowardsEven(mult, shift) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] = ab[i][j].q_even(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::QTowardsPlusInf(mult, shift) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] = ab[i][j].q_to_plus_inf(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::QAway(mult, shift) => {
                        for i in 0..4 {
                            for j in 0..4 {
                                ab[i][j] = ab[i][j].q_away(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TC, TI, _>(&tile, &mut ab),
                }
                pnl = pnl.add(1);
            }
            let Tile { ptr: c, row_byte_stride, col_byte_stride, .. } = spec.c;
            let c = *c as *mut TC;
            let rsc = *row_byte_stride as usize / std::mem::size_of::<TC>();
            let csc = *col_byte_stride as usize / std::mem::size_of::<TC>();
            let c = c as *mut TC;
            let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
            c[0 * csc + 0 * rsc] = ab[0][0].as_();
            c[1 * csc + 0 * rsc] = ab[0][1].as_();
            c[2 * csc + 0 * rsc] = ab[0][2].as_();
            c[3 * csc + 0 * rsc] = ab[0][3].as_();
            c[0 * csc + 1 * rsc] = ab[1][0].as_();
            c[1 * csc + 1 * rsc] = ab[1][1].as_();
            c[2 * csc + 1 * rsc] = ab[1][2].as_();
            c[3 * csc + 1 * rsc] = ab[1][3].as_();
            c[0 * csc + 2 * rsc] = ab[2][0].as_();
            c[1 * csc + 2 * rsc] = ab[2][1].as_();
            c[2 * csc + 2 * rsc] = ab[2][2].as_();
            c[3 * csc + 2 * rsc] = ab[2][3].as_();
            c[0 * csc + 3 * rsc] = ab[3][0].as_();
            c[1 * csc + 3 * rsc] = ab[3][1].as_();
            c[2 * csc + 3 * rsc] = ab[3][2].as_();
            c[3 * csc + 3 * rsc] = ab[3][3].as_();
        }
        return 0;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GenericMmm4x1<TA, TB, TC, TI>(PhantomData<(TA, TB, TC, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static;

unsafe impl<TA, TB, TC, TI> Send for GenericMmm4x1<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

unsafe impl<TA, TB, TC, TI> Sync for GenericMmm4x1<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

impl<TA, TB, TC, TI> MatMatMulKer<TI> for GenericMmm4x1<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static + Bounded,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PseudoRightShift
        + PartialOrd
        + Zero
        + Signed
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        "generic"
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
    fn kernel(spec: &MatMatMulKerSpec<TI>) -> isize {
        unsafe {
            let mut ab = [TI::zero(); 4];
            match (*spec.a, *spec.b, *spec.linear) {
                (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
                    let a = a as *const TA;
                    let b = b as *const TB;
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let b = *b.offset(i as isize);
                        ab[0] += a[0].as_() * b.as_();
                        ab[1] += a[1].as_() * b.as_();
                        ab[2] += a[2].as_() * b.as_();
                        ab[3] += a[3].as_() * b.as_();
                    }
                }
                (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
                    let a = a as *const TA;
                    let col_ptrs = col_ptrs as *const *const TB;
                    let pb0 = *(col_ptrs.offset(0));
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let offset = *row_byte_offsets.offset(i as isize)
                            / std::mem::size_of::<TB>() as isize;
                        let b0 = *(pb0.offset(offset));
                        ab[0] += a[0].as_() * b0.as_();
                        ab[1] += a[1].as_() * b0.as_();
                        ab[2] += a[2].as_() * b0.as_();
                        ab[3] += a[3].as_() * b0.as_();
                    }
                }
                _ => return 1,
            }
            let mut pnl = spec.non_linear;
            loop {
                if pnl.is_null() {
                    break;
                }
                match *pnl {
                    FusedKerSpec::Done => break,
                    FusedKerSpec::PerRowMul(bias) => {
                        for i in 0..4 {
                            ab[i] *= *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerRowAdd(bias) => {
                        for i in 0..4 {
                            ab[i] += *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerColMul(bias) => {
                        ab[0] *= *bias;
                        ab[1] *= *bias;
                        ab[2] *= *bias;
                        ab[3] *= *bias;
                    }
                    FusedKerSpec::PerColAdd(bias) => {
                        ab[0] += *bias;
                        ab[1] += *bias;
                        ab[2] += *bias;
                        ab[3] += *bias;
                    }
                    FusedKerSpec::Min(m) => {
                        for i in 0..4 {
                            ab[i] = if m < ab[i] { m } else { ab[i] }
                        }
                    }
                    FusedKerSpec::Max(m) => {
                        for i in 0..4 {
                            ab[i] = if m > ab[i] { m } else { ab[i] }
                        }
                    }
                    FusedKerSpec::AddRowColProducts(rows, cols) => {
                        let col = *cols;
                        for i in 0..4 {
                            ab[i] += *rows.offset(i as isize) * col;
                        }
                    }
                    FusedKerSpec::ScalarAdd(a) => {
                        for i in 0..4 {
                            ab[i] += a;
                        }
                    }
                    FusedKerSpec::ScalarMul(a) => {
                        for i in 0..4 {
                            ab[i] *= a;
                        }
                    }
                    FusedKerSpec::QTowardsEven(mult, shift) => {
                        for i in 0..4 {
                            ab[i] = ab[i].q_even(mult, shift);
                        }
                    }
                    FusedKerSpec::QTowardsPlusInf(mult, shift) => {
                        for i in 0..4 {
                            ab[i] = ab[i].q_to_plus_inf(mult, shift);
                        }
                    }
                    FusedKerSpec::QAway(mult, shift) => {
                        for i in 0..4 {
                            ab[i] = ab[i].q_away(mult, shift);
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TC, TI, _>(
                        &tile,
                        &mut [
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(0) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(1) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(2) as _, 1),
                            std::slice::from_raw_parts_mut(ab.as_ptr().offset(3) as _, 1),
                        ],
                    ),
                }
                pnl = pnl.add(1);
            }
            let Tile { ptr: c, row_byte_stride, .. } = spec.c;
            let c = *c as *mut TC;
            let rsc = *row_byte_stride as usize / std::mem::size_of::<TC>();
            let c = std::slice::from_raw_parts_mut(c, 1 + 3 * rsc);
            c[0 * rsc] = ab[0].as_();
            c[1 * rsc] = ab[1].as_();
            c[2 * rsc] = ab[2].as_();
            c[3 * rsc] = ab[3].as_();
        }
        return 0;
    }
}

#[cfg(test)]
#[derive(Copy, Clone, Debug)]
pub struct GenericMmmTest3x2<TA, TB, TC, TI>(PhantomData<(TA, TB, TC, TI)>)
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static;

#[cfg(test)]
unsafe impl<TA, TB, TC, TI> Send for GenericMmmTest3x2<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

#[cfg(test)]
unsafe impl<TA, TB, TC, TI> Sync for GenericMmmTest3x2<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PartialOrd
        + Zero
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
{
}

#[cfg(test)]
impl<TA, TB, TC, TI> MatMatMulKer<TI> for GenericMmmTest3x2<TA, TB, TC, TI>
where
    TA: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TB: Datum + Copy + fmt::Debug + AsPrimitive<TI>,
    TC: Datum + Copy + fmt::Debug + AsPrimitive<TI> + 'static + Bounded,
    TI: Datum
        + Copy
        + ops::AddAssign
        + ops::Mul<Output = TI>
        + ops::MulAssign
        + PartialOrd
        + PseudoRightShift
        + Zero
        + Signed
        + fmt::Debug
        + fmt::Display
        + AsPrimitive<TC>
        + 'static,
    usize: AsPrimitive<TI>,
{
    #[inline(always)]
    fn name() -> &'static str {
        "generic-test-3x2"
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
    fn kernel(spec: &MatMatMulKerSpec<TI>) -> isize {
        unsafe {
            let mut ab = [[TI::zero(); 2]; 3];
            match (*spec.a, *spec.b, *spec.linear) {
                (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
                    let a = a as *const TA;
                    let b = b as *const TB;
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
                (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
                    let a = a as *const TA;
                    let col_ptrs = col_ptrs as *const *const TB;
                    let pb0 = *(col_ptrs.offset(0));
                    let pb1 = *(col_ptrs.offset(1));
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(3 * i as isize), 3);
                        let offset = *row_byte_offsets.offset(i as isize)
                            / std::mem::size_of::<TB>() as isize;
                        let b0 = *(pb0.offset(offset));
                        let b1 = *(pb1.offset(offset));
                        ab[0][0] += a[0].as_() * b0.as_();
                        ab[0][1] += a[0].as_() * b1.as_();
                        ab[1][0] += a[1].as_() * b0.as_();
                        ab[1][1] += a[1].as_() * b1.as_();
                        ab[2][0] += a[2].as_() * b0.as_();
                        ab[2][1] += a[2].as_() * b1.as_();
                    }
                }
                _ => return 1,
            }
            let mut pnl = spec.non_linear;
            loop {
                if pnl.is_null() {
                    break;
                }
                match *pnl {
                    FusedKerSpec::Done => break,
                    FusedKerSpec::PerRowMul(bias) => {
                        for i in 0..3 {
                            ab[i][0] *= *bias.offset(i as isize);
                            ab[i][1] *= *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerRowAdd(bias) => {
                        for i in 0..3 {
                            ab[i][0] += *bias.offset(i as isize);
                            ab[i][1] += *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerColMul(bias) => {
                        for i in 0..2 {
                            ab[0][i] *= *bias.offset(i as isize);
                            ab[1][i] *= *bias.offset(i as isize);
                            ab[2][i] *= *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::PerColAdd(bias) => {
                        for i in 0..2 {
                            ab[0][i] += *bias.offset(i as isize);
                            ab[1][i] += *bias.offset(i as isize);
                            ab[2][i] += *bias.offset(i as isize);
                        }
                    }
                    FusedKerSpec::Min(m) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = if m < ab[i][j] { m } else { ab[i][j] }
                            }
                        }
                    }
                    FusedKerSpec::Max(m) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = if m > ab[i][j] { m } else { ab[i][j] }
                            }
                        }
                    }
                    FusedKerSpec::AddRowColProducts(rows, cols) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] += *rows.offset(i as isize) * *cols.offset(j as isize);
                            }
                        }
                    }
                    FusedKerSpec::ScalarAdd(a) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] += a;
                            }
                        }
                    }
                    FusedKerSpec::ScalarMul(a) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] *= a;
                            }
                        }
                    }
                    FusedKerSpec::QTowardsEven(mult, shift) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = ab[i][j].q_even(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::QTowardsPlusInf(mult, shift) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = ab[i][j].q_to_plus_inf(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::QAway(mult, shift) => {
                        for i in 0..3 {
                            for j in 0..2 {
                                ab[i][j] = ab[i][j].q_away(mult, shift);
                            }
                        }
                    }
                    FusedKerSpec::AddUnicast(tile) => add_unicast::<TC, TI, _>(&tile, &mut ab),
                }
                pnl = pnl.add(1);
            }
            let Tile { ptr: c, row_byte_stride, col_byte_stride, .. } = spec.c;
            let c = *c as *mut TC;
            let rsc = *row_byte_stride as usize / std::mem::size_of::<TC>();
            let csc = *col_byte_stride as usize / std::mem::size_of::<TC>();
            let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
            c[0 * csc + 0 * rsc] = ab[0][0].as_();
            c[1 * csc + 0 * rsc] = ab[0][1].as_();
            c[0 * csc + 1 * rsc] = ab[1][0].as_();
            c[1 * csc + 1 * rsc] = ab[1][1].as_();
            c[0 * csc + 2 * rsc] = ab[2][0].as_();
            c[1 * csc + 2 * rsc] = ab[2][1].as_();
        }
        return 0;
    }
}

unsafe fn add_unicast<TC, TI, AB>(tile: &Tile, ab: &mut [AB])
where
    TC: AsPrimitive<TI> + Copy,
    TI: Datum + ops::AddAssign<TI> + Copy,
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
                let value: *const TC = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                ab[i].as_mut()[j] += (*value).as_();
            }
        }
    } else {
        unimplemented!("Missing AddUnicast type");
    }
}

test_mmm_kernel_f32!(crate::generic::mmm::GenericMmm4x4<f32, f32, f32, f32>, test_GenericMmm4x4_f32, true);
test_mmm_kernel_i8!(crate::generic::mmm::GenericMmm4x4<i8, i8, i8, i32>, test_GenericMmm4x4_i8, true);
test_mmm_kernel_u8!(crate::generic::mmm::GenericMmm4x4<u8, u8, u8, i32>, test_GenericMmm4x4_u8, true);
test_mmm_kernel_i8_i32!(crate::generic::mmm::GenericMmm4x4<i8, i8, i32, i32>, test_GenericMmm4x4_i8_i32, true);
test_mmm_kernel_i8_u8_i32!(crate::generic::mmm::GenericMmm4x4<i8, u8, i32, i32>, test_GenericMmm4x4_i8_u8_i32, true);

test_mmm_kernel_f32!(crate::generic::mmm::GenericMmm4x1<f32, f32, f32, f32>, test_GenericMmm4x1_f32, true);
test_mmm_kernel_i8!(crate::generic::mmm::GenericMmm4x1<i8, i8, i8, i32>, test_GenericMmm4x1_i8, true);
test_mmm_kernel_u8!(crate::generic::mmm::GenericMmm4x1<u8, u8, u8, i32>, test_GenericMmm4x1_u8, true);
test_mmm_kernel_i8_i32!(crate::generic::mmm::GenericMmm4x1<i8, i8, i32, i32>, test_GenericMmm4x1_i8_i32, true);
test_mmm_kernel_i8_u8_i32!(crate::generic::mmm::GenericMmm4x1<i8, u8, i32, i32>, test_GenericMmm4x1_i8_u8_i32, true);

test_mmm_kernel_f32!(crate::generic::mmm::GenericMmmTest3x2<f32, f32, f32, f32>, test_GenericMmmTest3x2_f32, true);
test_mmm_kernel_i8!(crate::generic::mmm::GenericMmmTest3x2<i8, i8, i8, i32>, test_GenericMmmTest3x2_i8, true);
test_mmm_kernel_u8!(crate::generic::mmm::GenericMmmTest3x2<u8, u8, u8, i32>, test_GenericMmmTest3x2_u8, true);
test_mmm_kernel_i8_i32!(crate::generic::mmm::GenericMmmTest3x2<i8, i8, i32, i32>, test_GenericMmmTest3x2_i8_i32, true);
test_mmm_kernel_i8_u8_i32!(crate::generic::mmm::GenericMmmTest3x2<i8, u8, i32, i32>, test_GenericMmmTest3x2_i8_u8_i32, true);
