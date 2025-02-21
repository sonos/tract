#![allow(clippy::needless_range_loop)]
use num_traits::AsPrimitive;

use tract_data::prelude::f16;
use tract_data::prelude::*;

use super::*;
use crate::frame::block_quant::{BlockQuant, NibbleReader, PackedBlockQuantFormat, Q4_0};
use crate::frame::mmm::*;
use crate::{has_fp16, LADatum, Ops};

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

unsafe fn add_mat_mul_pq40<const MR: usize, const NR: usize, TB, TI>(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[TI; NR]; MR],
) where
    TI: LADatum,
    f16: AsPrimitive<TI>,
    TB: AsPrimitive<TI>,
    i8: AsPrimitive<TI>,
{
    assert!(k % Q4_0.block_len() == 0);
    let len = (k * MR) / Q4_0.block_len() * Q4_0.block_bytes();
    let mut pa = NibbleReader::for_slice(std::slice::from_raw_parts(pa, len));
    let b = pb as *const TB;
    for bk in 0..k / 32 {
        let mut scales: [TI; MR] = [TI::zero(); MR];
        scales.iter_mut().for_each(|x| *x = pa.read_f16().as_());
        for ik in 0..32 {
            let mut a: [TI; MR] = [TI::zero(); MR];
            a.iter_mut().zip(&scales).for_each(|(x, s)| *x = *s * (pa.read_i4() - 8).as_());
            let b = std::slice::from_raw_parts(b.add(NR * (ik + 32 * bk)), NR);
            for i in 0..MR {
                for j in 0..NR {
                    ab[i][j] += a[i] * b[j].as_();
                }
            }
        }
    }
}

unsafe fn add_mat_mul_pq40_scales_at_end<const MR: usize, const NR: usize, TB, TI>(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[TI; NR]; MR],
) where
    TI: LADatum,
    f16: AsPrimitive<TI>,
    TB: AsPrimitive<TI>,
    i8: AsPrimitive<TI>,
{
    assert!(k % Q4_0.block_len() == 0);
    let len = (k * MR) / Q4_0.block_len() * Q4_0.block_bytes();
    let mut pa = NibbleReader::for_slice(std::slice::from_raw_parts(pa, len));
    let b = pb as *const TB;
    for bk in 0..k / 32 {
        let mut temp = [[TI::zero(); NR]; MR];
        for ik in 0..32 {
            let mut a: [TI; MR] = [TI::zero(); MR];
            a.iter_mut().for_each(|x| *x = (pa.read_i4() - 8).as_());
            let b = std::slice::from_raw_parts(b.add(NR * (ik + 32 * bk)), NR);
            for i in 0..MR {
                for j in 0..NR {
                    temp[i][j] += a[i] * b[j].as_();
                }
            }
        }
        for i in 0..MR {
            let scale = pa.read_f16().as_();
            for j in 0..NR {
                ab[i][j] += temp[i][j] * scale;
            }
        }
    }
}

unsafe fn add_unicast<const MR: usize, const NR: usize, TI, TO>(
    ab: &mut [[TI; NR]; MR],
    other: &OutputStoreKer,
) where
    TI: LADatum,
    TO: LADatum + AsPrimitive<TI>,
{
    for i in 0usize..MR {
        for j in 0usize..NR {
            let value: *const TO = other
                .ptr
                .offset(other.row_byte_stride * i as isize + other.col_byte_stride * j as isize)
                as _;
            ab[i].as_mut()[j] += (*value).as_();
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

unsafe fn store_float_t<const MR: usize, const NR: usize, TC, TI>(
    tile: &OutputStoreKer,
    ab: &[[TI; NR]; MR],
) where
    TC: Copy + 'static,
    TI: Copy + 'static + AsPrimitive<TC>,
{
    for i in 0usize..MR {
        for j in 0usize..NR {
            let loc: *mut TC = tile
                .ptr
                .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                as _;
            let val = ab[i].as_ref()[j].as_();
            *loc = val
        }
    }
}

#[inline(never)]
unsafe fn kernel<TI, const MR: usize, const NR: usize>(mut pnl: *const FusedKerSpec<TI>) -> isize
where
    TI: LADatum + ScaleShiftAndRound + AsPrimitive<TI>,
    TI: AsPrimitive<f16> + AsPrimitive<f32> + AsPrimitive<f64>,
    usize: AsPrimitive<TI>,
    f16: AsPrimitive<TI>,
    f32: AsPrimitive<TI>,
    f64: AsPrimitive<TI>,
    i8: AsPrimitive<TI>,
    i32: AsPrimitive<TI>,
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
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..MR {
                        for col in 0..NR {
                            ab[row][col] = *col_major.add(col * MR + row);
                        }
                    }
                }
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
                FusedKerSpec::AddUnicast(other) => {
                    if TI::datum_type().is_float() && other.item_size == 2 {
                        add_unicast::<MR, NR, TI, f16>(&mut ab, &other)
                    } else if TI::datum_type().is_float() && other.item_size == 4 {
                        add_unicast::<MR, NR, TI, f32>(&mut ab, &other)
                    } else if TI::datum_type().is_float() && other.item_size == 8 {
                        add_unicast::<MR, NR, TI, f64>(&mut ab, &other)
                    } else if TI::datum_type() == i32::datum_type() && other.item_size == 1 {
                        add_unicast::<MR, NR, TI, i8>(&mut ab, &other)
                    } else if TI::datum_type() == i32::datum_type() && other.item_size == 4 {
                        add_unicast::<MR, NR, TI, i32>(&mut ab, &other)
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
                        match packing {
                            0 => add_mat_mul::<MR, NR, TI, TI, TI>(pa, pb, k, &mut ab),
                            1 => add_mat_mul::<MR, NR, TI, f16, f16>(pa, pb, k, &mut ab),
                            2 => add_mat_mul::<MR, NR, TI, f32, f32>(pa, pb, k, &mut ab),
                            3 => add_mat_mul::<MR, NR, TI, f16, f32>(pa, pb, k, &mut ab),
                            4 => add_mat_mul::<MR, NR, TI, f32, f16>(pa, pb, k, &mut ab),
                            5 => add_mat_mul_pq40::<MR, NR, f16, TI>(pa, pb, k, &mut ab),
                            6 => add_mat_mul_pq40_scales_at_end::<MR, NR, f16, TI>(
                                pa, pb, k, &mut ab,
                            ),
                            7 => add_mat_mul_pq40::<MR, NR, f32, TI>(pa, pb, k, &mut ab),
                            _ => unreachable!(),
                        }
                    } else if TI::datum_type() == i32::datum_type() {
                        // transmute to allow using explicitly i3 in add_mat_mul generic params
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
                FusedKerSpec::Store(tile) => {
                    if TI::datum_type().is_float() {
                        match tile.item_size {
                            2 => store_float_t::<MR, NR, f16, _>(&tile, &ab),
                            4 => store_float_t::<MR, NR, f32, _>(&tile, &ab),
                            8 => store_float_t::<MR, NR, f64, _>(&tile, &ab),
                            _ => unimplemented!(),
                        }
                    } else {
                        match tile.item_size {
                            1 => store_t::<MR, NR, u8, _>(&tile, &ab),
                            2 => store_t::<MR, NR, u16, _>(&tile, &ab),
                            4 => store_t::<MR, NR, u32, _>(&tile, &ab),
                            8 => store_t::<MR, NR, u64, _>(&tile, &ab),
                            _ => unimplemented!(),
                        }
                    }
                }
            };
            pnl = pnl.add(1);
        }
    }
    0
}

fn pq40_r4() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 4, 0, false)
}

fn pq40_r4_se() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 4, 0, true)
}

// f16 kernels
MMMRustKernel!(kernel::<f16, 4, 4> => generic_f16_4x4<f16>(4,4)
    packing[1] = f16f16bis => |k| k.with_packing(f16::packing(4), f16::packing(4));
    packing[2] = f32f32 => |k| k.with_packing(f32::packing(4), f32::packing(4));
    packing[3] = f16f32 => |k| k.with_packing(f16::packing(4), f32::packing(4));
    packing[4] = f32f16 => |k| k.with_packing(f32::packing(4), f16::packing(4));
    packing[5] = q40f16 => |k| k.with_packing(pq40_r4(), f16::packing(4));
    packing[6] = q40f16se => |k| k.with_packing(pq40_r4_se(), f16::packing(4));
    packing[7] = q40f32 => |k| k.with_packing(pq40_r4(), f32::packing(4));
    quality(if has_fp16() { ImplementationQuality::Generic } else { ImplementationQuality::Dreadful })
    store(f32, f64)
);

MMMRustKernel! {kernel::<f16, 4, 1> => generic_f16_4x1<f16>(4,1)
    packing[1] = f16f16bis => |k| k.with_packing(f16::packing(4), f16::packing(1));
    packing[2] = f32f32 => |k| k.with_packing(f32::packing(4), f32::packing(1));
    packing[3] = f16f32 => |k| k.with_packing(f16::packing(4), f32::packing(1));
    packing[4] = f32f16 => |k| k.with_packing(f32::packing(4), f16::packing(1));
    packing[5] = q40f16 => |k| k.with_packing(pq40_r4(), f16::packing(1));
    packing[6] = q40f16se => |k| k.with_packing(pq40_r4_se(), f16::packing(1));
    packing[7] = q40f32 => |k| k.with_packing(pq40_r4(), f32::packing(1));
    quality(if has_fp16() { ImplementationQuality::Generic } else { ImplementationQuality::Dreadful })
    store(f32, f64)
}

// f32 kernels
MMMRustKernel!(kernel::<f32, 4, 4> => generic_f32_4x4<f32>(4,4)
    packing[1] = f16f16 => |k| k.with_packing(f16::packing(4), f16::packing(4));
    packing[2] = f32f32bis => |k| k.with_packing(f32::packing(4), f32::packing(4));
    packing[3] = f16f32 => |k| k.with_packing(f16::packing(4), f32::packing(4));
    packing[4] = f32f16 => |k| k.with_packing(f32::packing(4), f16::packing(4));
    packing[5] = q40f16 => |k| k.with_packing(pq40_r4(), f16::packing(4));
    packing[6] = q40f16se => |k| k.with_packing(pq40_r4_se(), f16::packing(4));
    packing[7] = q40f32 => |k| k.with_packing(pq40_r4(), f32::packing(4));
    quality(ImplementationQuality::Generic)
    store(f16, f64)
);
MMMRustKernel! {kernel::<f32, 4, 1> => generic_f32_4x1<f32>(4,1)
    packing[1] = f16f16 => |k| k.with_packing(f16::packing(4), f16::packing(1));
    packing[2] = f32f32bis => |k| k.with_packing(f32::packing(4), f32::packing(1));
    packing[3] = f16f32 => |k| k.with_packing(f16::packing(4), f32::packing(1));
    packing[4] = f32f16 => |k| k.with_packing(f32::packing(4), f16::packing(1));
    packing[5] = q40f16 => |k| k.with_packing(pq40_r4(), f16::packing(1));
    packing[6] = q40f16se => |k| k.with_packing(pq40_r4_se(), f16::packing(1));
    packing[7] = q40f32 => |k| k.with_packing(pq40_r4(), f32::packing(1));
    quality(ImplementationQuality::Generic)
    store(f16, f64)
}

// f64 kernels
MMMRustKernel!(kernel::<f64, 4, 4> => generic_f64_4x4<f64>(4,4)
    quality(ImplementationQuality::Generic)
    store(f16, f32));
MMMRustKernel!(kernel::<f64, 4, 1> => generic_f64_4x1<f64>(4,1)
    quality(ImplementationQuality::Generic)
    store(f16, f32));

// I32 kernels
MMMRustKernel! {kernel::<i32, 4, 4> => generic_i32_4x4<i32>(4,4)
    packing[1] = i8i8 => |k| k.with_packing(i8::packing(4), i8::packing(4));
    quality(ImplementationQuality::Generic)
    store(i8)
}

MMMRustKernel! {kernel::<i32, 4, 1> => generic_i32_4x1<i32>(4,1)
    packing[1] = i8i8 => |k| k.with_packing(i8::packing(4), i8::packing(1));
    quality(ImplementationQuality::Generic)
    store(i8)
}

// extra tests kernels
#[cfg(test)]
MMMRustKernel!(kernel::<f32, 3, 2> => generic_f32_3x2<f32>(3,2) store(f16, f64));

#[cfg(test)]
MMMRustKernel! {kernel::<i32, 3, 2> => generic_i32_3x2<i32>(3,2)
    packing[1] = i8i8 => |k| k.with_packing(i8::packing(3), i8::packing(2));
    store(i8)
}

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(generic_f16_4x4.mmm());
    ops.mmm_impls.push(generic_f16_4x1.mmm());
    ops.mmm_impls.push(generic_f32_4x4.mmm());
    ops.mmm_impls.push(generic_f32_4x1.mmm());
    ops.mmm_impls.push(generic_f64_4x4.mmm());
    ops.mmm_impls.push(generic_f64_4x1.mmm());
    ops.mmm_impls.push(generic_i32_4x4.mmm());
    ops.mmm_impls.push(generic_i32_4x1.mmm());
}

#[cfg(test)]
mod test {

    #[test]
    fn kits() {
        let mut ops = crate::generic();
        super::plug(&mut ops);
    }
}
