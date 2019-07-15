use num_traits::Zero;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::PackA;
use super::PackB;

#[repr(C, usize)]
#[derive(PartialEq, Clone, Debug)]
pub enum StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq,
{
    Strides { ptr: *const T, row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
    Packed { ptr: *const T, panel_len: isize },
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_ptrs: Vec<*const T>, nr: usize },
}

impl<T> StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq,
{
    unsafe fn panel_a(&self, i: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                TileStorageSpec::Packed { ptr: ptr.offset(panel_len * i as isize) }
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn panel_b(&self, i: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                TileStorageSpec::Packed { ptr: ptr.offset(panel_len * i as isize) }
            }
            StorageSpec::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr } => {
                TileStorageSpec::OffsetsAndPtrs {
                    row_byte_offsets: row_byte_offsets.as_ptr(),
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            _ => unimplemented!(),
        }
    }

    fn tile(&self, down: usize, right: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                TileStorageSpec::Strides {
                    ptr: ((*ptr as isize)
                        + (*row_byte_stride as usize * down * mr
                            + *col_byte_stride as usize * right * nr)
                            as isize) as *mut T,
                    row_byte_stride: *row_byte_stride,
                    col_byte_stride: *col_byte_stride,
                }
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn set(&mut self, row: usize, col: usize, val: T) {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, .. } => {
                *(((*ptr as isize)
                    + (*row_byte_stride as usize * row + *col_byte_stride as usize * col) as isize)
                    as *mut T) = val;
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TileOp<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq,
    K: TilingKer<T>,
{
    pub m: usize,
    pub n: usize,
    pub k: usize,
    phantom: PhantomData<(K, T)>,
}

impl<K, T> TileOp<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq,
    K: TilingKer<T>,
{
    pub fn a_pack(&self) -> PackA<T> {
        PackA::new(self.k, self.m, K::mr(), K::alignment_bytes_packed_a())
    }

    pub fn b_pack(&self) -> PackB<T> {
        PackB::new(self.k, self.n, K::nr(), K::alignment_bytes_packed_b())
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub unsafe fn a_from_packed(&self, ptr: *const T) -> StorageSpec<T> {
        StorageSpec::Packed { ptr, panel_len: (self.k * K::mr()) as isize }
    }

    pub unsafe fn b_from_data_and_offsets(
        &self,
        data: *const T,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> StorageSpec<T> {
        let mut col_ptrs: Vec<_> = cols_offsets.iter().map(|&co| data.offset(co)).collect();
        let wanted = (col_ptrs.len() + K::nr() - 1) / K::nr() * K::nr();
        while col_ptrs.len() < wanted {
            col_ptrs.push(col_ptrs[col_ptrs.len() - 1]);
        }
        StorageSpec::OffsetsAndPtrs {
            col_ptrs,
            row_byte_offsets: rows_offsets
                .iter()
                .map(|&ro| ro * std::mem::size_of::<T>() as isize)
                .collect(),
            nr: K::nr(),
        }
    }

    pub unsafe fn c_from_data_and_strides(
        &self,
        data: *const T,
        row_stride: isize,
        col_stride: isize,
    ) -> StorageSpec<T> {
        StorageSpec::Strides {
            ptr: data,
            row_byte_stride: row_stride * std::mem::size_of::<T>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<T>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    pub unsafe fn run(&self, a: &StorageSpec<T>, b: &StorageSpec<T>, c: &mut StorageSpec<T>) {
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let k = self.k;
        let n = self.n;
        let tmpc = vec![T::zero(); mr * nr];
        let ref mut tmp_tile = self.c_from_data_and_strides(tmpc.as_ptr(), nr as isize, 1);
        let linear = LinearSpec::Mul { k };
        let linear = (&linear) as *const LinearSpec;
        let non_linear = NonLinearSpec::Done;
        let non_linear = (&non_linear) as *const NonLinearSpec<T>;
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                let ref b = b.panel_b(ib);
                let ref tile_c = c.tile(ia, ib);
                K::kernel(&TileOpSpec { a: a as _, b: b as _, c: tile_c as _, linear, non_linear });
            }
            if n % nr != 0 {
                let ref b = b.panel_b(n / nr);
                let ref tmp_tile_c = tmp_tile.tile(0, 0);
                K::kernel(&TileOpSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                for y in 0..mr {
                    for x in 0..(n % nr) {
                        c.set(mr * ia + y, x + n / nr * nr, tmpc[y * nr + x])
                    }
                }
            }
        }
        if m % mr != 0 {
            let ref a = a.panel_a(m / mr);
            let ref tmp_tile_c = tmp_tile.tile(0, 0);
            for ib in 0..n / nr {
                let ref b = b.panel_b(ib);
                K::kernel(&TileOpSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                for y in 0..(m % mr) {
                    for x in 0..nr {
                        c.set(m / mr * mr + y, x + ib * nr, tmpc[y * nr + x])
                    }
                }
            }
            if n % nr != 0 {
                let ref b = b.panel_b(n / nr);
                K::kernel(&TileOpSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                for y in 0..(m % mr) {
                    for x in 0..(n % nr) {
                        c.set(m / mr * mr + y, x + n / nr * nr, tmpc[y * nr + x])
                    }
                }
            }
        }
    }
}

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct TileOpSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    pub a: *const TileStorageSpec<T>,
    pub b: *const TileStorageSpec<T>,
    pub c: *const TileStorageSpec<T>,
    pub linear: *const LinearSpec,
    pub non_linear: *const NonLinearSpec<T>,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum LinearSpec {
    Mul { k: usize },
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum NonLinearSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Done,
    Min(T),
    Max(T),
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum TileStorageSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Strides { ptr: *mut T, row_byte_stride: isize, col_byte_stride: isize },
    Packed { ptr: *const T },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const T },
}

pub trait TilingKer<T>: Copy + Clone + Debug + Send + Sync
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    #[inline(always)]
    fn name() -> &'static str;
    #[inline(always)]
    fn kernel(op: &TileOpSpec<T>) -> isize;
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize;
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize;
}
