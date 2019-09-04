use std::fmt::Debug;
use std::ops::{Add, Mul};

use num_traits::Zero;

#[repr(C, usize)]
#[derive(PartialEq, Clone, Debug)]
pub enum StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
{
    Strides { ptr: *const T, row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
    Packed { ptr: *const T, panel_len: usize },
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_ptrs: Vec<*const T>, nr: usize },
    VecStride { ptr: *const T, byte_stride: isize, mr: usize, nr: usize },
}

impl<T> StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
{
    pub(super) unsafe fn panel_a(&self, i: usize) -> StorageKerSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                StorageKerSpec::Packed { ptr: ptr.offset((panel_len * i) as isize) }
            }
            _ => unimplemented!(),
        }
    }

    pub(super) unsafe fn panel_b(&self, nr: usize, i: usize, n: usize) -> StorageKerSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                if nr * i + 1 == n {
                    StorageKerSpec::VecStride {
                        ptr: ptr.offset((panel_len * i) as isize),
                        byte_stride: (nr * std::mem::size_of::<T>()) as isize,
                    }
                } else {
                    StorageKerSpec::Packed { ptr: ptr.offset((panel_len * i) as isize) }
                }
            }
            StorageSpec::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr } => {
                StorageKerSpec::OffsetsAndPtrs {
                    row_byte_offsets: row_byte_offsets.as_ptr(),
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            StorageSpec::VecStride { ptr, byte_stride, .. } => {
                StorageKerSpec::VecStride { ptr: *ptr, byte_stride: *byte_stride }
            }
            _ => unimplemented!(),
        }
    }

    pub(super) fn mmm(&self, down: usize, right: usize) -> StorageKerSpec<T> {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                StorageKerSpec::Strides {
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

    pub(super) unsafe fn set_from_mmm(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        mmm: &[T],
    ) {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                for y in 0..height {
                    for x in 0..width {
                        let ptr = ((*ptr as isize)
                            + (*row_byte_stride as usize * (down * *mr + y)
                                + *col_byte_stride as usize * (right * *nr + x))
                                as isize) as *mut T;
                        let value = *mmm.get_unchecked(y * *nr + x);
                        *ptr = value;
                    }
                }
            }
            StorageSpec::VecStride { ptr, byte_stride, mr, nr } => {
                for y in 0..height {
                    let ptr =
                        ((*ptr as isize) + (*byte_stride * (down * *mr + y) as isize)) as *mut T;
                    let value = *mmm.get_unchecked(y * *nr);
                    *ptr = value;
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum StorageKerSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Strides { ptr: *mut T, row_byte_stride: isize, col_byte_stride: isize },
    Packed { ptr: *const T },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const T },
    VecStride { ptr: *const T, byte_stride: isize },
}
