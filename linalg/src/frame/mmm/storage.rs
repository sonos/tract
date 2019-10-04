use std::fmt::Debug;

#[derive(PartialEq, Clone, Debug)]
pub enum MatrixStoreSpec {
    Packed { panel_len: usize },
    Strides { row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_byte_offsets: Vec<isize>, nr: usize },
    VecStride { byte_stride: isize, mr: usize, nr: usize },
}

impl MatrixStoreSpec {
    pub unsafe fn wrap<T: Copy + Debug>(&self, ptr: *const T) -> MatrixStore<T> {
        match self {
            MatrixStoreSpec::Packed { panel_len } => {
                MatrixStore::Packed { ptr, panel_len: *panel_len }
            }
            MatrixStoreSpec::Strides { row_byte_stride, col_byte_stride, mr, nr } => {
                MatrixStore::Strides {
                    ptr,
                    row_byte_stride: *row_byte_stride,
                    col_byte_stride: *col_byte_stride,
                    mr: *mr,
                    nr: *nr,
                }
            }
            MatrixStoreSpec::VecStride { byte_stride, mr, nr } => {
                MatrixStore::VecStride { byte_stride: *byte_stride, mr: *mr, nr: *nr, ptr }
            }
            MatrixStoreSpec::OffsetsAndPtrs { row_byte_offsets, col_byte_offsets, nr } => {
                let col_ptrs: Vec<_> =
                    col_byte_offsets.iter().map(|&i| (ptr as *const u8).offset(i) as _).collect();
                MatrixStore::OffsetsAndPtrs { col_ptrs, row_byte_offsets, nr: *nr }
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum MatrixStore<'a, T>
where
    T: Copy + Debug,
{
    Strides { ptr: *const T, row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
    Packed { ptr: *const T, panel_len: usize },
    OffsetsAndPtrs { row_byte_offsets: &'a [isize], col_ptrs: Vec<*const T>, nr: usize },
    VecStride { ptr: *const T, byte_stride: isize, mr: usize, nr: usize },
}

impl<'a, T> MatrixStore<'a, T>
where
    T: Copy + Debug,
{
    pub(super) unsafe fn panel_a(&self, i: usize) -> PanelStore<T> {
        match self {
            MatrixStore::Packed { ptr, panel_len } => {
                PanelStore::Packed { ptr: ptr.offset((panel_len * i) as isize) }
            }
            _ => unimplemented!(),
        }
    }

    pub(super) unsafe fn panel_b(&self, nr: usize, i: usize, n: usize) -> PanelStore<T> {
        match self {
            MatrixStore::Packed { ptr, panel_len } => {
                if nr * i + 1 == n {
                    PanelStore::VecStride {
                        ptr: ptr.offset((panel_len * i) as isize),
                        byte_stride: (nr * std::mem::size_of::<T>()) as isize,
                    }
                } else {
                    PanelStore::Packed { ptr: ptr.offset((panel_len * i) as isize) }
                }
            }
            MatrixStore::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr } => {
                PanelStore::OffsetsAndPtrs {
                    row_byte_offsets: row_byte_offsets.as_ptr(),
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            MatrixStore::VecStride { ptr, byte_stride, .. } => {
                PanelStore::VecStride { ptr: *ptr, byte_stride: *byte_stride }
            }
            _ => unimplemented!(),
        }
    }

    pub(super) fn tile_c(&self, down: usize, right: usize) -> PanelStore<T> {
        match self {
            MatrixStore::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                PanelStore::Strides {
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

    pub(super) unsafe fn set_from_tile(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &[T],
    ) {
        match self {
            MatrixStore::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                for y in 0..height {
                    for x in 0..width {
                        let ptr = ((*ptr as isize)
                            + (*row_byte_stride as usize * (down * *mr + y)
                                + *col_byte_stride as usize * (right * *nr + x))
                                as isize) as *mut T;
                        let value = *tile.get_unchecked(y * *nr + x);
                        *ptr = value;
                    }
                }
            }
            MatrixStore::VecStride { ptr, byte_stride, mr, nr } => {
                for y in 0..height {
                    let ptr =
                        ((*ptr as isize) + (*byte_stride * (down * *mr + y) as isize)) as *mut T;
                    let value = *tile.get_unchecked(y * *nr);
                    *ptr = value;
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum PanelStore<T>
where
    T: Copy + Debug,
{
    Strides { ptr: *mut T, row_byte_stride: isize, col_byte_stride: isize },
    Packed { ptr: *const T },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const T },
    VecStride { ptr: *const T, byte_stride: isize },
}
