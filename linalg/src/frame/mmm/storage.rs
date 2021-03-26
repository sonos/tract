use std::ffi::c_void;
use std::fmt;
use std::fmt::Debug;
use tract_data::internal::*;

#[derive(PartialEq, Clone, Debug, Hash)]
pub enum MatrixStoreSpec {
    View { axes: Option<(usize, usize)> },
    Packed { panel_bytes: usize },
    Strides { row_item_stride: isize, col_item_stride: isize },
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_byte_offsets: Vec<isize>, nr: usize },
    VecStride { item_stride: isize, mr: usize, nr: usize },
}

impl MatrixStoreSpec {
    pub unsafe fn wrap<'t>(&self, tensor: &'t TensorView) -> MatrixStore<'_, 't> {
        MatrixStore::new(self, tensor)
    }
}

impl fmt::Display for MatrixStoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixStoreSpec::View { .. } => write!(fmt, "ViewAxis"),
            MatrixStoreSpec::Packed { .. } => write!(fmt, "Packed"),
            MatrixStoreSpec::Strides { .. } => write!(fmt, "Strides"),
            MatrixStoreSpec::OffsetsAndPtrs { .. } => write!(fmt, "OffsetsAndPtrs"),
            MatrixStoreSpec::VecStride { .. } => write!(fmt, "VecStrides"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum MatrixStore<'s, 't> {
    Strides {
        ptr: *mut c_void,
        row_item_stride: isize,
        col_item_stride: isize,
        row_byte_stride: isize,
        col_byte_stride: isize,
        item_size: usize,
        item_count: usize,
    },
    Packed {
        ptr: *const c_void,
        item_size: usize,
        panel_bytes: isize,
    },
    OffsetsAndPtrs {
        row_byte_offsets: *const isize,
        col_ptrs: Vec<*const c_void>,
        nr: usize,
    },
    VecStride {
        ptr: *const c_void,
        item_stride: isize,
        byte_stride: isize,
        item_size: usize,
        item_count: usize,
    },
    _Phantom(&'s (), &'t ()),
}

impl<'s, 't> MatrixStore<'s, 't> {
    unsafe fn new(spec: &'s MatrixStoreSpec, tensor: &'t TensorView) -> MatrixStore<'s, 't> {
        use MatrixStore::*;
        use MatrixStoreSpec as S;
        let item_size = tensor.datum_type().size_of();
        match spec {
            S::View { .. } | S::Strides { .. } => {
                let (row_item_stride, col_item_stride, row_byte_stride, col_byte_stride) =
                    Self::compute_strides(spec, tensor);
                Strides {
                    ptr: tensor.as_ptr_unchecked::<u8>() as _,
                    row_item_stride,
                    col_item_stride,
                    row_byte_stride,
                    col_byte_stride,
                    item_size,
                    item_count: tensor.len(),
                }
            }
            S::VecStride { .. } => {
                let (item_stride, _, byte_stride, _) = Self::compute_strides(spec, tensor);
                VecStride {
                    ptr: tensor.as_ptr_unchecked::<u8>() as _,
                    item_stride,
                    byte_stride,
                    item_size,
                    item_count: tensor.len(),
                }
            }
            S::Packed { panel_bytes } => Packed {
                ptr: tensor.as_ptr_unchecked::<u8>() as _,
                item_size: tensor.datum_type().size_of(),
                panel_bytes: *panel_bytes as isize,
            },
            S::OffsetsAndPtrs { row_byte_offsets, col_byte_offsets, nr } => OffsetsAndPtrs {
                row_byte_offsets: row_byte_offsets.as_ptr(),
                col_ptrs: col_byte_offsets
                    .iter()
                    .map(|offset| tensor.as_ptr_unchecked::<u8>().offset(*offset) as _)
                    .collect(),
                nr: *nr,
            },
        }
    }

    pub(super) unsafe fn panel_a(&self, i: usize) -> PanelStore {
        match self {
            MatrixStore::Packed { ptr, panel_bytes, .. } => {
                PanelStore::Packed { ptr: ptr.offset(panel_bytes * i as isize) }
            }
            _ => unimplemented!(),
        }
    }

    pub(super) unsafe fn panel_b(&self, nr: usize, i: usize, n: usize) -> PanelStore {
        match self {
            MatrixStore::Packed { ptr, panel_bytes, item_size } => {
                if nr * i + 1 == n {
                    PanelStore::VecStride {
                        ptr: ptr.offset(panel_bytes * i as isize),
                        byte_stride: (nr * *item_size) as isize,
                        item_size: *item_size,
                    }
                } else {
                    PanelStore::Packed { ptr: ptr.offset(panel_bytes * i as isize) }
                }
            }
            MatrixStore::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr, .. } => {
                PanelStore::OffsetsAndPtrs {
                    row_byte_offsets: *row_byte_offsets,
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            MatrixStore::VecStride { ptr, byte_stride, item_size, .. } => PanelStore::VecStride {
                ptr: *ptr,
                byte_stride: *byte_stride,
                item_size: *item_size,
            },
            _ => unimplemented!(),
        }
    }

    unsafe fn compute_strides(
        spec: &MatrixStoreSpec,
        tensor: &TensorView,
    ) -> (isize, isize, isize, isize) {
        let size_of = tensor.datum_type().size_of() as isize;
        match spec {
            MatrixStoreSpec::View { axes } => {
                let (m_axis, n_axis) = if let Some(axes) = axes {
                    axes.clone()
                } else {
                    let rank = tensor.rank();
                    (rank - 2, rank - 1)
                };
                let tensor_strides = tensor.strides();
                let row_item_stride = *tensor_strides.get_unchecked(m_axis);
                let col_item_stride = *tensor_strides.get_unchecked(n_axis);
                let row_byte_stride = row_item_stride * size_of;
                let col_byte_stride = col_item_stride * size_of;
                (row_item_stride, col_item_stride, row_byte_stride, col_byte_stride)
            }
            MatrixStoreSpec::Strides { row_item_stride, col_item_stride } => {
                let row_byte_stride = row_item_stride * size_of;
                let col_byte_stride = col_item_stride * size_of;
                (*row_item_stride, *col_item_stride, row_byte_stride, col_byte_stride)
            }
            MatrixStoreSpec::VecStride { item_stride, .. } => {
                (*item_stride, 0, *item_stride * size_of, 0)
            }
            _ => panic!(),
        }
    }

    pub(super) unsafe fn tile_c(
        &self,
        down: usize,
        right: usize,
        mr: usize,
        nr: usize,
    ) -> PanelStore {
        let (down, right, mr, nr) = (down as isize, right as isize, mr as isize, nr as isize);
        match self {
            MatrixStore::Strides { ptr, row_byte_stride, col_byte_stride, item_size, .. } => {
                PanelStore::Strides {
                    ptr: ptr.offset(row_byte_stride * down * mr + col_byte_stride * right * nr)
                        as *mut _,
                    row_byte_stride: *row_byte_stride,
                    col_byte_stride: *col_byte_stride,
                    item_size: *item_size,
                }
            }
            MatrixStore::VecStride { ptr, byte_stride, item_size, .. } => PanelStore::VecStride {
                ptr: ptr.offset(byte_stride * down * mr) as *mut _,
                byte_stride: *byte_stride,
                item_size: *item_size,
            },
            _ => unimplemented!(),
        }
    }

    pub(super) unsafe fn set_from_tile<T: Datum + Copy>(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &PanelStore,
        mr: usize,
        nr: usize,
    ) {
        let tile = if let PanelStore::Strides { ptr, .. } = tile {
            *ptr as *mut T
        } else {
            panic!("tile is expected to be in PanelStrides form")
        };
        match self {
            MatrixStore::Strides { ptr, row_byte_stride, col_byte_stride, .. } => {
                let dst = ptr.offset(
                    (*row_byte_stride as usize * (down * mr)
                        + *col_byte_stride as usize * (right * nr)) as isize,
                );
                for y in 0..height as isize {
                    for x in 0..width as isize {
                        let value = tile.offset(y + x * mr as isize);
                        let dst =
                            dst.offset((y * *row_byte_stride + x * *col_byte_stride) as isize);
                        *(dst as *mut T) = *value;
                    }
                }
            }
            MatrixStore::VecStride { ptr, byte_stride, .. } => {
                let dst = ptr.offset(*byte_stride * (down * mr) as isize);
                for y in 0..height as isize {
                    let value = *tile.offset(y as isize);
                    let dst = dst.offset(y as isize * *byte_stride);
                    *(dst as *mut T) = value;
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum PanelStore {
    Strides { ptr: *mut c_void, row_byte_stride: isize, col_byte_stride: isize, item_size: usize },
    Packed { ptr: *const c_void },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const c_void },
    VecStride { ptr: *const c_void, byte_stride: isize, item_size: usize },
}
