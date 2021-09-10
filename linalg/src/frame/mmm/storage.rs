use std::ffi::c_void;
use std::fmt;
use std::fmt::Debug;
use tract_data::internal::*;

#[derive(PartialEq, Clone, Debug, Hash)]
pub enum MatrixStoreSpec {
    View {
        axes: Option<(usize, usize)>,
        mr: usize,
        nr: usize,
    },
    Packed {
        panel_bytes: usize,
    },
    Strides {
        row_byte_stride: isize,
        row_item_stride: isize,
        col_byte_stride: isize,
        col_item_stride: isize,
        mr: usize,
        nr: usize,
    },
    OffsetsAndPtrs {
        row_byte_offsets: Vec<isize>,
        col_byte_offsets: Vec<isize>,
        nr: usize,
    },
}

impl MatrixStoreSpec {
    #[inline]
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
        }
    }
}

#[derive(Clone, Debug)]
pub enum MatrixStore<'s, 't> {
    Strides {
        ptr: *mut c_void,
        row_byte_stride: isize,
        col_byte_stride: isize,
        row_item_stride: isize,
        col_item_stride: isize,
        panel_row_byte_stride: isize,
        panel_col_byte_stride: isize,
        item_size: usize,
        item_count: usize,
        mr: usize,
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
    _Phantom(&'s (), &'t ()),
}

impl<'s, 't> MatrixStore<'s, 't> {
    unsafe fn new(spec: &'s MatrixStoreSpec, tensor: &'t TensorView) -> MatrixStore<'s, 't> {
        use MatrixStore::*;
        use MatrixStoreSpec as S;
        match spec {
            S::View { mr, nr, .. } | S::Strides { mr, nr, .. } => {
                let (row_item_stride, col_item_stride, row_byte_stride, col_byte_stride) =
                    Self::compute_strides(spec, tensor);
                Strides {
                    ptr: tensor.as_ptr_unchecked::<u8>() as _,
                    row_byte_stride,
                    col_byte_stride,
                    row_item_stride,
                    col_item_stride,
                    panel_row_byte_stride: row_byte_stride * *mr as isize,
                    panel_col_byte_stride: col_byte_stride * *nr as isize,
                    item_size: tensor.datum_type().size_of(),
                    mr: *mr,
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

    #[inline]
    pub(super) unsafe fn panel_a(&self, i: usize) -> PanelStore {
        match self {
            MatrixStore::Packed { ptr, panel_bytes, .. } => {
                PanelStore::Packed { ptr: ptr.offset(panel_bytes * i as isize) }
            }
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub(super) unsafe fn panel_b(&self, i: usize) -> PanelStore {
        match self {
            MatrixStore::Packed { ptr, panel_bytes, .. } => {
                PanelStore::Packed { ptr: ptr.offset(panel_bytes * i as isize) }
            }
            MatrixStore::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr, .. } => {
                PanelStore::OffsetsAndPtrs {
                    row_byte_offsets: *row_byte_offsets,
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn compute_strides(
        spec: &MatrixStoreSpec,
        tensor: &TensorView,
        ) -> (isize, isize, isize, isize) {
        let size_of = tensor.datum_type().size_of() as isize;
        match spec {
            MatrixStoreSpec::View { axes, .. } => {
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
            MatrixStoreSpec::Strides {
                row_byte_stride,
                col_byte_stride,
                col_item_stride,
                row_item_stride,
                ..
            } => (*row_item_stride, *col_item_stride, *row_byte_stride, *col_byte_stride),
            _ => panic!(),
        }
    }

    #[inline]
    pub(super) unsafe fn tile_c(&self, down: usize, right: usize) -> Tile {
        let (down, right) = (down as isize, right as isize);
        match self {
            MatrixStore::Strides {
                ptr,
                row_byte_stride,
                col_byte_stride,
                item_size,
                panel_row_byte_stride,
                panel_col_byte_stride,
                ..
            } => Tile {
                ptr: ptr.offset(panel_row_byte_stride * down + panel_col_byte_stride * right)
                    as *mut _,
                    row_byte_stride: *row_byte_stride,
                    col_byte_stride: *col_byte_stride,
                    item_size: *item_size,
            },
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub(super) unsafe fn set_from_tile(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &Tile)
    {
        if self.item_size() == 1 {
            self.set_from_tile_t::<i8>(down, right, height, width, tile)
        } else {
            self.set_from_tile_t::<i32>(down, right, height, width, tile)
        }
    }

    #[inline]
    unsafe fn set_from_tile_t<T: Datum + Copy>(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &Tile,
    ) {
        let tile = tile.ptr as *mut T;
        match self {
            MatrixStore::Strides {
                ptr,
                row_byte_stride,
                col_byte_stride,
                panel_row_byte_stride,
                panel_col_byte_stride,
                mr,
                ..
            } => {
                let dst = ptr.offset(
                    (*panel_row_byte_stride as usize * down
                     + *panel_col_byte_stride as usize * right) as isize,
                     );
                for y in 0..height as isize {
                    for x in 0..width as isize {
                        let value = tile.offset(y + x * *mr as isize);
                        let dst =
                            dst.offset((y * *row_byte_stride + x * *col_byte_stride) as isize);
                        *(dst as *mut T) = *value;
                    }
                }
            }
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub fn item_size(&self) -> usize {
        match self {
            MatrixStore::Strides { item_size, .. } => *item_size,
            _ => unimplemented!(),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum PanelStore {
    Strides(Tile),
    Packed { ptr: *const c_void },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const c_void },
}

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Tile {
    pub ptr: *mut c_void,
    pub row_byte_stride: isize,
    pub col_byte_stride: isize,
    pub item_size: usize,
}
