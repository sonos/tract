use std::ffi::c_void;
use std::fmt;
use std::fmt::Debug;
use tract_data::internal::*;

#[derive(Clone, Copy, Debug)]
pub enum OutputStoreSpec {
    View {
        axes: Option<(usize, usize)>,
        mr: usize,
        nr: usize,
    },
    Strides {
        row_byte_stride: isize,
        row_item_stride: isize,
        col_byte_stride: isize,
        col_item_stride: isize,
        mr: usize,
        nr: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct OutputStore {
    pub(crate) ptr: *mut c_void,
    pub(crate) row_byte_stride: isize,
    pub(crate) col_byte_stride: isize,
    pub(crate) row_item_stride: isize,
    pub(crate) col_item_stride: isize,
    pub(crate) panel_row_byte_stride: isize,
    pub(crate) panel_col_byte_stride: isize,
    pub(crate) item_size: usize,
    pub(crate) item_count: usize,
    pub(crate) mr: usize,
}

impl OutputStoreSpec {
    #[inline]
    pub unsafe fn wrap(self: &Self, tensor: &TensorView) -> OutputStore {
        let (mr, nr, row_item_stride, col_item_stride, row_byte_stride, col_byte_stride) =
            self.compute_strides(tensor);
        OutputStore {
            ptr: tensor.as_ptr_unchecked::<u8>() as _,
            row_byte_stride,
            col_byte_stride,
            row_item_stride,
            col_item_stride,
            panel_row_byte_stride: row_byte_stride * mr as isize,
            panel_col_byte_stride: col_byte_stride * nr as isize,
            item_size: tensor.datum_type().size_of(),
            mr,
            item_count: tensor.len(),
        }
    }

    #[inline]
    unsafe fn compute_strides(
        &self,
        tensor: &TensorView,
    ) -> (usize, usize, isize, isize, isize, isize) {
        let size_of = tensor.datum_type().size_of() as isize;
        match self {
            OutputStoreSpec::View { axes, mr, nr, .. } => {
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
                (*mr, *nr, row_item_stride, col_item_stride, row_byte_stride, col_byte_stride)
            }
            OutputStoreSpec::Strides {
                row_byte_stride,
                col_byte_stride,
                col_item_stride,
                row_item_stride,
                mr,
                nr,
            } => (*mr, *nr, *row_item_stride, *col_item_stride, *row_byte_stride, *col_byte_stride),
        }
    }
}

impl OutputStore {
    #[inline]
    pub(super) unsafe fn tile_c(&self, down: usize, right: usize) -> OutputStoreKer {
        let (down, right) = (down as isize, right as isize);
        OutputStoreKer {
            ptr: self
                .ptr
                .offset(self.panel_row_byte_stride * down + self.panel_col_byte_stride * right)
                as *mut _,
            row_byte_stride: self.row_byte_stride,
            col_byte_stride: self.col_byte_stride,
            item_size: self.item_size,
        }
    }

    #[inline]
    pub fn item_size(&self) -> usize {
        self.item_size
    }

    #[inline]
    pub(super) unsafe fn set_from_tile(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        if self.item_size() == 1 {
            self.set_from_tile_t::<i8>(down, right, height, width, tile)
        } else {
            self.set_from_tile_t::<i32>(down, right, height, width, tile)
        }
    }

    #[inline]
    unsafe fn set_from_tile_t<T: Datum + Copy>(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        let tile = tile.ptr as *mut T;
        let dst = self.ptr.offset(
            (self.panel_row_byte_stride as usize * down
                + self.panel_col_byte_stride as usize * right) as isize,
        );
        for y in 0..height as isize {
            for x in 0..width as isize {
                let value = tile.offset(y + x * self.mr as isize);
                let dst =
                    dst.offset((y * self.row_byte_stride + x * self.col_byte_stride) as isize);
                *(dst as *mut T) = *value;
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug, Hash)]
pub enum InputStoreSpec {
    Packed(PackedStoreSpec),
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_byte_offsets: Vec<isize>, nr: usize },
}

#[derive(PartialEq, Clone, Copy, Debug, Hash)]
pub struct PackedStoreSpec {
    pub(crate) panel_bytes: usize,
}

impl PackedStoreSpec {
    #[inline]
    pub unsafe fn wrap(&self, tensor: &TensorView) -> PackedStore {
        PackedStore {
            ptr: tensor.as_ptr_unchecked::<u8>() as _,
            item_size: tensor.datum_type().size_of(),
            panel_bytes: self.panel_bytes as isize,
        }
    }
}

impl InputStoreSpec {
    #[inline]
    pub unsafe fn wrap(&self, tensor: &TensorView) -> InputStore {
        use InputStore::*;
        use InputStoreSpec as S;
        match self {
            S::Packed(PackedStoreSpec { panel_bytes }) => Packed(PackedStore {
                ptr: tensor.as_ptr_unchecked::<u8>() as _,
                item_size: tensor.datum_type().size_of(),
                panel_bytes: *panel_bytes as isize,
            }),
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
}

impl fmt::Display for InputStoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputStoreSpec::Packed { .. } => write!(fmt, "Packed"),
            InputStoreSpec::OffsetsAndPtrs { .. } => write!(fmt, "OffsetsAndPtrs"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum InputStore {
    Packed(PackedStore),
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: Vec<*const c_void>, nr: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct PackedStore {
    ptr: *const c_void,
    item_size: usize,
    panel_bytes: isize,
}

impl PackedStore {
    #[inline]
    pub(super) unsafe fn panel(&self, i: usize) -> PackedStoreKer {
        PackedStoreKer { ptr: self.ptr.offset(self.panel_bytes * i as isize) }
    }
}

impl InputStore {
    #[inline]
    pub(super) unsafe fn panel_b(&self, i: usize) -> InputStoreKer {
        match self {
            InputStore::Packed(packed) => InputStoreKer::Packed(packed.panel(i)),
            InputStore::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr, .. } => {
                InputStoreKer::OffsetsAndPtrs {
                    row_byte_offsets: *row_byte_offsets,
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum InputStoreKer {
    Packed(PackedStoreKer),
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const c_void },
}

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct PackedStoreKer {
    pub ptr: *const c_void,
}


#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct OutputStoreKer {
    pub ptr: *mut c_void,
    pub row_byte_stride: isize,
    pub col_byte_stride: isize,
    pub item_size: usize,
}
