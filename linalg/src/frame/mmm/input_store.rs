use std::ffi::c_void;
use std::fmt;
use tract_data::internal::*;

#[derive(PartialEq, Clone, Debug, Hash)]
pub enum InputStoreSpec {
    Packed(PackedStoreSpec),
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_byte_offsets: Vec<isize>, nr: usize },
}

#[derive(PartialEq, Clone, Copy, Debug, Hash)]
pub struct PackedStoreSpec {
    pub(crate) panel_bytes: usize,
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

impl PackedStore {
    #[inline]
    pub(super) unsafe fn panel(&self, i: usize) -> PackedStoreKer {
        PackedStoreKer { ptr: self.ptr.offset(self.panel_bytes * i as isize) }
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
