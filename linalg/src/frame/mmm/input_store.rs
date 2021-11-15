use std::{alloc::Layout, fmt};
use tract_data::internal::*;

use crate::frame::Packer;
#[derive(PartialEq, Clone, Debug, Hash)]
pub enum InputStoreSpec {
    Prepacked(PackedStoreSpec),
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_byte_offsets: Vec<isize>, nr: usize },
    LatePacking { packer: Packer, k_axis: usize, mn_axis: usize },
}

#[derive(PartialEq, Clone, Copy, Debug, Hash)]
pub struct PackedStoreSpec {
    pub(crate) panel_bytes: usize,
}

impl InputStoreSpec {
    #[inline]
    pub unsafe fn wrap(&self, tensor: &TensorView) -> TractResult<InputStore> {
        use InputStore::*;
        use InputStoreSpec as S;
        match self {
            S::Prepacked(PackedStoreSpec { panel_bytes }) => Ok(Packed(PackedStore {
                ptr: tensor.as_ptr_unchecked::<u8>() as _,
                item_size: tensor.datum_type().size_of(),
                panel_bytes: *panel_bytes as isize,
            })),
            S::OffsetsAndPtrs { row_byte_offsets, col_byte_offsets, nr } => Ok(OffsetsAndPtrs {
                row_byte_offsets: row_byte_offsets.as_ptr(),
                col_ptrs: col_byte_offsets
                    .iter()
                    .map(|offset| tensor.as_ptr_unchecked::<u8>().offset(*offset) as _)
                    .collect(),
                nr: *nr,
            }),
            S::LatePacking { packer, k_axis, mn_axis } => Ok(InputStore::LatePacking {
                packer: packer.clone(),
                ptr: tensor.as_ptr_unchecked::<u8>() as _,
                dt: tensor.datum_type(),
                k: tensor.shape()[*k_axis],
                mn: tensor.shape()[*mn_axis],
                k_stride: tensor.strides()[*k_axis],
                mn_stride: tensor.strides()[*mn_axis],
            }),
        }
    }
}

impl fmt::Display for InputStoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputStoreSpec::Prepacked { .. } => write!(fmt, "Packed"),
            InputStoreSpec::OffsetsAndPtrs { .. } => write!(fmt, "OffsetsAndPtrs"),
            InputStoreSpec::LatePacking { .. } => write!(fmt, "LatePacking"),
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
    OffsetsAndPtrs {
        row_byte_offsets: *const isize,
        col_ptrs: Vec<*const u8>,
        nr: usize,
    },
    LatePacking {
        packer: Packer,
        ptr: *const u8,
        dt: DatumType,
        k: usize,
        mn: usize,
        k_stride: isize,
        mn_stride: isize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct PackedStore {
    ptr: *const u8,
    item_size: usize,
    panel_bytes: isize,
}

impl InputStore {
    pub(super) unsafe fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        if let InputStore::LatePacking { packer, dt, k, .. } = self {
            let size = packer.single_panel_len(*k) * dt.size_of();
            let align = packer.alignment();
            Some(Layout::from_size_align_unchecked(size, align))
        } else {
            None
        }
    }

    #[inline]
    pub(super) unsafe fn panel_b(&self, i: usize, buffer: Option<*const u8>) -> InputStoreKer {
        match self {
            InputStore::Packed(packed) => InputStoreKer::Packed(packed.panel(i)),
            InputStore::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr, .. } => {
                InputStoreKer::OffsetsAndPtrs {
                    row_byte_offsets: *row_byte_offsets,
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            InputStore::LatePacking { packer, ptr, dt, k, mn, mn_stride, k_stride } => {
                dispatch_copy!(Packer::pack_t(dt)(
                    packer,
                    buffer.unwrap() as _,
                    *ptr as _,
                    *mn,
                    *k_stride,
                    *mn_stride,
                    0..*k,
                    packer.r * i..packer.r * (i + 1)
                ));
                InputStoreKer::Packed(PackedStoreKer { ptr: buffer.unwrap() })
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
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const u8 },
}

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct PackedStoreKer {
    pub ptr: *const u8,
}
