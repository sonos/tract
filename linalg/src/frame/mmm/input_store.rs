use std::alloc::Layout;
use std::fmt;
use std::ops::Range;
use tract_data::internal::*;

use crate::frame::Packer;

pub trait VirtualInputSpec: dyn_clone::DynClone + std::fmt::Debug + Sync + Send {
    fn wrap(&self, view: &TensorView) -> Box<dyn VirtualInput>;
}
dyn_clone::clone_trait_object!(VirtualInputSpec);

pub trait VirtualInput: dyn_clone::DynClone + std::fmt::Debug + Sync + Send {
    fn input(&self, packer: &Packer, packed_output: *mut u8, k: Range<usize>, mn: Range<usize>);
}
dyn_clone::clone_trait_object!(VirtualInput);

#[derive(Clone, Debug)]
pub enum InputStoreSpec {
    Prepacked{panel_bytes: usize},
    VirtualPacking { packer: Packer, func: Box<dyn VirtualInputSpec>, k: usize },
}

impl InputStoreSpec {
    #[inline]
    pub unsafe fn wrap(&self, tensor: &TensorView) -> InputStore {
        use InputStoreSpec as S;
        match self {
            S::Prepacked { panel_bytes } => InputStore::Packed {
                ptr: tensor.as_ptr_unchecked::<u8>() as _,
                panel_bytes: *panel_bytes as isize,
            },
            S::VirtualPacking { packer, func, k } => InputStore::VirtualPacking {
                packer: packer.clone(),
                input: func.wrap(tensor),
                k: *k,
                dt: tensor.datum_type(),
            },
        }
    }
}

impl fmt::Display for InputStoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputStoreSpec::Prepacked { .. } => write!(fmt, "Packed"),
            InputStoreSpec::VirtualPacking { .. } => write!(fmt, "VirtualPacking"),
        }
    }
}


#[derive(Clone, Debug)]
pub enum InputStore {
    Packed {
        ptr: *const u8,
        panel_bytes: isize,
    },
    VirtualPacking {
        packer: Packer,
        input: Box<dyn VirtualInput>,
        k: usize,
        dt: DatumType, // TODO discard me ?
    },
}

impl InputStore {
    pub(super) unsafe fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        match self {
            InputStore::Packed { .. } => None,
            InputStore::VirtualPacking { packer, dt, k, .. } => {
                let size = packer.single_panel_len(*k) * dt.size_of();
                let align = packer.alignment();
                Some(Layout::from_size_align_unchecked(size, align))
            }
        }
    }

    #[inline]
    pub(super) unsafe fn panel(&self, i: usize, buffer: Option<*const u8>) -> *const u8 {
        match self {
            InputStore::Packed { ptr, panel_bytes } => ptr.offset(panel_bytes * i as isize),
            InputStore::VirtualPacking { packer, input, k, .. } => {
                input.input(packer, buffer.unwrap() as _, 0..*k, packer.r * i..packer.r * (i + 1));
                buffer.unwrap()
            }
        }
    }
}
