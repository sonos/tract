use std::alloc::Layout;
use std::fmt::Debug;
use tract_data::internal::*;

pub trait InputStoreSpec: dyn_clone::DynClone + Debug + Send + Sync {
    fn wrap(&self, view: &TensorView) -> Box<dyn InputStore>;
}
dyn_clone::clone_trait_object!(InputStoreSpec);

pub trait InputStore: dyn_clone::DynClone + Debug {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout>;
    fn panel(&self, i: usize, buffer: Option<*mut u8>) -> *const u8;
}
dyn_clone::clone_trait_object!(InputStore);

#[derive(Debug, Clone)]
pub struct PrepackedSpec {
    pub panel_bytes: usize,
}

impl InputStoreSpec for PrepackedSpec {
    fn wrap(&self, view: &TensorView) -> Box<dyn InputStore> {
        let ptr = unsafe { view.as_ptr_unchecked() };
        Box::new(Prepacked { ptr, panel_bytes: self.panel_bytes as isize })
    }
}

#[derive(Debug, Clone)]
pub struct Prepacked {
    pub ptr: *const u8,
    pub panel_bytes: isize,
}

impl InputStore for Prepacked {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel(&self, i: usize, _buffer: Option<*mut u8>) -> *const u8 {
        unsafe { self.ptr.offset(self.panel_bytes * i as isize) }
    }
}
