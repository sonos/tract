use crate::MetalTensor;
use metal::{ComputeCommandEncoderRef, MTLResourceUsage};
use tract_core::internal::*;

pub trait EncoderExt {
    fn set_metal_tensor(&self, idx: u64, t: &MetalTensor, usage: MTLResourceUsage);
    fn set_metal_tensor_with_offset(
        &self,
        idx: u64,
        t: &MetalTensor,
        offset: u64,
        usage: MTLResourceUsage,
    );
    fn set_tensor(&self, idx: u64, t: &Tensor);
    fn set_slice<T: Copy>(&self, idx: u64, data: &[T]);
}

impl EncoderExt for &ComputeCommandEncoderRef {
    fn set_metal_tensor(&self, idx: u64, t: &MetalTensor, usage: MTLResourceUsage) {
        self.set_buffer(idx, Some(t.metal()), t.metal_offset());
        self.use_resource(t.metal(), usage);
    }

    fn set_metal_tensor_with_offset(
        &self,
        idx: u64,
        t: &MetalTensor,
        offset: u64,
        usage: MTLResourceUsage,
    ) {
        self.set_buffer(idx, Some(t.metal()), t.metal_offset::<u64>() + offset);
        self.use_resource(t.metal(), usage);
    }

    fn set_tensor(&self, idx: u64, t: &Tensor) {
        self.set_bytes(idx, (t.datum_type().size_of() * t.len()) as _, unsafe {
            t.as_ptr_unchecked::<u8>()
        } as *const _);
    }

    fn set_slice<T: Copy>(&self, idx: u64, data: &[T]) {
        self.set_bytes(idx, std::mem::size_of_val(data) as _, data.as_ptr() as *const _)
    }
}
