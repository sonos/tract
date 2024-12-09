use crate::MetalTensor;
use metal::{ComputeCommandEncoderRef, MTLResourceUsage, MTLSize};
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
    fn dipatch_non_uniform_threadgroup(&self, size: MTLSize);
}

/*
MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2 /* = 10 */) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (dim0 >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim2 >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == pow2) {
      break;
    }
  }
  return MTL::Size{1ul << pows[0], 1ul << pows[1], 1ul << pows[2]};
}
*/

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

    fn dipatch_non_uniform_threadgroup(&self, grid_size: MTLSize) {
        let threadgroup_size = compute_non_uniform_threadgroup_size(grid_size);
        self.dispatch_threads(grid_size, threadgroup_size)
    }
}

pub fn compute_non_uniform_threadgroup_size(size: MTLSize) -> MTLSize {
    let MTLSize { width: x, height: y, depth: z } = size;

    let (mut x_pow, mut y_pow, mut z_pow) = (0, 0, 0);

    let max_pow = 10; // 1024
    let mut sum = 0;
    loop {
        let prev = sum;
        if x >= (1 << (x_pow + 1)) {
            x_pow += 1;
            sum += 1;
        }

        if sum == max_pow {
            break;
        }

        if y >= (1 << (y_pow + 1)) {
            y_pow += 1;
            sum += 1;
        }

        if sum == max_pow {
            break;
        }

        if z >= (1 << (z_pow + 1)) {
            z_pow += 1;
            sum += 1;
        }

        if sum == max_pow {
            break;
        }
        if prev == sum {
            break;
        }
    }

    MTLSize { width: 1 << x_pow, height: 1 << y_pow, depth: 1 << z_pow }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_compute_non_uniform_threadgroup_size() -> TractResult<()> {
        assert_eq!(
            compute_non_uniform_threadgroup_size(MTLSize { width: 1024, height: 1, depth: 1 }),
            MTLSize { width: 1024, height: 1, depth: 1 }
        );

        assert_eq!(
            compute_non_uniform_threadgroup_size(MTLSize { width: 30, height: 60, depth: 1 }),
            MTLSize { width: 16, height: 32, depth: 1 }
        );

        assert_eq!(
            compute_non_uniform_threadgroup_size(MTLSize { width: 1000, height: 16, depth: 1 }),
            MTLSize { width: 64, height: 16, depth: 1 }
        );

        Ok(())
    }
}
