use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuantValue, Q4_0};

pub fn check_strides_validity(shape: TVec<usize>, strides: TVec<isize>) -> TractResult<()> {
    let mut zipped_shape_strides: Vec<_> = shape.into_iter().zip(strides).collect();
    zipped_shape_strides.sort_by_key(|&(_, stride)| stride);

    let mut prev_stride = 1;
    for (dim, stride) in zipped_shape_strides {
        ensure!((stride == prev_stride) || (dim == 1), "Invalid strides");
        prev_stride *= dim as isize;
    }
    Ok(())
}

pub fn as_q40_tensor(a: &Tensor) -> Option<&BlockQuantValue> {
    a.to_scalar::<Opaque>().ok().and_then(|od| {
        od.downcast_ref::<BlockQuantValue>().and_then(|bqv| {
            if bqv.fact.format.same_as(&Q4_0) {
                Some(bqv)
            } else {
                None
            }
        })
    })
}
