use crate::internal::*;

mod affine_trim;
mod broadcast;
mod concat;
mod cumsum;
mod mask;
mod pad;
mod range;
mod reshape;
mod slice;

pub use tract_pulse_opl::ops::AffineChunkTrim;

register_all_mod!(affine_trim, broadcast, concat, cumsum, pad, range, reshape, slice);
