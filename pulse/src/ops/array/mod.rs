use crate::internal::*;

mod broadcast;
mod concat;
mod gather;
mod mask;
mod pad;
mod range;
mod reshape;
mod slice;

register_all_mod!(broadcast, concat, gather, pad, range, reshape, slice);
