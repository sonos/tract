use crate::internal::*;

mod concat;
mod pad;
mod slice;

register_all_mod!(concat, pad, slice);
