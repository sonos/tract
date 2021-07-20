use crate::internal::*;

mod conv;
mod deconv;
mod pools;

register_all_mod!(conv, deconv, pools);
