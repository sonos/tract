#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate educe;
#[macro_use]
extern crate log;
extern crate serde;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate typetag;

pub mod model;
mod ops;
pub mod parser;

pub use model::Kaldi;
pub use model::KaldiProtoModel;

pub fn kaldi() -> Kaldi {
    let mut kaldi = Kaldi::default();
    ops::register_all_ops(&mut kaldi.op_register);
    kaldi
}
