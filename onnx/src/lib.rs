#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num_integer;
extern crate num_traits;
extern crate protobuf;
#[allow(unused_imports)]
#[macro_use]
extern crate tract_core;
extern crate tract_linalg;

pub mod model;
pub mod ops;
pub mod pb;
pub mod pb_helpers;
pub mod tensor;

pub use self::model::for_path;
pub use self::model::for_reader;

use tract_core::model::Framework;
use crate::pb::NodeProto;

pub fn onnx() -> Framework<NodeProto> {
    let mut reg = Framework::default();
    ops::register_all_ops(&mut reg);
    reg
}

