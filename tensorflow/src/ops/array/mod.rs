use tract_core::ops::prelude::*;
use crate::tfpb::node_def::NodeDef;

mod concatv2;
mod expand_dims;
mod fill;
mod pack;
mod pad;
mod reshape;
mod squeeze;
mod strided_slice;

pub fn register_all_ops(reg: &mut OpRegister<NodeDef>) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", expand_dims::build);
    reg.insert("Identity", |_| {
        Ok(Box::new(::tract_core::ops::identity::Identity::default()))
    });
    reg.insert("Fill", fill::fill);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Reshape", reshape::reshape);
    reg.insert("Shape", |_| {
        Ok(Box::new(::tract_core::ops::array::Shape::new(
            DatumType::I32,
        )))
    });
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("StridedSlice", strided_slice::build);
}
