use crate::model::TfOpRegister;
use tract_core::internal::*;

mod concatv2;
mod expand_dims;
mod fill;
mod gather;
mod pack;
mod pad;
mod range;
mod slice;
mod squeeze;
mod strided_slice;
mod transpose;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", expand_dims::build);
    reg.insert("Fill", fill::fill);
    reg.insert("GatherNd", gather::gather_nd);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Range", range::range);
    reg.insert("Reshape", |_, _|Ok(Box::new(::tract_core::ops::array::Reshape::new())));
    reg.insert("Shape", |_, _| Ok(Box::new(::tract_core::ops::array::Shape::new(DatumType::I32))));
    reg.insert("Slice", |_, _| Ok(Box::new(slice::Slice)));
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("StridedSlice", strided_slice::build);
    reg.insert("Tile", |_, _| Ok(Box::new(::tract_core::ops::array::Tile)));
    reg.insert("Transpose", transpose::transpose);
}
