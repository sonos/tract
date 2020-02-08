/// # Operators on array and shapes
///
/// ## "Valid" reshaping operators
///
/// ### Both ONNX and TF
///
/// * Squeeze, unary (with or without an axis list, and... TF consider the empty list as
/// an absent list)
/// * Reshape, binary (input, shape as a tensor)
///
/// ### ONNX only
///
/// * Unsqueeze, unary, with required list of axes (referring to output)
/// * (Expand is a broadcasting operators, it does not beling here)
///
/// ### TF Only
///
/// * ExpandDims, binary (input, axis list)
///
/// ### Ours
///
/// * AddDims, just like ONNX's unsqueeze (Unsqueeze actually instantiate AddDims)
/// * RmDims, like Squeeze but with a mandatory axis list as an attribute.
///     Squeeze can always reduce to RmDims after inference.
///
/// ## Slicing and Upsampling
///
/// ### TF
///
/// * StridedSlice does everything
///
/// ### ONNX
///
/// * [Slice](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice),
///     unary, attr are: begins, ends, and optional axes remapping them
/// * [Upsample](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample),
///     unary, attrs are scales (floats) and mode of interpolation (nearest or
///     linear). not impl.
/// * DynamicSlice, experimental, not impl
///
/// ### Ours
///
/// * Slice, unary, mandatory attrs are begin and end.
mod broadcast;
pub(crate) mod concat;
mod constant_of_shape;
mod crop;
mod flatten;
mod gather;
mod pad;
mod permute_axes;
mod reshape;
mod shape;
mod size;
mod slice;
mod split;
mod strided_slice;
mod tile;

pub use self::broadcast::MultiBroadcastTo;
pub use self::concat::{Concat, ConcatSlice};
pub use self::constant_of_shape::ConstantOfShape;
pub use self::crop::Crop;
pub use self::flatten::Flatten;
pub use self::gather::Gather;
pub use self::pad::{Pad, PadMode};
pub use self::permute_axes::PermuteAxes;
pub use self::reshape::{FiniteReshape, Reshape, TypedReshape};
pub use self::shape::Shape;
pub use self::size::Size;
pub use self::slice::Slice;
pub use self::split::Split;
pub use self::strided_slice::StridedSlice;
pub use self::tile::Tile;
