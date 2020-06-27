mod arg_max_min;
mod data_formats;
mod reduce;

pub use self::arg_max_min::ArgMaxMin;
pub use self::data_formats::{BaseDataShape, DataFormat, DataShape};
pub use self::reduce::{Reduce, Reducer};

pub use crate::internal::*;

element_wise!(sigmoid, Sigmoid, [f32] => |_, xs| {
    (tract_linalg::ops().sigmoid_f32)().run(xs);
    Ok(())
};
    cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);
