mod data_formats;
mod reduce;
mod softmax;

pub use self::data_formats::{BaseDataShape, DataFormat, DataShape, SymDataShape};
pub use self::reduce::{Reduce, Reducer};
pub use self::softmax::Softmax;

pub use crate::internal::*;

element_wise!(sigmoid, Sigmoid,
 [f16] => |_, xs| { (tract_linalg::ops().sigmoid_f16)().run(xs) },
 [f32] => |_, xs| { (tract_linalg::ops().sigmoid_f32)().run(xs) };
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(hard_swish, HardSwish,
[f32] => |_, xs| { xs.iter_mut().for_each(|x| *x = *x * 0f32.max(1f32.min((1. / 6.) * *x + 0.5))); Ok(()) }
                                         );

element_wise!(leaky_relu, LeakyRelu { alpha: f32 },
 [f16] => |op, xs| { (tract_linalg::ops().leaky_relu_f16)().run_with_params(xs, f16::from_f32(op.alpha)) },
 [f32] => |op, xs| { (tract_linalg::ops().leaky_relu_f32)().run_with_params(xs, op.alpha) }
);
