mod data_formats;
mod reduce;
mod softmax;

pub use self::data_formats::{BaseDataShape, DataFormat, DataShape, SymDataShape};
pub use self::reduce::{Reduce, Reducer, expand_mean_of_squares};
pub use self::softmax::{Softmax, SoftmaxExp};

pub use crate::internal::*;

use tract_num_traits::AsPrimitive;

element_wise!(sigmoid, Sigmoid,
 [f16] => |_, xs| { (tract_linalg::ops().sigmoid_f16)().run(xs) },
 [f32] => |_, xs| { (tract_linalg::ops().sigmoid_f32)().run(xs) };
 q: [i8, u8, i32, i32] => |x: f32| 1.0 / (1.0+(-x).exp());
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(hard_swish, HardSwish,
[f16] => |_, xs| { xs.iter_mut().for_each(|x| *x = *x * f16::from_f32(0.0).max(f16::from_f32(1.0).min(f16::from_f32(1. / 6.) * *x + f16::from_f32(0.5)))); Ok(()) },
[f32] => |_, xs| { xs.iter_mut().for_each(|x| *x = *x * 0f32.max(1f32.min((1. / 6.) * *x + 0.5))); Ok(()) }
                                         );

element_wise!(leaky_relu, LeakyRelu { alpha: f32 },
 [f16] => |op, xs| { (tract_linalg::ops().leaky_relu_f16)().run_with_params(xs, f16::from_f32(op.alpha)) },
 [f32] => |op, xs| { (tract_linalg::ops().leaky_relu_f32)().run_with_params(xs, op.alpha) }
);
