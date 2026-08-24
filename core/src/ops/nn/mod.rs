mod data_formats;
pub mod gelu_approximate;
pub mod gelu_exact;
pub mod grid_sample;
mod reduce;
pub mod resize;
pub mod rms_norm;
pub mod silu;
mod softmax;

pub use self::data_formats::{BaseDataShape, DataFormat, DataShape, SymDataShape};
pub use self::gelu_approximate::GeluApproximate;
pub use self::gelu_exact::GeluExact;
pub use self::grid_sample::{GridSample, InterpolationMode, PaddingMode};
pub use self::reduce::{Reduce, Reducer, expand_mean_of_squares};
pub use self::resize::{CoordTransformer, Interpolator, Nearest, Resize};
pub use self::rms_norm::RmsNorm;
pub use self::silu::Silu;
pub use self::softmax::{Softmax, SoftmaxKind};

pub use crate::internal::*;

use tract_linalg::routines::{Func, ew_f16, ew_f16_param, ew_f32, ew_f32_param};
use tract_num_traits::AsPrimitive;

element_wise!(sigmoid, Sigmoid,
 [f16] => |_, xs| { ew_f16(Func::Sigmoid)?.run(xs) },
 [f32] => |_, xs| { ew_f32(Func::Sigmoid)?.run(xs) };
 q: [i8, u8, i32, i32] => |x: f32| 1.0 / (1.0+(-x).exp());
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))};
 declutter: silu::detect_silu
);

element_wise!(hard_swish, HardSwish,
[f16] => |_, xs| { xs.iter_mut().for_each(|x| *x = *x * f16::from_f32(0.0).max(f16::from_f32(1.0).min(f16::from_f32(1. / 6.) * *x + f16::from_f32(0.5)))); Ok(()) },
[f32] => |_, xs| { ew_f32(Func::Hardswish)?.run(xs) }
                                         );

element_wise!(leaky_relu, LeakyRelu { alpha: f32 },
 [f16] => |op, xs| { ew_f16_param(Func::LeakyRelu)?.run_with_params(xs, f16::from_f32(op.alpha)) },
 [f32] => |op, xs| { ew_f32_param(Func::LeakyRelu)?.run_with_params(xs, op.alpha) }
);
