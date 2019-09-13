mod arg_max_min;
mod data_formats;
mod global_pools;
mod layer_max;
mod lrn;
mod reduce;

pub use self::arg_max_min::ArgMaxMin;
pub use self::data_formats::{BaseDataShape, DataFormat, DataShape};
pub use self::global_pools::{GlobalAvgPool, GlobalLpPool, GlobalMaxPool};
pub use self::layer_max::{LayerHardmax, LayerLogSoftmax, LayerSoftmax};
pub use self::lrn::Lrn;
pub use self::reduce::{Reduce, Reducer};

use num_traits::{ AsPrimitive, Float};

pub use crate::internal::*;

unary!(softplus, Softplus, [f32] => |_, xs| xs.iter_mut().for_each(|x| *x = (x.exp() + 1.0).ln()));
unary!(softsign, Softsign, [f32] => |_, xs| xs.iter_mut().for_each(|x| *x = *x / (x.abs() + 1.0)));
unary!(sigmoid, Sigmoid, [f32] => |_, xs| f32::sigmoid().run(xs));

unary!(elu, Elu { alpha: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.elu(e.alpha); })
);

unary!(hard_sigmoid, HardSigmoid { alpha: f32, beta: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.hard_sigmoid(e.alpha, e.beta); })
);

unary!(leaky_relu, LeakyRelu { alpha: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.leaky_relu(e.alpha); })
);

unary!(parametric_softplus, ParametricSoftplus { alpha: f32, beta: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.parametric_softplus(e.alpha, e.beta); })
);

unary!(scaled_tanh, ScaledTanh { alpha: f32, beta: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.scaled_tanh(e.alpha, e.beta); })
);

unary!(selu, Selu { alpha: f32, gamma: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.selu(e.alpha, e.gamma); })
);

unary!(threshold_relu, ThresholdRelu { alpha: f32 },
    [f32, f64] => |e, xs| xs.iter_mut().for_each(|x| { *x = x.threshold_relu(e.alpha); })
);

trait Activations {
    fn elu(self, alpha: f32) -> Self;
    fn hard_sigmoid(self, alpha: f32, beta: f32) -> Self;
    fn leaky_relu(self, alpha: f32) -> Self;
    fn parametric_softplus(self, alpha: f32, beta:f32) -> Self;
    fn scaled_tanh(self, alpha: f32, beta:f32) -> Self;
    fn selu(self, alpha: f32, gamma:f32) -> Self;
    fn threshold_relu(self, alpha: f32) -> Self;
}

impl<T> Activations for T
where T: Datum + Float,
      f32: AsPrimitive<T>,
{
    fn elu(self, alpha: f32) -> Self {
        if self < 0.0.as_() {
            alpha.as_() * (self.exp() - 1.0.as_())
        } else {
            self
        }
    }
    fn hard_sigmoid(self, alpha:f32, beta:f32) -> Self {
        (alpha.as_() * self + beta.as_()).min(1.0.as_()).max(0.0.as_())
    }
    fn leaky_relu(self, alpha: f32) -> Self {
        if self < 0.0.as_() {
            alpha.as_() * self
        } else {
            self
        }
    }
    fn parametric_softplus(self, alpha: f32, beta:f32) -> Self {
        alpha.as_() * ((beta.as_() * self).exp() + 1.0.as_()).ln()
    }
    fn scaled_tanh(self, alpha: f32, beta:f32) -> Self {
        alpha.as_() * (beta.as_() * self).tanh()
    }
    fn selu(self, alpha: f32, gamma:f32) -> Self {
        if self < 0.0.as_() {
            gamma.as_() * (alpha.as_() * self.exp() - alpha.as_())
        } else {
            gamma.as_() * self
        }
    }
    fn threshold_relu(self, alpha: f32) -> Self {
        if self <= alpha.as_() {
            0.0.as_()
        } else {
            self
        }
    }
}

