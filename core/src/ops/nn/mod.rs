mod arg_max_min;
mod data_formats;
mod global_pools;
mod reduce;

pub use self::arg_max_min::ArgMaxMin;
pub use self::data_formats::{BaseDataShape, DataFormat, DataShape};
pub use self::global_pools::{GlobalAvgPool, GlobalLpPool, GlobalMaxPool};
pub use self::reduce::{Reduce, Reducer};

use num_traits::{AsPrimitive, Float};

pub use crate::internal::*;

element_wise!(softplus, Softplus, [f32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = (x.exp() + 1.0).ln());
    Ok(())
});

element_wise!(softsign, Softsign, [f32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = *x / (x.abs() + 1.0));
    Ok(())
});

element_wise!(sigmoid, Sigmoid, [f32] => |_, xs| {
    (tract_linalg::ops().sigmoid_f32)().run(xs);
    Ok(())
};
    cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(elu,
    Elu {
        #[educe(Hash(method = "hash_f32"))] alpha: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.elu(e.alpha); });
        Ok(())
});

element_wise!(hard_sigmoid,
    HardSigmoid {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32,
        #[educe(Hash(method = "hash_f32"))]
        beta: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.hard_sigmoid(e.alpha, e.beta); });
        Ok(())
});

element_wise!(leaky_relu,
    LeakyRelu {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.leaky_relu(e.alpha); });
        Ok(())
});

element_wise!(parametric_softplus,
    ParametricSoftplus {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32,
        #[educe(Hash(method = "hash_f32"))]
        beta: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.parametric_softplus(e.alpha, e.beta); });
        Ok(())
});

element_wise!(scaled_tanh,
    ScaledTanh {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32,
        #[educe(Hash(method = "hash_f32"))]
        beta: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.scaled_tanh(e.alpha, e.beta); });
        Ok(())
});

element_wise!(selu,
    Selu {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32,
        #[educe(Hash(method = "hash_f32"))]
        gamma: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.selu(e.alpha, e.gamma); });
        Ok(())
});

element_wise!(threshold_relu,
    ThresholdRelu {
        #[educe(Hash(method = "hash_f32"))]
        alpha: f32
    },
    [f32, f64] => |e, xs| {
        xs.iter_mut().for_each(|x| { *x = x.threshold_relu(e.alpha); });
        Ok(())
});

trait Activations {
    fn elu(self, alpha: f32) -> Self;
    fn hard_sigmoid(self, alpha: f32, beta: f32) -> Self;
    fn leaky_relu(self, alpha: f32) -> Self;
    fn parametric_softplus(self, alpha: f32, beta: f32) -> Self;
    fn scaled_tanh(self, alpha: f32, beta: f32) -> Self;
    fn selu(self, alpha: f32, gamma: f32) -> Self;
    fn threshold_relu(self, alpha: f32) -> Self;
}

impl<T> Activations for T
where
    T: Datum + Float,
    f32: AsPrimitive<T>,
{
    fn elu(self, alpha: f32) -> Self {
        if self < 0.0.as_() {
            alpha.as_() * (self.exp() - 1.0.as_())
        } else {
            self
        }
    }
    fn hard_sigmoid(self, alpha: f32, beta: f32) -> Self {
        (alpha.as_() * self + beta.as_()).min(1.0.as_()).max(0.0.as_())
    }
    fn leaky_relu(self, alpha: f32) -> Self {
        if self < 0.0.as_() {
            alpha.as_() * self
        } else {
            self
        }
    }
    fn parametric_softplus(self, alpha: f32, beta: f32) -> Self {
        alpha.as_() * ((beta.as_() * self).exp() + 1.0.as_()).ln()
    }
    fn scaled_tanh(self, alpha: f32, beta: f32) -> Self {
        alpha.as_() * (beta.as_() * self).tanh()
    }
    fn selu(self, alpha: f32, gamma: f32) -> Self {
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
