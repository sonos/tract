mod data_formats;
mod reduce;
mod softmax;

use tract_data::itertools::izip;

pub use self::data_formats::{BaseDataShape, DataFormat, DataShape, SymDataShape};
pub use self::reduce::{Reduce, Reducer};
pub use self::softmax::Softmax;

pub use crate::internal::*;

element_wise!(sigmoid, Sigmoid, 
    [f16] => |_, xs| {
        let mut fs:Vec<f32> = xs.iter().map(|x| x.to_f32()).collect();
        (tract_linalg::ops().sigmoid_f32)().run(&mut *fs)?;
        izip!(xs, fs).for_each(|(x, f)| *x = f16::from_f32(f));
        Ok(())
    },
    [f32] => |_, xs| { (tract_linalg::ops().sigmoid_f32)().run(xs) };
    cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(leaky_relu, LeakyRelu { #[educe(Hash(method = "hash_f32"))] alpha: f32 },
    [f32] => |op, xs| { xs.iter_mut().for_each(|x| *x *= if *x < 0. { op.alpha } else { 1.0 }); Ok(()) }
);
