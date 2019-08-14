pub mod gemm;
pub mod mat_mul;

pub use self::gemm::Gemm;
pub use self::mat_mul::MatMul;
use crate::internal::*;
use num_traits::AsPrimitive;
use num_traits::Float;
use num_traits::Zero;

pub use crate::ops::binary::{ add, sub, mul, div, rem, min, max, pow };

element_map!(Abs, [f16, f32, i32], |x| x.abs());
element_map!(Exp, [f16, f32, f64], |x| x.exp());
element_map!(Ln, [f16, f32, f64], |x| x.ln());
element_map!(Sqrt, [f16, f32, f64], |x| x.sqrt());
element_map!(Recip, [f16, f32], |x| x.recip());
element_map!(Rsqrt, [f16, f32], |x| x.sqrt().recip());

element_map!(Ceil, [f16, f32, f64], |x| x.ceil());
element_map!(Floor, [f16, f32, f64], |x| x.floor());

element_map_with_params!(Clip, [f16, f32, f64], { min: f32, max: f32 },
    fn eval_one<T>(clip: &Clip, x:T) -> T
    where T: Datum+::num_traits::Float, f32: ::num_traits::AsPrimitive<T>
    {
        x.max(clip.min.as_()).min(clip.max.as_())
    }
);

element_map!(Cos, [f16, f32, f64], |x| x.cos());
element_map!(Sin, [f16, f32, f64], |x| x.sin());
element_map!(Tan, [f16, f32, f64], |x| x.tan());
element_map!(Acos, [f16, f32, f64], |x| x.acos());
element_map!(Asin, [f16, f32, f64], |x| x.asin());
element_map!(Atan, [f16, f32, f64], |x| x.atan());

element_map!(Cosh, [f16, f32, f64], |x| x.cosh());
element_map!(Sinh, [f16, f32, f64], |x| x.sinh());
element_map!(Tanh, [f16, f32, f64], |x| x.tanh());
element_map!(Acosh, [f16, f32, f64], |x| x.acosh());
element_map!(Asinh, [f16, f32, f64], |x| x.asinh());
element_map!(Atanh, [f16, f32, f64], |x| x.atanh());

element_map!(Neg, [i8, i16, i32, i64, f16, f32, f64, TDim], |x| -x);

element_map!(Sign, match
     f16 => { |a:f16| if a.is_zero() { (0.0).into() } else { a.signum()} },
     f32 => { |a:f32| if a == 0.0 { 0.0 } else { a.signum()} },
     f64 => { |a:f64| if a == 0.0 { 0.0 } else { a.signum()} }
);

element_map_move!(IsNan, match
     f16 => bool { |a:f16| a.is_nan() },
     f32 => bool { |a:f32| a.is_nan() },
     f64 => bool { |a:f64| a.is_nan() }
);

fn fcmp<F: ::num_traits::Float>(a: &F, b: &F) -> ::std::cmp::Ordering {
    a.partial_cmp(b).unwrap()
}

element_nary!(AddN, [f16, f32, f64] { |v:&[_]| v.iter().sum() });
element_nary!(MaxN, match
  f16 => f16 { |v:&[f16]| v.iter().cloned().max_by(fcmp).unwrap() },
  f32 => f32 { |v:&[f32]| v.iter().cloned().max_by(fcmp).unwrap() },
  f64 => f64 { |v:&[f64]| v.iter().cloned().max_by(fcmp).unwrap() }
);
element_nary!(MinN, match
  f16 => f16 { |v:&[f16]| v.iter().cloned().min_by(fcmp).unwrap() },
  f32 => f32 { |v:&[f32]| v.iter().cloned().min_by(fcmp).unwrap() },
  f64 => f64 { |v:&[f64]| v.iter().cloned().min_by(fcmp).unwrap() }
);
element_nary!(MeanN, match
  f16 => f16 { |v:&[f16]| v.iter().cloned().sum::<f16>() / f16::from(v.len() as f32) },
  f32 => f32 { |v:&[f32]| v.iter().cloned().sum::<f32>() / v.len() as f32 },
  f64 => f64 { |v:&[f64]| v.iter().cloned().sum::<f64>() / v.len() as f64 }
);

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    #[test]
    fn mul() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a * b, arr2(&[[1., 0.], [0., 0.]]));
    }
    #[test]
    fn dot() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a.dot(&b), arr2(&[[1., 0.], [3., 0.]]));
    }
}
