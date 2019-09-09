pub mod mat_mul;

pub use self::mat_mul::MatMul;
use crate::internal::*;
use num_traits::AsPrimitive;
use num_traits::Float;
use num_traits::Zero;

use super::binary::*;

bin_to_super_type!(add, Add, flip:commute,
     [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() + b);
bin_to_super_type!(sub, Sub, flip:flip_sub,
     [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);
#[inline]
bin_to_super_type!(mul, Mul, flip:commute,
     [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() * b);
bin_to_super_type!(div, Div,
     [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() / b);
bin_to_super_type!(rem, Rem,
     [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() % b);
bin_to_super_type!(min, Min, flip:commute,
     [f32, f64] => |c,a,b| *c = a.min(*b),
     [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.min(b));
bin_to_super_type!(max, Max, flip:commute,
     [f32, f64] => |c,a,b| *c = a.max(*b),
     [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.max(b));
bin_to_super_type!(pow, Pow,
     [f32, f64] => |c,a,b| *c = a.powf(*b));

fn flip_sub(_op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    let mut t = t.clone().into_tensor();
    fn negate<T: Datum + std::ops::Neg<Output = T>>(t: &mut Tensor) {
        t.as_slice_mut::<T>().unwrap().iter_mut().for_each(|p| *p = -p.clone());
    }
    (|t: &mut Tensor| -> TractResult<()> {
        dispatch_signed!(negate(t.datum_type())(t));
        Ok(())
    })(&mut t)
    .unwrap();
    Some(UnaryOp::new(Box::new(Add), Arc::new(t)))
}

element_map!(Abs, [f16, f32, i32], |x| x.abs());
element_map!(Exp, [f16, f32, f64], |x| x.exp());
element_map!(Ln, [f16, f32, f64], |x| x.ln());
element_map!(Sqrt, [f16, f32, f64], |x| x.sqrt());
element_map!(Recip, [f16, f32], |x| x.recip());
element_map!(Rsqrt, [f16, f32], |x| x.sqrt().recip());

element_map!(Ceil, [f16, f32, f64], |x| x.ceil());
element_map!(Floor, [f16, f32, f64], |x| x.floor());

element_map_with_params!(ScalarMinMax, [f16, f32, f64], { min: f32, max: f32 },
    fn eval_one<T>(clip: &ScalarMinMax, x:T) -> T
    where T: Datum+::num_traits::Float, f32: ::num_traits::AsPrimitive<T>
    {
        x.max(clip.max.as_()).min(clip.min.as_())
    }
);

element_map_with_params!(
    ScalarMin,
    [f32, f64],
    { min: f32 },
    fn eval_one<T>(sm: &ScalarMin, x: T) -> T
    where
        T: Datum + ::num_traits::Float,
        f32: ::num_traits::AsPrimitive<T>,
    {
        x.min(sm.min.as_())
    }
);

element_map_with_params!(
    ScalarMax,
    [f32, f64],
    { max: f32 },
    fn eval_one<T>(sm: &ScalarMax, x: T) -> T
    where
        T: Datum + ::num_traits::Float,
        f32: ::num_traits::AsPrimitive<T>,
    {
        x.max(sm.max.as_())
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
element_map_inplace!(Tanh, [f32], |xs| <f32 as FloatLike>::tanh().run(xs));
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
