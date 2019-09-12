pub mod mat_mul;

pub use self::mat_mul::MatMul;
use crate::internal::*;
use num_traits::AsPrimitive;
use num_traits::Float;
use num_traits::Zero;

use super::binary::*;
use super::unary::*;

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

unary!(abs, Abs, [f16, f32, i32] => |xs| xs.iter_mut().for_each(|x| *x = x.abs()));
unary!(exp, Exp, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.exp()));
unary!(ln, Ln, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.ln()));
unary!(sqrt, Sqrt, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.sqrt()));
unary!(recip, Recip, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.recip()));
unary!(rsqrt, Rsqrt, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.sqrt().recip()));

unary!(ceil, Ceil, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.ceil()));
unary!(floor, Floor, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.floor()));

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

unary!(cos, Cos, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.cos()));
unary!(sin, Sin, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.sin()));
unary!(tan, Tan, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.tan()));
unary!(acos, Acos, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.acos()));
unary!(asin, Asin, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.asin()));
unary!(atan, Atan, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.atan()));

unary!(cosh, Cosh, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.cosh()));
unary!(sinh, Sinh, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.sinh()));
element_map_inplace!(Tanh, [f32], |xs| <f32 as FloatLike>::tanh().run(xs));
unary!(acosh, Acosh, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.acosh()));
unary!(asinh, Asinh, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.asinh()));
unary!(atanh, Atanh, [f16, f32, f64] => |xs| xs.iter_mut().for_each(|x| *x = x.atanh()));

unary!(neg, Neg, [i8, i16, i32, i64, f16, f32, f64, TDim] => |xs| xs.iter_mut().for_each(|x| *x = -*x));

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
