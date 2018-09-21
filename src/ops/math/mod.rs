use ops::prelude::*;

pub mod add_n;

element_map!(Abs, [f32, i32], |x| x.abs());

element_map!(Ceil, [f32, f64], |x| x.ceil());
element_map!(Floor, [f32, f64], |x| x.floor());

element_map!(Cos, [f32, f64], |x| x.cos());
element_map!(Sin, [f32, f64], |x| x.sin());
element_map!(Tan, [f32, f64], |x| x.tan());
element_map!(Acos, [f32, f64], |x| x.acos());
element_map!(Asin, [f32, f64], |x| x.asin());
element_map!(Atan, [f32, f64], |x| x.atan());

element_map!(Neg, [i32, f32, TDim], |x| -x);

element_bin!(Add, [i32, f32, TDim], |mut a, b| { a += &b; a });
element_bin!(Sub, [i32, f32, TDim], |mut a, b| { a -= &b; a });
element_bin!(Mul, [i32, f32, TDim], |mut a, b| { a *= &b; a });
element_bin!(Div, [i32, f32, TDim], |mut a, b| { a /= &b; a });
element_bin!(Rem, [i32, f32, TDim], |mut a, b| { a %= &b; a });

element_map!(Rsqrt, [f32], |x| x.sqrt().recip());
element_map!(Tanh, [f32], |x| x.tanh());

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
