use ops::prelude::*;

element_map!(Abs, [f32, i32], |x| x.abs());

element_map!(Exp, [f32, f64], |x| x.exp());
element_map!(Ln, [f32, f64], |x| x.ln());
element_map!(Sqrt, [f32, f64], |x| x.sqrt());
element_map!(Recip, [f32], |x| x.recip());
element_map!(Rsqrt, [f32], |x| x.sqrt().recip());

element_map!(Ceil, [f32, f64], |x| x.ceil());
element_map!(Floor, [f32, f64], |x| x.floor());

element_map!(Cos, [f32, f64], |x| x.cos());
element_map!(Sin, [f32, f64], |x| x.sin());
element_map!(Tan, [f32, f64], |x| x.tan());
element_map!(Acos, [f32, f64], |x| x.acos());
element_map!(Asin, [f32, f64], |x| x.asin());
element_map!(Atan, [f32, f64], |x| x.atan());

element_map!(Neg, [i32, f32, TDim], |x| -x);

element_bin!(Add, [i32, f32, TDim] { |a, b| a + b });
element_bin!(Sub, [i32, f32, TDim] { |a, b| a - b });
element_bin!(Mul, [i32, f32, TDim] { |a, b| a * b });
element_bin!(Div, [i32, f32, TDim] { |a, b| a / b });
element_bin!(Rem, [i32, f32, TDim] { |a, b| a % b });
element_bin!(Pow, match
     f32 => f32 { |a:f32, b| a.powf(b) },
     f64 => f64 { |a:f64, b| a.powf(b) }
);

element_map!(Tanh, [f32], |x| x.tanh());

fn fcmp<F: ::num::Float>(a:&F,b:&F) -> ::std::cmp::Ordering {
    a.partial_cmp(b).unwrap()
}

element_nary!(AddN, [f32, f64] { |v:&[_]| v.iter().sum() });
element_nary!(MaxN, match
  f32 => f32 { |v:&[f32]| v.iter().cloned().max_by(fcmp).unwrap() },
  f64 => f64 { |v:&[f64]| v.iter().cloned().max_by(fcmp).unwrap() }
);
element_nary!(MinN, match
  f32 => f32 { |v:&[f32]| v.iter().cloned().min_by(fcmp).unwrap() },
  f64 => f64 { |v:&[f64]| v.iter().cloned().min_by(fcmp).unwrap() }
);
element_nary!(MeanN, match
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
