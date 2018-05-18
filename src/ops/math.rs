use Result;
use super::{Op, OpRegister};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Abs", Abs::build);
    reg.insert("Add", add);
    reg.insert("BiasAdd", add);
    reg.insert("Div", div);
    reg.insert("Mul", mul);
    reg.insert("FloorMod", rem);
    reg.insert("Rsqrt", Rsqrt::build);
    reg.insert("Sub", sub);
    reg.insert("Neg", Neg::build);
}

element_map!(Rsqrt, |x: f32| 1.0 / (x.sqrt()));
element_map!(Abs, |x: f32| x.abs());
element_map!(Neg, |x| -1. * x);

element_bin!(Add, add, |mut a, b| {
    a += &b;
    a
});
element_bin!(Div, div, |mut a, b| {
    a /= &b;
    a
});
element_bin!(Mul, mul, |mut a, b| {
    a *= &b;
    a
});
element_bin!(Sub, sub, |mut a, b| {
    a -= &b;
    a
});
element_bin!(Rem, rem, |mut a, b| {
    a %= &b;
    a
});

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
