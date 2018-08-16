use ops::prelude::*;

mod add_n;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Abs", abs);
    reg.insert("Add", add);
    reg.insert("AddN", add_n::add_n);
    reg.insert("BiasAdd", add);
    reg.insert("Div", div);
    reg.insert("FloorMod", rem);
    reg.insert("Mul", mul);
    reg.insert("Neg", neg);
    reg.insert("Rsqrt", rsqrt);
    reg.insert("Sub", sub);
    reg.insert("Tanh", tanh);
}

element_map_signed!(Abs, abs, |x| x.abs());
element_map_float!(Rsqrt, rsqrt, |x| x.sqrt().recip());
element_map_float!(Tanh, tanh, |x| x.tanh());

element_map!(Neg, neg, [i32, f32, TDim], |x| -x);

element_bin!(Add, add, [i32, f32, TDim], |mut a, b| {
    a += &b;
    a
});
element_bin!(Div, div, [i32, f32, TDim], |mut a, b| {
    a /= &b;
    a
});
element_bin!(Mul, mul, [i32, f32, TDim], |mut a, b| {
    a *= &b;
    a
});
element_bin!(Sub, sub, [i32, f32, TDim], |mut a, b| {
    a -= &b;
    a
});
element_bin!(Rem, rem, [i32, f32, TDim], |mut a, b| {
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
