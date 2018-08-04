use super::{Op, OpRegister};
use Result;

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

element_map!(Abs, abs, |x| x.abs());
element_map!(Neg, neg, |x| x.neg());
element_map!(Rsqrt, rsqrt, |x| x.sqrt().recip());
element_map!(Tanh, tanh, |x| x.tanh());

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
