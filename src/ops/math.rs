use {Matrix, Result};
use super::{ Op, OpRegister };

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Abs", Abs::build);
    reg.insert("Add", Add::build);
    reg.insert("BiasAdd", Add::build);
    reg.insert("Div", Div::build);
    reg.insert("Mul", Mul::build);
    reg.insert("Rsqrt", Rsqrt::build);
    reg.insert("Sub", Sub::build);
}

element_map!(Rsqrt, |x: f32| 1.0 / (x.sqrt()));
element_map!(Abs, |x: f32| x.abs());

element_bin!(Add, |mut a,b| { a+=&b ; a });
element_bin!(Div, |mut a,b| { a/=&b ; a });
element_bin!(Mul, |mut a,b| { a*=&b ; a });
element_bin!(Sub, |mut a,b| { a-=&b ; a });
