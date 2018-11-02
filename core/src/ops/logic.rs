use ops::prelude::*;

element_map!(Not, [bool], |a| !a);

element_bin!(And, [bool] { |a, b| a & b});
element_bin!(Or, [bool] { |a, b| a | b});
element_bin!(Xor, [bool] { |a, b| a ^ b});

element_bin!(Equals, [bool, u8, i8, i32, f32, f64, TDim] => bool { |a,b| a==b });
element_bin!(Lesser, [u8, i8, i32, f32, f64] => bool { |a,b| a<b });
element_bin!(Greater, [u8, i8, i32, f32, f64] => bool { |a,b| a>b });
