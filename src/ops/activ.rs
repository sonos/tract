use {Matrix, Result};
use super::Op;

element_map!(Relu, |x| if x<0.0 { 0.0 } else {x});
