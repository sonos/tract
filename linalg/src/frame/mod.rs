/// What a binary operation computes, for the tests to compare against. One place, so a kernel's
/// reference cannot drift from the operation its descriptor declares.
macro_rules! bin_reference {
    (Mul) => {
        |a, b| a * b
    };
    (Add) => {
        |a, b| a + b
    };
    (Sub) => {
        |a, b| a - b
    };
    (SubF) => {
        |a, b| b - a
    };
    (Min) => {
        |a, b| a.min(b)
    };
    (Max) => {
        |a, b| a.max(b)
    };
}

#[macro_use]
pub mod block_quant;
#[macro_use]
pub mod element_wise;
pub mod element_wise_helper;
#[macro_use]
pub mod unicast;
#[macro_use]
pub mod by_scalar;
#[macro_use]
pub mod erf;
#[macro_use]
pub mod gelu;
#[macro_use]
pub mod hardswish;
#[macro_use]
pub mod leaky_relu;
#[macro_use]
pub mod lut;
#[macro_use]
pub mod mmm;
#[macro_use]
pub mod pack;
#[macro_use]
pub mod reduce;
#[macro_use]
pub mod sigmoid;
#[macro_use]
pub mod silu;
#[macro_use]
pub mod tanh;
#[macro_use]
pub mod weights;
