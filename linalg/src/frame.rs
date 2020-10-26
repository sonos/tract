#[macro_use]
pub mod lut;
#[macro_use]
pub mod mmm;
pub mod pack_a;
pub mod pack_b;
#[macro_use]
pub mod sigmoid;
#[macro_use]
pub mod tanh;

pub use pack_a::PackA;
pub use pack_b::PackB;

pub use self::mmm::{MatMatMul, MatMatMulImpl};

pub use self::sigmoid::SigmoidImpl;
pub use self::tanh::TanhImpl;
