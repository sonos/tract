pub mod lut;
pub mod mmm;
pub mod sigmoid;
pub mod tanh;
pub mod vecmatmul;

pub use self::lut::GenericLut8;
pub use self::mmm::GenericMmm4x4;
pub use self::sigmoid::SSigmoid4;
pub use self::tanh::STanh4;
pub use self::vecmatmul::SVecMatMul8;
