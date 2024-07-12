pub mod by_scalar;
pub mod erf;
pub mod leaky_relu;
pub mod lut;
pub mod mmm;
pub mod reduce;
pub mod rounding;
pub mod sigmoid;
pub mod tanh;
pub mod unicast;

pub use self::by_scalar::{HMulByScalar8, SMulByScalar4};
pub use self::erf::SErf4;
pub use self::leaky_relu::{HLeakyRelu8, SLeakyRelu4};
pub use self::lut::GenericLut8;
pub use self::rounding::{ScaleShiftAndRound, Scaler};
pub use self::sigmoid::{HSigmoid8, SSigmoid4};
pub use self::reduce::softmax_l2::SSoftMaxL2;
pub use self::tanh::{HTanh8, STanh4};
