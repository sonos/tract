pub mod activations;
pub mod erf;
pub mod lut;
pub mod mmm;
pub mod rounding;
pub mod sigmoid;
pub mod tanh;

pub use self::erf::SErf4;
pub use self::lut::GenericLut8;
pub use self::mmm::GenericMmm4x1;
pub use self::mmm::GenericMmm4x4;
pub use self::rounding::{ScaleShiftAndRound, Scaler};
pub use self::sigmoid::{HSigmoid8, SSigmoid4};
pub use self::tanh::{ HTanh8, STanh4};
