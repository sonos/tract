pub mod by_scalar;
pub mod erf;
pub mod gelu;
pub mod hardswish;
pub mod leaky_relu;
pub mod lut;
pub mod mmm;
pub mod reduce;
pub mod rms_norm;
pub mod rounding;
pub mod sigmoid;
pub mod silu;
pub mod tanh;
pub mod unicast;

use tract_data::prelude::DatumType;

use crate::by_scalar::ByScalarKer;
use crate::unicast::UnicastKer;
use crate::{BinOp, LinalgRegistry};

pub use self::by_scalar::{HMulByScalar8, SMulByScalar4};
pub use self::erf::SErf4;
pub use self::gelu::{HGelu8, SGelu4};
pub use self::hardswish::{HHardSwish8, SHardSwish4};
pub use self::leaky_relu::{HLeakyRelu8, SLeakyRelu4};
pub use self::lut::GenericLut8;
pub use self::rounding::{ScaleShiftAndRound, Scaler};
pub use self::sigmoid::{HSigmoid8, SSigmoid4};
pub use self::silu::{HSiLU8, SSiLU4};
pub use self::tanh::{HTanh8, STanh4};

pub(crate) fn register_all_unicast(registry: &mut LinalgRegistry) {
    registry.insert((BinOp::Mul, DatumType::F32), Box::new(|| unicast::SUnicastMul4::bin()));
    registry.insert((BinOp::Mul, DatumType::F16), Box::new(|| unicast::HUnicastMul8::bin()));
    registry.insert((BinOp::Add, DatumType::F32), Box::new(|| unicast::SUnicastAdd4::bin()));
    registry.insert((BinOp::Add, DatumType::F16), Box::new(|| unicast::HUnicastAdd8::bin()));
    registry.insert((BinOp::Sub, DatumType::F32), Box::new(|| unicast::SUnicastSub4::bin()));
    registry.insert((BinOp::Sub, DatumType::F16), Box::new(|| unicast::HUnicastSub8::bin()));
    registry.insert((BinOp::SubF, DatumType::F32), Box::new(|| unicast::SUnicastSubF4::bin()));
    registry.insert((BinOp::SubF, DatumType::F16), Box::new(|| unicast::HUnicastSubF8::bin()));
    registry.insert((BinOp::Min, DatumType::F32), Box::new(|| unicast::SUnicastMin4::bin()));
    registry.insert((BinOp::Min, DatumType::F16), Box::new(|| unicast::HUnicastMin8::bin()));
    registry.insert((BinOp::Max, DatumType::F32), Box::new(|| unicast::SUnicastMax4::bin()));
    registry.insert((BinOp::Max, DatumType::F16), Box::new(|| unicast::HUnicastMax8::bin()));
}

pub(crate) fn register_all_by_scalar(registry: &mut LinalgRegistry) {
    registry.insert((BinOp::Mul, DatumType::F32), Box::new(|| by_scalar::SMulByScalar4::bin()));
    registry.insert((BinOp::Mul, DatumType::F16), Box::new(|| by_scalar::HMulByScalar8::bin()));
    registry.insert((BinOp::Add, DatumType::F32), Box::new(|| by_scalar::SAddByScalar4::bin()));
    registry.insert((BinOp::Add, DatumType::F16), Box::new(|| by_scalar::HAddByScalar8::bin()));
    registry.insert((BinOp::Sub, DatumType::F32), Box::new(|| by_scalar::SSubByScalar4::bin()));
    registry.insert((BinOp::Sub, DatumType::F16), Box::new(|| by_scalar::HSubByScalar8::bin()));
    registry.insert((BinOp::SubF, DatumType::F32), Box::new(|| by_scalar::SSubFByScalar4::bin()));
    registry.insert((BinOp::SubF, DatumType::F16), Box::new(|| by_scalar::HSubFByScalar8::bin()));
    registry.insert((BinOp::Min, DatumType::F32), Box::new(|| by_scalar::SMinByScalar4::bin()));
    registry.insert((BinOp::Min, DatumType::F16), Box::new(|| by_scalar::HMinByScalar8::bin()));
    registry.insert((BinOp::Max, DatumType::F32), Box::new(|| by_scalar::SMaxByScalar4::bin()));
    registry.insert((BinOp::Max, DatumType::F16), Box::new(|| by_scalar::HMaxByScalar8::bin()));
}

routine!(F32, Sigmoid, sigmoid::SSigmoid4);
routine!(F16, Sigmoid, sigmoid::HSigmoid8);
routine!(F32, Tanh, tanh::STanh4);
routine!(F16, Tanh, tanh::HTanh8);
routine!(F32, Silu, silu::SSiLU4);
routine!(F16, Silu, silu::HSiLU8);
routine!(F32, Gelu, gelu::SGelu4);
routine!(F16, Gelu, gelu::HGelu8);
routine!(F32, Erf, erf::SErf4);
routine!(F32, Hardswish, hardswish::SHardSwish4);
routine!(F16, Hardswish, hardswish::HHardSwish8);
routine!(F32Param, LeakyRelu, leaky_relu::SLeakyRelu4);
routine!(F16Param, LeakyRelu, leaky_relu::HLeakyRelu8);
routine!(F32Param, MulByScalar, by_scalar::SMulByScalar4);
routine!(F16Param, MulByScalar, by_scalar::HMulByScalar8);
routine!(F32Reduce, ReduceMax, reduce::max::SMax4);
routine!(F16Reduce, ReduceMax, reduce::max::HMax8);
routine!(F32Reduce, ReduceMin, reduce::min::SMin4);
routine!(F32Reduce, ReduceSum, reduce::sum::SSum4);
routine!(F16Reduce, ReduceSum, reduce::sum::HSum8);
routine!(F32MapReduce, Softmax2, reduce::softmax_l2::SSoftMaxL2Accurate);
routine!(RmsNormF32, RmsNorm, "generic", rms_norm::rms_norm_f32);
