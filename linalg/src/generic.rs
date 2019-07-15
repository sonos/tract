pub mod conv;
pub mod matmul;
pub mod tiling;
pub mod vecmatmul;

pub use self::conv::SConv4x4;
pub use self::matmul::DMatMul4x2;
pub use self::matmul::SMatMul4x4;
pub use self::tiling::STiling4x4;
pub use self::vecmatmul::SVecMatMul8;
