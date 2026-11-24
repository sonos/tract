//! The cooperative-matrix WGSL of naga, which compiles native shaders, and of Dawn, which compiles
//! a browser's. Every matrix is 8x8 and row major.

use crate::kernels::shaders::ShaderDtype;
use crate::kernels::subgroup::{MatrixTile, SubgroupCaps};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Dialect {
    Naga,
    Dawn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    Left,
    Right,
}

impl Dialect {
    pub(crate) const CURRENT: Dialect =
        if cfg!(target_arch = "wasm32") { Dialect::Dawn } else { Dialect::Naga };

    fn operand(self, side: Side, scalar: &str) -> String {
        match (self, side) {
            (Dialect::Naga, Side::Left) => format!("coop_mat8x8<{scalar}, A>"),
            (Dialect::Naga, Side::Right) => format!("coop_mat8x8<{scalar}, B>"),
            (Dialect::Dawn, Side::Left) => format!("subgroup_matrix_left<{scalar}, 8, 8>"),
            (Dialect::Dawn, Side::Right) => format!("subgroup_matrix_right<{scalar}, 8, 8>"),
        }
    }

    /// A zero accumulator.
    pub(crate) fn accumulator(self, scalar: &str) -> String {
        match self {
            Dialect::Naga => format!("coop_mat8x8<{scalar}, C>()"),
            Dialect::Dawn => format!("subgroup_matrix_result<{scalar}, 8, 8>()"),
        }
    }

    /// A zero operand.
    pub(crate) fn zero(self, side: Side, scalar: &str) -> String {
        format!("{}()", self.operand(side, scalar))
    }

    /// Loads the operand whose first element is `place` in naga, and `offset` elements into the
    /// array `array` in Dawn, where the array has to span the whole operand.
    pub(crate) fn load(
        self,
        side: Side,
        scalar: &str,
        place: &str,
        (array, offset): (&str, &str),
        stride: usize,
    ) -> String {
        let ty = self.operand(side, scalar);
        match self {
            Dialect::Naga => format!("coopLoadT<{ty}>(&{place}, {stride}u)"),
            Dialect::Dawn => {
                format!("subgroupMatrixLoad<{ty}, row_major>(&{array}, {offset}, {stride}u)")
            }
        }
    }

    pub(crate) fn multiply_add(self, left: &str, right: &str, accumulator: &str) -> String {
        let function = match self {
            Dialect::Naga => "coopMultiplyAdd",
            Dialect::Dawn => "subgroupMatrixMultiplyAccumulate",
        };
        format!("{function}({left}, {right}, {accumulator})")
    }

    pub(crate) fn store(self, matrix: &str, array: &str, offset: &str, stride: usize) -> String {
        match self {
            Dialect::Naga => format!("coopStoreT({matrix}, &{array}[{offset}], {stride}u);"),
            Dialect::Dawn => {
                format!("subgroupMatrixStore<row_major>(&{array}, {offset}, {matrix}, {stride}u);")
            }
        }
    }
}

/// The scalar an 8x8 product of `dt` is accumulated in. `f16` operands accumulate in `f32` where
/// the device has such a tile. Otherwise they accumulate in `f16`, which `f16_accumulation` allows.
pub(crate) fn accumulator(
    caps: &SubgroupCaps,
    dt: ShaderDtype,
    f16_accumulation: bool,
) -> Option<ShaderDtype> {
    let square = |tile: Option<MatrixTile>| tile.is_some_and(|t| (t.m, t.n, t.k) == (8, 8, 8));
    match dt {
        ShaderDtype::F32 if square(caps.f32_tile) => Some(ShaderDtype::F32),
        ShaderDtype::F16 if square(caps.f16_mixed_tile) => Some(ShaderDtype::F32),
        ShaderDtype::F16 if f16_accumulation && square(caps.f16_tile) => Some(ShaderDtype::F16),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILE: Option<MatrixTile> = Some(MatrixTile { m: 8, n: 8, k: 8 });

    #[test]
    fn f16_accumulates_in_f16_only_when_allowed() {
        let browser = SubgroupCaps { f32_tile: TILE, f16_tile: TILE, ..Default::default() };
        assert_eq!(accumulator(&browser, ShaderDtype::F32, false), Some(ShaderDtype::F32));
        assert_eq!(accumulator(&browser, ShaderDtype::F16, false), None);
        assert_eq!(accumulator(&browser, ShaderDtype::F16, true), Some(ShaderDtype::F16));
    }

    #[test]
    fn a_mixed_tile_accumulates_in_f32_whatever_the_knob() {
        let native = SubgroupCaps { f16_tile: TILE, f16_mixed_tile: TILE, ..Default::default() };
        assert_eq!(accumulator(&native, ShaderDtype::F16, false), Some(ShaderDtype::F32));
        assert_eq!(accumulator(&native, ShaderDtype::F16, true), Some(ShaderDtype::F32));
        assert_eq!(accumulator(&native, ShaderDtype::F32, false), None);
    }

    #[test]
    fn other_tile_sizes_are_declined() {
        let wide = Some(MatrixTile { m: 16, n: 16, k: 16 });
        let caps = SubgroupCaps { f32_tile: wide, f16_tile: wide, ..Default::default() };
        assert_eq!(accumulator(&caps, ShaderDtype::F32, true), None);
        assert_eq!(accumulator(&caps, ShaderDtype::F16, true), None);
    }

    #[test]
    fn dawn_addresses_an_array_and_naga_an_element() {
        let load = |d: Dialect| d.load(Side::Left, "f16", "b[0][i]", ("b", "i"), 64);
        assert_eq!(load(Dialect::Naga), "coopLoadT<coop_mat8x8<f16, A>>(&b[0][i], 64u)");
        assert_eq!(
            load(Dialect::Dawn),
            "subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&b, i, 64u)"
        );
        let store = |d: Dialect| d.store("acc", "stage", "8u", 8);
        assert_eq!(store(Dialect::Naga), "coopStoreT(acc, &stage[8u], 8u);");
        assert_eq!(store(Dialect::Dawn), "subgroupMatrixStore<row_major>(&stage, 8u, acc, 8u);");
    }
}
