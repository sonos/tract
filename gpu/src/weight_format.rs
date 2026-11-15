use tract_core::internal::*;
use tract_core::tract_linalg::WeightType;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;

/// The weight formats the GPU backends have kernels for: the closed subset of [`WeightType`]
/// they read. Backends name their per-format kernels with [`GpuWeightFormat::name`] and match
/// on it exhaustively, so a new format shows every kernel it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuWeightFormat {
    Q40,
    Q20T,
    F16,
    F32,
}

impl GpuWeightFormat {
    /// The suffix of this format's kernels: `q4_0`, `q2_0_t`, `f16` or `f32`.
    pub fn name(self) -> &'static str {
        match self {
            GpuWeightFormat::Q40 => "q4_0",
            GpuWeightFormat::Q20T => "q2_0_t",
            GpuWeightFormat::F16 => "f16",
            GpuWeightFormat::F32 => "f32",
        }
    }

    /// The block format of a block-quantized format.
    pub fn block_quant(self) -> Option<&'static dyn BlockQuant> {
        match self {
            GpuWeightFormat::Q40 => Some(&Q4_0),
            GpuWeightFormat::Q20T => Some(&Q2_0_T),
            GpuWeightFormat::F16 | GpuWeightFormat::F32 => None,
        }
    }

    /// The format of weights of this [`WeightType`], if the GPU backends read it.
    pub fn of(weight_type: &WeightType) -> Option<GpuWeightFormat> {
        match weight_type {
            WeightType::Plain(DatumType::F16) => Some(GpuWeightFormat::F16),
            WeightType::Plain(DatumType::F32) => Some(GpuWeightFormat::F32),
            WeightType::Plain(_) => None,
            WeightType::BlockQuant(bq) => GpuWeightFormat::of_block_quant(&**bq),
        }
    }

    /// The format of weights in this block format, if the GPU backends read it.
    pub fn of_block_quant(format: &dyn BlockQuant) -> Option<GpuWeightFormat> {
        [GpuWeightFormat::Q40, GpuWeightFormat::Q20T]
            .into_iter()
            .find(|f| f.block_quant().is_some_and(|b| b.dyn_eq(format)))
    }

    /// The format of block-quantized weights, when their rows are whole blocks.
    pub fn of_block_quant_fact(fact: &BlockQuantFact) -> Option<GpuWeightFormat> {
        GpuWeightFormat::of_block_quant(&*fact.format)
            .filter(|_| fact.k().is_multiple_of(fact.format.block_len()))
    }

    /// The format of weights with this fact: plain ones through [`TypedFact::weight_type`],
    /// block-quantized ones through [`GpuWeightFormat::of_block_quant_fact`].
    pub fn of_fact(fact: &TypedFact) -> Option<GpuWeightFormat> {
        match fact.exotic_fact() {
            Some(exotic) => GpuWeightFormat::of_block_quant_fact(exotic.downcast_ref()?),
            None => GpuWeightFormat::of(&fact.weight_type()?),
        }
    }
}
