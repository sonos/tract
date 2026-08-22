//! What a caller needs a matmul for, and what it can be given in return.

use tract_data::prelude::{DatumType, TVec};

use super::{MatMatMul, PanelExtractor};
use crate::WeightType;

/// One way to compute a matmul: a kernel, which of its packings to use, and the panel
/// extractor to reach that packing when the weights are not already in it.
pub type Candidate = (Box<dyn MatMatMul>, usize, Option<PanelExtractor>);

/// A matmul as kernel selection sees it: the operand types a kernel must handle, and the
/// shape where the caller knows it. A `None` dim is one the caller cannot pin — a streaming
/// axis, or an `n` still symbolic at optimisation time.
#[derive(Clone)]
pub struct Query {
    pub weight: WeightType,
    pub activation: DatumType,
    /// Internal accumulator types the caller accepts.
    pub accumulators: TVec<DatumType>,
    /// The datum type the kernel must be able to store, when the caller constrains it.
    pub store: Option<DatumType>,
    pub m: Option<usize>,
    pub k: Option<usize>,
    pub n: Option<usize>,
}
