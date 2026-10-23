use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::array::CumSum;

/// Per-backend CumSum kernel launcher: `(axis, exclusive, reverse, input, output)`, both
/// contiguous and of the same shape and type.
pub type DispatchCumSumFn = fn(usize, bool, bool, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

/// Backend-agnostic cumulative sum, mirroring `core::ops::array::CumSum` for the types of
/// [`GpuCumSum::is_supported_dt`]. f16 accumulates in f32.
#[derive(Clone)]
pub struct GpuCumSum {
    pub axis: usize,
    pub exclusive: bool,
    pub reverse: bool,
    pub backend_name: &'static str,
    pub dispatch: DispatchCumSumFn,
}

impl GpuCumSum {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16 | DatumType::I32 | DatumType::I64)
    }

    pub fn from_tract_core(
        op: &CumSum,
        backend_name: &'static str,
        dispatch: DispatchCumSumFn,
    ) -> GpuCumSum {
        GpuCumSum {
            axis: op.axis,
            exclusive: op.exclusive,
            reverse: op.reverse,
            backend_name,
            dispatch,
        }
    }
}

impl std::fmt::Debug for GpuCumSum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}CumSum(axis={}, exclusive={}, reverse={})",
            self.backend_name, self.axis, self.exclusive, self.reverse
        )
    }
}

impl PartialEq for GpuCumSum {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axis == other.axis
            && self.exclusive == other.exclusive
            && self.reverse == other.reverse
    }
}

impl Eq for GpuCumSum {}

impl std::hash::Hash for GpuCumSum {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.exclusive.hash(state);
        self.reverse.hash(state);
    }
}

impl Op for GpuCumSum {
    fn name(&self) -> StaticName {
        format!("{}CumSum", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis: {} exclusive: {} reverse: {}",
            self.axis, self.exclusive, self.reverse
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuCumSum {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        (self.dispatch)(self.axis, self.exclusive, self.reverse, input, &output)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuCumSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(self.axis < facts[0].rank(), "{} axis out of range", self.name());
            Ok(tvec!(facts[0].without_value()))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
