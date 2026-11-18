use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::logic::{CompEq, CompGT, CompGTE, CompLT, CompLTE, CompNE};

/// The comparison a [`GpuCompareSelect`] makes between its first two inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl Comparison {
    /// The comparison a core comparison mini-op makes, if it is one.
    pub fn from_mini_op(op: &dyn BinMiniOp) -> Option<Comparison> {
        Some(if op.is::<CompEq>() {
            Comparison::Eq
        } else if op.is::<CompNE>() {
            Comparison::Ne
        } else if op.is::<CompLT>() {
            Comparison::Lt
        } else if op.is::<CompLTE>() {
            Comparison::Lte
        } else if op.is::<CompGT>() {
            Comparison::Gt
        } else if op.is::<CompGTE>() {
            Comparison::Gte
        } else {
            return None;
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Comparison::Eq => "eq",
            Comparison::Ne => "ne",
            Comparison::Lt => "lt",
            Comparison::Lte => "lte",
            Comparison::Gt => "gt",
            Comparison::Gte => "gte",
        }
    }
}

pub type DispatchCompareSelectFn = fn(
    Comparison,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
) -> TractResult<()>;

/// `cmp(a, b) ? t : f`, core's `Iff(Comp(a, b), t, f)` as one op, for backends with no boolean
/// tensors to carry the condition. Inputs `a, b, t, f` share one float type and one rank and
/// broadcast against each other; the output has their common shape.
#[derive(Clone, new)]
pub struct GpuCompareSelect {
    pub comparison: Comparison,
    pub backend_name: &'static str,
    pub dispatch: DispatchCompareSelectFn,
}

impl std::fmt::Debug for GpuCompareSelect {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}CompareSelect({:?})", self.backend_name, self.comparison)
    }
}

impl PartialEq for GpuCompareSelect {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.comparison == other.comparison
    }
}

impl Eq for GpuCompareSelect {}

impl std::hash::Hash for GpuCompareSelect {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.comparison.hash(state);
    }
}

impl Op for GpuCompareSelect {
    fn name(&self) -> StaticName {
        format!("{}CompareSelect", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.comparison)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuCompareSelect {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b, t, f) = args_4!(inputs);
        let (a, b, t, f) = (
            a.to_device_tensor()?,
            b.to_device_tensor()?,
            t.to_device_tensor()?,
            f.to_device_tensor()?,
        );
        let shape =
            tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape(), t.shape(), f.shape()])?;
        let output = crate::turn_handler::make_tensor_for_node(ctx, t.datum_type(), &shape)?;
        (self.dispatch)(self.comparison, a, b, t, f, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuCompareSelect {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let shapes: TVec<_> = facts.iter().map(|f| f.shape.to_tvec()).collect();
            let shape = tract_core::broadcast::multi_broadcast(&shapes)?;
            Ok(tvec!(facts[2].datum_type.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
