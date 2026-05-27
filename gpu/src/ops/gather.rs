use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

/// `output = data.gather(axis, indices)`, i.e.
/// `output[..., i, ...] = data[..., indices[i], ...]` along `axis`.
/// Negative indices wrap (matches the CPU op).
///
/// First implementation supports the plain-tensor path only (no block-quant,
/// no packed-matrix storage); the translator's `rule_if` guards the rest out.
pub type DispatchGatherFn = fn(
    data: &DeviceTensor,
    indices: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuGather {
    pub axis: usize,
    pub backend_name: &'static str,
    pub dispatch: DispatchGatherFn,
}

impl std::fmt::Debug for GpuGather {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Gather", self.backend_name)
    }
}

impl PartialEq for GpuGather {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.axis == other.axis
    }
}
impl Eq for GpuGather {}

impl std::hash::Hash for GpuGather {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
    }
}

impl Op for GpuGather {
    fn name(&self) -> StaticName {
        format!("{}Gather", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={}", self.axis)])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuGather {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (data_val, indices_val) = args_2!(inputs);
        let data = data_val.to_device_tensor()?;
        let indices = indices_val.to_device_tensor()?;
        let out_shape = compute_output_shape(self.axis, data.shape(), indices.shape())?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            data.datum_type(),
            &out_shape,
        )?;
        (self.dispatch)(data, indices, self.axis, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuGather {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            ensure!(facts[1].datum_type == i64::datum_type());
            ensure!(facts[0].rank() > self.axis);
            let dt = facts[0].datum_type;
            let mut shape: TVec<TDim> = facts[0].shape.iter().take(self.axis).cloned().collect();
            shape.extend(facts[1].shape.iter().cloned());
            shape.extend(facts[0].shape.iter().skip(self.axis + 1).cloned());
            Ok(tvec!(dt.fact(&shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}

fn compute_output_shape(
    axis: usize,
    data: &[usize],
    indices: &[usize],
) -> TractResult<TVec<usize>> {
    ensure!(data.len() > axis);
    let mut out: TVec<usize> = data[..axis].into();
    out.extend(indices.iter().copied());
    out.extend(data[axis + 1..].iter().copied());
    Ok(out)
}
