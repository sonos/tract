use crate::tensor::DeviceTensorExt;
use crate::utils::DispatchCopyNdFn;
use tract_core::internal::*;

#[derive(Clone)]
pub struct GpuConcat {
    pub axis: usize,
    pub backend_name: &'static str,
    pub dispatch: DispatchCopyNdFn,
}

impl GpuConcat {
    pub fn new(axis: usize, backend_name: &'static str, dispatch: DispatchCopyNdFn) -> Self {
        Self { axis, backend_name, dispatch }
    }

    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        for slice in inputs {
            let len = slice.shape[self.axis].clone();
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl std::fmt::Debug for GpuConcat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Concat(axis={})", self.backend_name, self.axis)
    }
}

impl PartialEq for GpuConcat {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.axis == other.axis
    }
}

impl Eq for GpuConcat {}

impl std::hash::Hash for GpuConcat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
    }
}

impl Op for GpuConcat {
    fn name(&self) -> StaticName {
        format!("{}Concat", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;

        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis] = inputs.iter().map(|it| it.shape()[self.axis]).sum();
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape,
        )?;

        let mut cursor = 0usize;
        for input in &inputs {
            let slice_len = input.shape()[self.axis];
            if slice_len == 0 {
                continue;
            }
            // Build zone shape (same as input shape for this slice)
            let zone_shape = input.shape();
            // Output offset along concat axis
            let dst_offset =
                cursor * output.strides()[self.axis] as usize * output.datum_type().size_of();

            (self.dispatch)(
                input,
                0,
                input.strides(),
                &output,
                dst_offset,
                zone_shape,
                output.strides(),
            )
            .with_context(|| {
                format!(
                    "Error in concat dispatch for slice at offset {} (shape {:?})",
                    cursor, zone_shape
                )
            })?;
            cursor += slice_len;
        }

        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuConcat {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            for input in facts {
                if input.rank() != fact.rank()
                    || input
                        .shape
                        .iter()
                        .zip(fact.shape.iter())
                        .enumerate()
                        .filter(|(ax, _)| *ax != self.axis)
                        .any(|(_, (i, f))| i != f)
                {
                    bail!("Inconsistent {:?} inputs: {:?}", self, facts);
                }
            }
            fact.shape.set(self.axis, self.offsets(facts)?.pop().unwrap());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
