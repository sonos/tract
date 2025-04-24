use crate::kernels::array::Concat;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::array::TypedConcat;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(new, Debug, Clone, Hash)]
pub struct MetalConcat {
    pub kernel: Concat,
}

impl MetalConcat {
    pub fn from_tract_core(op: &TypedConcat) -> Self {
        Self { kernel: Concat { axis: op.axis } }
    }

    pub fn axis(&self) -> usize {
        self.kernel.axis
    }

    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        for slice in inputs {
            let len = slice.shape[self.axis()].clone();
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl Op for MetalConcat {
    fn name(&self) -> Cow<str> {
        "MetalConcat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis())])
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalConcat);

impl MetalEvalOp for MetalConcat {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        opaque_inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs = opaque_inputs
            .iter()
            .map(|it| it.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;

        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis()] = inputs.iter().map(|it| it.shape()[self.axis()]).sum();
        let output = crate::ops::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape,
        )?;
        self.kernel.dispatch_eval(stream, &inputs, &output)?;

        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            for input in facts {
                if input.rank() != fact.rank()
                    || input
                        .shape
                        .iter()
                        .zip(fact.shape.iter())
                        .enumerate()
                        .filter(|(ax, _)| *ax != self.axis())
                        .any(|(_, (i, f))| i != f)
                {
                    bail!("Inconsistent {:?} inputs: {:?}", self, facts);
                }
            }
            fact.shape.set(self.axis(), self.offsets(facts)?.pop().unwrap());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
}
