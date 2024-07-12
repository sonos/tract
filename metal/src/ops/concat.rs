use crate::kernels::array::Concat;
use crate::tensor::MetalTensorExt;
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::array::TypedConcat;

#[derive(new, Debug, Clone, Hash)]
pub struct MetalConcat {
    pub axis: usize,
}

impl MetalConcat {
    pub fn from_tract_core(op: &TypedConcat) -> Self {
        Self { axis: op.axis }
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

impl Op for MetalConcat {
    fn name(&self) -> Cow<str> {
        "MetalConcat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for MetalConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let metal_inputs = inputs
                    .iter()
                    .map(|it| it.to_metal_tensor())
                    .collect::<TractResult<TVec<_>>>()?;
                Ok(tvec!(Concat
                    .dispatch_eval(context, &metal_inputs, self.axis)?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| {
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
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }
}
