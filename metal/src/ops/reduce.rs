use crate::kernels::nn::Reducer;
use crate::tensor::MetalTensorExt;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_itertools::Itertools;

#[derive(Clone, Debug, Hash)]
pub struct MetalReduce {
    pub axes: TVec<usize>,
    pub reducer: Reducer,
}

impl MetalReduce {
    pub fn new(axes: TVec<usize>, reducer: Reducer) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axe of reduce is supported by MetalReduce");
        Ok(Self { axes, reducer })
    }

    pub fn from_tract_core(core_reduce: &core_ops_nn::Reduce) -> TractResult<Self> {
        let metal_reducer = match core_reduce.reducer {
            core_ops_nn::Reducer::Sum => Reducer::Sum,
            core_ops_nn::Reducer::MeanOfSquares => Reducer::MeanOfSquares,
            core_ops_nn::Reducer::Prod => Reducer::Prod,
            core_ops_nn::Reducer::Min => Reducer::Min,
            core_ops_nn::Reducer::Max => Reducer::Max,
            _ => bail!("Unsupported reducer {:?} on metal", core_reduce.reducer),
        };
        Self::new(core_reduce.axes.clone(), metal_reducer)
    }
}

impl Op for MetalReduce {
    fn name(&self) -> Cow<str> {
        format!("MetalReduce<{:?}>", self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }
    op_as_typed_op!();
}

impl EvalOp for MetalReduce {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let input = args_1!(inputs);
                let input_metal = input.to_metal_tensor()?;
                Ok(tvec!(self
                    .reducer
                    .dispatch_eval(context, input_metal, self.axes[0])?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalReduce {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axes.iter().tuple_windows().all(|(a, b)| a < b));
        crate::utils::metal_output_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            for &ax in &self.axes {
                shape[ax] = 1.to_dim();
            }
            let dt = facts[0].datum_type;
            Ok(tvec!(dt.fact(shape)))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
