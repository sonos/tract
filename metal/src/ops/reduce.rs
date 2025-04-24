use crate::kernels::nn::Reducer;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_gpu::tensor::DeviceTensorExt;
use tract_itertools::Itertools;

#[derive(Clone, Debug, Hash)]
pub struct MetalReduce {
    pub axes: TVec<usize>,
    pub reducer: Reducer,
}

impl MetalReduce {
    pub fn new(axes: TVec<usize>, reducer: Reducer) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axis of reduce is supported by MetalReduce");
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

crate::impl_eval_op_for_metal_op!(MetalReduce);

impl MetalEvalOp for MetalReduce {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;
        let mut output_shape = input.shape().to_vec();
        output_shape[self.axes[0]] = 1;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), &output_shape)?;

        self.reducer.dispatch_eval(stream, input, self.axes[0], &output)?;

        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalReduce {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axes.iter().tuple_windows().all(|(a, b)| a < b));
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            for &ax in &self.axes {
                shape[ax] = 1.to_dim();
            }
            let dt = facts[0].datum_type;
            Ok(tvec!(dt.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
