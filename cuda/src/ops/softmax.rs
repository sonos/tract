use crate::context::CUDA_STREAM;
use crate::kernels::nn::Softmax;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Hash, Default)]
pub struct CudaSoftmax {
    pub axes: TVec<usize>,
}

impl CudaSoftmax {
    pub fn new(axes: TVec<usize>) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axis of softmax is supported by CudaSoftmax");
        Ok(Self { axes })
    }

    pub fn from_tract_core(core_softmax: &core_ops_nn::Softmax) -> TractResult<Self> {
        ensure!(core_softmax.quant_output_dt.is_none());
        Self::new(core_softmax.axes.clone())
    }
}

impl Op for CudaSoftmax {
    fn name(&self) -> StaticName {
        "CudaSoftmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }

    op_as_typed_op!();
}

impl EvalOp for CudaSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let opaque = args_1!(inputs);
            let input = opaque.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input.datum_type(),
                input.shape(),
            )?;
            Softmax.dispatch_eval(stream, input, self.axes[0], &output)?;

            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axes: Option<TVec<usize>> =
            self.axes.iter().map(|it| change.transform_axis(*it)).collect();
        if let Some(axes) = axes {
            Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(CudaSoftmax { axes })),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    as_op!();
}
