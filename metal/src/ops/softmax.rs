use crate::kernels::nn::Softmax;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Hash, Default)]
pub struct MetalSoftmax {
    pub axes: TVec<usize>,
}

impl MetalSoftmax {
    pub fn new(axes: TVec<usize>) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axis of softmax is supported by MetalSoftmax");
        Ok(Self { axes })
    }

    pub fn from_tract_core(core_softmax: &core_ops_nn::Softmax) -> TractResult<Self> {
        ensure!(core_softmax.quant_output_dt.is_none());
        Self::new(core_softmax.axes.clone())
    }
}

impl Op for MetalSoftmax {
    fn name(&self) -> Cow<str> {
        "MetalSoftmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalSoftmax);

impl MetalEvalOp for MetalSoftmax {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), input.shape())?;
        Softmax.dispatch_eval(stream, input, self.axes[0], &output)?;

        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalSoftmax {
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
                Some(Box::new(MetalSoftmax { axes })),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    as_op!();
}
