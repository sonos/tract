use crate::kernels::nn::Softmax;
use crate::tensor::MetalTensorExt;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;

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

impl TypedOp for MetalSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
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

impl EvalOp for MetalSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let input = args_1!(inputs);
                let t = input.to_metal_tensor()?;

                Ok(tvec!(Softmax
                    .dispatch_eval(context, t, self.axes[0])?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}
