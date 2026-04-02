use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;

use crate::tensor::DeviceTensor;

pub type DispatchSoftmaxFn = fn(&DeviceTensor, usize, &DeviceTensor) -> TractResult<()>;

#[derive(Clone)]
pub struct GpuSoftmax {
    pub axes: TVec<usize>,
    pub backend_name: &'static str,
    pub dispatch: DispatchSoftmaxFn,
}

impl GpuSoftmax {
    pub fn new(
        axes: TVec<usize>,
        backend_name: &'static str,
        dispatch: DispatchSoftmaxFn,
    ) -> TractResult<Self> {
        ensure!(
            axes.len() == 1,
            "Only one axis of softmax is supported by {}Softmax",
            backend_name
        );
        Ok(Self { axes, backend_name, dispatch })
    }

    pub fn from_tract_core(
        core_softmax: &core_ops_nn::Softmax,
        backend_name: &'static str,
        dispatch: DispatchSoftmaxFn,
    ) -> TractResult<Self> {
        ensure!(core_softmax.quant_output_dt.is_none());
        Self::new(core_softmax.axes.clone(), backend_name, dispatch)
    }
}

impl std::fmt::Debug for GpuSoftmax {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Softmax(axes: {:?})", self.backend_name, self.axes)
    }
}

impl PartialEq for GpuSoftmax {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.axes == other.axes
    }
}

impl Eq for GpuSoftmax {}

impl std::hash::Hash for GpuSoftmax {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axes.hash(state);
    }
}

impl Op for GpuSoftmax {
    fn name(&self) -> StaticName {
        format!("{}Softmax", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            input.shape(),
        )?;
        (self.dispatch)(input, self.axes[0], &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
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
                Some(Box::new(GpuSoftmax {
                    axes,
                    backend_name: self.backend_name,
                    dispatch: self.dispatch,
                })),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    as_op!();
}
