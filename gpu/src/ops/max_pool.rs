use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

/// Geometry of a two-spatial-axis max pooling, with every axis stride given
/// explicitly so one kernel serves both NCHW and NHWC.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaxPool2dGeometry {
    pub batch: usize,
    pub channels: usize,
    pub input_hw: (usize, usize),
    pub output_hw: (usize, usize),
    pub kernel: (usize, usize),
    pub strides: (usize, usize),
    pub dilations: (usize, usize),
    pub padding: (usize, usize),
    pub input_strides: [usize; 4],
    pub output_strides: [usize; 4],
}

pub type DispatchMaxPool2dFn =
    fn(input: &DeviceTensor, geo: &MaxPool2dGeometry, output: &DeviceTensor) -> TractResult<()>;

/// Max pooling over two spatial axes, geometry resolved at translation time.
/// Windows lying entirely in the padding take the datum type's lowest value,
/// matching the CPU op.
#[derive(Clone, new)]
pub struct GpuMaxPool {
    pub geometry: MaxPool2dGeometry,
    pub output_shape: TVec<usize>,
    pub backend_name: &'static str,
    pub dispatch: DispatchMaxPool2dFn,
}

impl std::fmt::Debug for GpuMaxPool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}MaxPool", self.backend_name)
    }
}

impl PartialEq for GpuMaxPool {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.geometry == other.geometry
            && self.output_shape == other.output_shape
    }
}
impl Eq for GpuMaxPool {}

impl std::hash::Hash for GpuMaxPool {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.geometry.hash(state);
        self.output_shape.hash(state);
    }
}

impl Op for GpuMaxPool {
    fn name(&self) -> StaticName {
        format!("{}MaxPool", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "kernel={:?} strides={:?} dilations={:?} padding={:?}",
            self.geometry.kernel,
            self.geometry.strides,
            self.geometry.dilations,
            self.geometry.padding
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuMaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &self.output_shape,
        )?;
        (self.dispatch)(input, &self.geometry, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuMaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 1);
            let shape: TVec<TDim> = self.output_shape.iter().map(|d| d.to_dim()).collect();
            Ok(tvec!(facts[0].datum_type.fact(&shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}
