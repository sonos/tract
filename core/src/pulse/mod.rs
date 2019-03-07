use std::fmt;
use crate::datum::TryInto;
use crate::ops::prelude::*;
use crate::ops::source::Source;

pub mod delay;

#[derive(Clone, PartialEq)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl fmt::Debug for PulsedTensorFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "Pulse:{}x{:?}{{ax:{},d:{},dim:{:?}}}", self.shape.iter().join("x"), self.dt, self.axis, self.delay, self.dim)
    }
}

impl TensorInfo for PulsedTensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.dt, &self.shape)
    }
}

impl TryInto<TypedTensorInfo> for PulsedTensorFact {
    fn try_into(&self) -> TractResult<TypedTensorInfo> {
        Ok(TypedTensorInfo { shape: (&self.shape).into(), datum_type: self.dt, konst: None })
    }
}

impl PulsedTensorFact {
    pub fn from_tensor_fact_pulse(
        tf: &NormalizedTensorInfo,
        pulse: usize,
    ) -> TractResult<PulsedTensorFact> {
        let dt = tf.datum_type;
        let stream = tf.shape.stream_info.ok_or("Can not pulse a tensor with no streaming dim")?;
        let shape =
            tf.shape.iter().map(|d| d.to_integer().map(|d| d as usize).unwrap_or(pulse)).collect();
        Ok(PulsedTensorFact { dt, shape, axis: stream.axis, dim: stream.len, delay: 0 })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis]
    }

    pub fn to_pulse_fact(&self) -> NormalizedTensorInfo {
        NormalizedTensorInfo { datum_type: self.dt, shape: ShapeInfo::from(&*self.shape) }
    }

    pub fn streaming_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| if ix == self.axis { self.dim } else { d.to_dim() })
            .collect()
    }

    pub fn to_streaming_fact(&self) -> NormalizedTensorInfo {
        let mut info = self.to_pulse_fact();
        info.shape.stream_info = Some(StreamInfo { axis: self.axis, len: self.dim });
        info
    }
}

pub type PulsedModel = Model<PulsedTensorFact>;

impl PulsedModel {
    pub fn new(source: &NormalizedModel, pulse: usize) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, pulse)?.0)
    }

    pub fn new_with_mapping(
        source: &NormalizedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        let mut target = PulsedModel::default();
        let mut mapping = HashMap::new();
        for old_id in source.eval_order()? {
            trace!(
                "Pulsify node {} {} ({})",
                old_id,
                source.node(old_id).name,
                source.node(old_id).op().name()
            );
            if source.node(old_id).op_as::<Source>().is_some() {
                let node = source.node(old_id);
                let pulsed_fact =
                    PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
                let id = target.add_source(node.name.clone(), pulsed_fact)?;
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let node = &source.nodes()[old_id];
                let outlets = node.op().pulsify(&source, &node, &mut target, &mapping)?;
                for (ix, outlet) in outlets.into_iter().enumerate() {
                    mapping.insert(OutletId::new(node.id, ix), outlet);
                }
            }
            trace!("Target is now {}", target.nodes().len());
        }
        // maintaining order of i/o interface
        target.inputs = source.inputs()?.iter().map(|i| mapping[&i]).collect();
        target.outputs = source.outputs()?.iter().map(|o| mapping[&o]).collect();
        Ok((target, mapping))
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        crate::model::compact::compact(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = Model::default();
        let _a =
            model.add_source("a", TensorFact::dt_shape(DatumType::F32, vec![1, 2, 3])).unwrap();
        assert!(
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).is_err()
        );

        let mut model = Model::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            )
            .unwrap();
        let pulse =
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).unwrap();
        assert_eq!(
            pulse.fact(OutletId::new(0, 0)).unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(1, 4, 3))
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = Model::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![TDim::s(), 2.to_dim(), 3.to_dim()]),
            )
            .unwrap();

        let pulse = PulsedModel::new(&model.into_normalized().unwrap(), 4).unwrap();

        assert_eq!(
            pulse.input_fact().unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.output_fact().unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
    }

    fn test_regular_against_pulse(
        model: InferenceModel,
        input_array: ArrayD<f32>,
        axis: usize,
        expected_nodes_in_pulsed_net: usize,
        expected_delay: usize,
    ) {
        let model = model.into_normalized().unwrap();
        let input = Tensor::from(input_array.clone());

        assert_eq!(model.nodes().len(), 2);
        let plan = crate::plan::SimplePlan::new(&model).unwrap();
        let outputs = plan.run(tvec!(input.clone())).unwrap();

        let pulse = 4;
        let pulsed = PulsedModel::new(&model, pulse).unwrap();
        assert_eq!(pulsed.nodes().len(), expected_nodes_in_pulsed_net);
        let output_fact = pulsed.output_fact().unwrap().clone();
        assert_eq!(output_fact.delay, expected_delay);
        let output_stream_axis = output_fact.axis;
        let mut initial_output_shape = output_fact.shape.clone();
        initial_output_shape[output_stream_axis] = 0;

        let pulsed_plan = crate::plan::SimplePlan::new(pulsed).unwrap();
        let mut state = crate::plan::SimpleState::new(&pulsed_plan).unwrap();

        let mut got: ArrayD<f32> = ArrayD::zeros(&*initial_output_shape);

        for p in 0..(input.shape()[axis] / pulse) {
            let chunk = input_array.slice_axis(Axis(axis), ((p * pulse)..((p + 1) * pulse)).into());
            let mut outputs = state.run(tvec!(Tensor::from(chunk.to_owned()).into())).unwrap();
            got = stack(
                Axis(output_stream_axis),
                &[got.view(), outputs.remove(0).to_array_view::<f32>().unwrap()],
            )
            .unwrap();
        }

        let pulsed_output = got.slice_axis(Axis(output_stream_axis), (output_fact.delay..).into());

        assert_eq!(pulsed_output, outputs[0].to_array_view::<f32>().unwrap());
    }

    #[test]
    fn test_simple_conv() {
        use crate::ops::nn::*;

        let mut model = Model::default();
        let ker = model.add_const("kernel", arr3(&[[[0.5f32, 1.0, -0.1]]]).into()).unwrap();
        let _ = model
            .add_source("a", TensorFact::shape(shapefact!(1, 1, S))) // NCT
            .unwrap();
        let conv = model.chain_default("conv", Conv::default()).unwrap();
        model.add_edge(OutletId::new(ker, 0), InletId::new(conv, 1)).unwrap();

        let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]);
        test_regular_against_pulse(model, input.into_dyn(), 2, 3, 2);
    }

    #[test]
    fn test_crop_at_start() {
        let mut model = Model::default();
        let _ = model
            .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S)))
            .unwrap();
        model.chain_default("slice", crate::ops::array::Slice::new(vec![(1, 0)])).unwrap();

        let input = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        test_regular_against_pulse(model, input.into_dyn(), 0, 2, 1);
    }
}
