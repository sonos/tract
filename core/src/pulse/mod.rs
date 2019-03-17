use crate::datum::TryInto;
use crate::ops::prelude::*;
use crate::ops::source::Source;

pub mod delay;

#[derive(Clone, Debug, PartialEq)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
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
                let id = target.add_source_fact(node.name.clone(), pulsed_fact)?;
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
        Ok(target)
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        crate::model::compact::compact(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = Model::default();
        let _a = model
            .add_source_fact("a", TensorFact::dt_shape(DatumType::F32, vec![1, 2, 3]))
            .unwrap();
        assert!(
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).is_err()
        );

        let mut model = Model::default();
        let _a = model
            .add_source_fact(
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
            .add_source_fact(
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

    #[test]
    fn test_simple_conv() {
        use crate::ops::nn::*;
        use ndarray::*;

        let mut model = Model::default();
        let ker = model.add_const("kernel", arr3(&[[[0.5f32, 1.0, -0.1]]]).into()).unwrap();
        let _ = model
            .add_source_fact("a", TensorFact::shape(shapefact!(1, 1, S))) // NCT
            .unwrap();
        let conv = model.chain("conv", Box::new(Conv::default())).unwrap();
        model.add_edge(OutletId::new(ker, 0), InletId::new(conv, 1)).unwrap();
        model.analyse().unwrap();
        assert_eq!(model.nodes().len(), 3);

        let input = [1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let t_input = Tensor::from(arr3(&[[input]]));

        let model = model.into_typed().unwrap().declutter().unwrap();

        println!("model: {:#?}", model);

        assert_eq!(model.nodes().len(), 2);
        let plan = crate::plan::SimplePlan::new(&model).unwrap();
        let outputs = plan.run(tvec!(t_input.clone())).unwrap();

        let pulse = 4;
        let pulsed = PulsedModel::new(&model.into_normalized().unwrap(), pulse).unwrap();
        assert_eq!(pulsed.nodes().len(), 3); // source - delay - conv
        assert_eq!(pulsed.fact(OutletId::new(2, 0)).unwrap().delay, 2);

        let pulsed_plan = crate::plan::SimplePlan::new(pulsed).unwrap();
        let mut state = crate::plan::SimpleState::new(&pulsed_plan).unwrap();
        let mut got: Vec<f32> = vec![];

        for p in 0..(input.len() / pulse) {
            let chunk = &input[(p * pulse)..((p + 1) * pulse)];
            let mut outputs = state
                .run(tvec!(ndarray::Array::from_shape_vec((1usize, 1, 4), chunk.to_vec())
                    .unwrap()
                    .into()))
                .unwrap();
            got.extend(outputs.remove(0).to_array_view::<f32>().unwrap().iter());
        }

        assert_eq!(&got[2..], outputs[0].to_array_view::<f32>().unwrap().as_slice().unwrap());
    }
}
