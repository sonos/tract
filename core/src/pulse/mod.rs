use crate::ops::source::Source;
use crate::ops::prelude::*;

pub mod delay;

#[derive(Clone, Debug, PartialEq)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
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

#[derive(Clone, Debug, new)]
pub struct PulsifiedOp {
    pub op: Box<Op>,
    pub outputs: TVec<PulsedTensorFact>,
}

pub fn pulsify(
    old: &NormalizedModel,
    pulse: usize,
) -> TractResult<(NormalizedModel, PulsedTensorFact, PulsedTensorFact)> {
    let mut p_model = PulsifiedModel::new(old, pulse)?;
    let in_id = p_model.model.inputs()?[0];
    let out_id = p_model.model.outputs()?[0];
    let in_fact: PulsedTensorFact = p_model.facts.remove(&in_id).unwrap();
    let out_fact: PulsedTensorFact = p_model.facts.remove(&out_id).unwrap();
    Ok((p_model.model, in_fact, out_fact))
}

#[derive(Clone, Debug)]
struct PulsifiedModel {
    model: NormalizedModel,
    facts: HashMap<OutletId, PulsedTensorFact>,
}

impl PulsifiedModel {
    fn new(old: &NormalizedModel, pulse: usize) -> TractResult<PulsifiedModel> {
        let mut model = NormalizedModel::default();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        let mut facts: HashMap<OutletId, PulsedTensorFact> = HashMap::new();
        for old_id in old.eval_order()? {
            if old.node(old_id).op_as::<Source>().is_some() {
                let node = old.node(old_id);
                let pulsed_fact =
                    PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
                let id = model.add_source_fact(node.name.clone(), pulsed_fact.to_pulse_fact())?;
                facts.insert(OutletId::new(id, 0), pulsed_fact);
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let pulsed_chain = {
                    let inputs =
                        old.node(old_id).inputs.iter().map(|i| &facts[&mapping[i]]).collect();
                    trace!("  inputs: {:?}", inputs);
                    old.node(old_id).op().pulsify(inputs)?
                };
                let mut previous = None;
                let count = pulsed_chain.len();
                for (ix, pulsed) in pulsed_chain.into_iter().enumerate() {
                    let PulsifiedOp { op, outputs } = pulsed;
                    let name = if ix == count - 1 {
                        old.node(old_id).name.clone()
                    } else {
                        format!("{}#{}", old.node(old_id).name, ix)
                    };
                    let new_id = model.add_node(
                        name,
                        op,
                        outputs.iter().map(|o| o.to_pulse_fact()).collect(),
                    )?;
                    if let Some(prev) = previous {
                        model.add_edge(OutletId::new(prev, 0), InletId::new(new_id, 0))?;
                    } else {
                        for (ix, input) in old.node(old_id).inputs.iter().enumerate() {
                            model.add_edge(mapping[&input], InletId::new(new_id, ix))?;
                        }
                    };
                    previous = Some(new_id);
                    for (ix, output_fact) in outputs.into_iter().enumerate() {
                        model.set_fact(OutletId::new(new_id, ix), output_fact.to_pulse_fact())?;
                        facts.insert(OutletId::new(new_id, ix), output_fact);
                        mapping.insert(OutletId::new(old_id, ix), OutletId::new(new_id, ix));
                    }
                }
            }
        }
        Ok(PulsifiedModel { model, facts })
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
        assert!(PulsifiedModel::new(&model.into_declutterd().unwrap(), 4).is_err());

        let mut model = Model::default();
        let _a = model
            .add_source_fact(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            )
            .unwrap();
        let pulse = PulsifiedModel::new(&model.into_declutterd().unwrap(), 4).unwrap();
        assert_eq!(
            pulse.model.fact(OutletId::new(0, 0)).unwrap().to_tensor_fact(),
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

        let pulse = PulsifiedModel::new(&model.into_declutterd().unwrap(), 4).unwrap();

        assert_eq!(
            pulse.model.input_fact().unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.model.output_fact().unwrap().to_tensor_fact(),
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

        let model = model.into_declutterd().unwrap();

        assert_eq!(model.nodes().len(), 2);
        let plan = crate::plan::SimplePlan::new(&model).unwrap();
        let outputs = plan.run(tvec!(t_input.clone())).unwrap();

        let pulse = 4;
        let pulsed = PulsifiedModel::new(&model, pulse).unwrap();
        assert_eq!(pulsed.model.nodes().len(), 3); // source - delay - conv
        assert_eq!(pulsed.facts[&OutletId::new(2, 0)].delay, 2);

        let pulsed_plan = crate::plan::SimplePlan::new(pulsed.model).unwrap();
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
