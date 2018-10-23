use ops::source::Source;
use std::collections::HashMap;

use model::dsl::*;
use ops::prelude::*;

pub mod delay;

#[derive(Clone, Debug)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: Vec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl PulsedTensorFact {
    pub fn from_tensor_fact_pulse(tf: &TensorFact, pulse: usize) -> TfdResult<PulsedTensorFact> {
        let dt = tf
            .datum_type
            .concretize()
            .ok_or("Can not use pulse a tensor with no type")?;
        let axis = tf
            .stream_info()?
            .ok_or("Can not pulse a tensor with no streaming dim")?
            .axis;
        let shape = tf
            .shape
            .concretize()
            .ok_or("Can not pulse a tensor with unknown shape")?;
        let dim = shape[axis];
        let shape = shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| {
                if ix == axis {
                    Ok(pulse)
                } else {
                    d.to_integer().map(|d| d as usize)
                }
            }).collect::<TfdResult<_>>()?;
        Ok(PulsedTensorFact {
            dt,
            shape,
            axis,
            dim,
            delay: 0,
        })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis]
    }

    pub fn to_little_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.dt, self.shape.clone())
    }

    pub fn big_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| {
                if ix == self.axis {
                    self.dim
                } else {
                    d.to_dim()
                }
            }).collect()
    }

    pub fn to_big_fact(&self) -> TensorFact {
        let mut fact = self.to_little_fact();
        fact.shape.dims[self.axis] = self.dim.into();
        fact
    }
}

#[derive(Clone, Debug, new)]
pub struct PulsifiedOp {
    pub op: Box<Op>,
    pub outputs: TVec<PulsedTensorFact>,
}

pub fn pulsify(
    old: &Model,
    pulse: usize,
) -> TfdResult<(Model, PulsedTensorFact, PulsedTensorFact)> {
    let p_model = PulsifiedModel::new(old, pulse)?;
    let in_fact: PulsedTensorFact = p_model.input_fact()?.clone();
    let out_fact: PulsedTensorFact = p_model.output_fact()?.clone();
    Ok((p_model.model, in_fact, out_fact))
}

#[derive(Clone, Debug)]
pub struct PulsifiedModel {
    pub model: Model,
    pub facts: HashMap<OutletId, PulsedTensorFact>,
}

impl PulsifiedModel {
    pub fn new(old: &Model, pulse: usize) -> TfdResult<PulsifiedModel> {
        let mut model = Model::default();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        let mut facts: HashMap<OutletId, PulsedTensorFact> = HashMap::new();
        for old_id in old.eval_order()? {
            if old.node(old_id).op_as::<Source>().is_some() {
                let node = old.node(old_id);
                let pulsed_fact =
                    PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
                let id = model.add_source_fact(node.name.clone(), pulsed_fact.to_little_fact())?;
                facts.insert(OutletId::new(id, 0), pulsed_fact);
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let pulsed_chain = {
                    let inputs = old
                        .node(old_id)
                        .inputs
                        .iter()
                        .map(|i| &facts[&mapping[i]])
                        .collect();
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
                    let new_id = model.add_node(name, op)?;
                    if let Some(prev) = previous {
                        model.add_edge(OutletId::new(prev, 0), InletId::new(new_id, 0))?;
                    } else {
                        for (ix, input) in old.node(old_id).inputs.iter().enumerate() {
                            model.add_edge(mapping[&input], InletId::new(new_id, ix))?;
                        }
                    };
                    previous = Some(new_id);
                    for (ix, output_fact) in outputs.into_iter().enumerate() {
                        model.set_fact(OutletId::new(new_id, ix), output_fact.to_little_fact())?;
                        facts.insert(OutletId::new(new_id, ix), output_fact);
                        mapping.insert(OutletId::new(old_id, ix), OutletId::new(new_id, ix));
                    }
                }
            }
        }
        Ok(PulsifiedModel { model, facts })
    }

    pub fn output_fact(&self) -> TfdResult<&PulsedTensorFact> {
        let output = self.model.outputs()?[0];
        Ok(&self.facts[&output])
    }

    pub fn input_fact(&self) -> TfdResult<&PulsedTensorFact> {
        let input = self.model.inputs()?[0];
        Ok(&self.facts[&input])
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
        assert!(PulsifiedModel::new(&model, 4).is_err());

        let mut model = Model::default();
        let _a = model
            .add_source_fact(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            ).unwrap();
        let mut pulse = PulsifiedModel::new(&model, 4).unwrap();
        pulse.model.analyse().unwrap();
        assert_eq!(
            pulse.model.fact(OutletId::new(0, 0)).unwrap(),
            &TensorFact::dt_shape(DatumType::F32, vec!(1, 4, 3))
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = Model::default();
        let _a = model
            .add_source_fact(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![TDim::s(), 2.to_dim(), 3.to_dim()]),
            ).unwrap();

        let pulse = PulsifiedModel::new(&model, 4).unwrap();

        assert_eq!(
            pulse.model.input_fact().unwrap(),
            &TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.model.output_fact().unwrap(),
            &TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
    }

    #[test]
    fn test_simple_conv() {
        use ndarray::*;
        use ops::nn::*;

        let mut model = Model::default();
        let ker = model
            .add_const("kernel", arr3(&[[[0.5f32, 1.0, -0.1]]]).into())
            .unwrap();
        let _ = model
            .add_source_fact("a", TensorFact::shape(shapefact!(1, 1, S))) // NCT
            .unwrap();
        let conv = model.chain("conv", Box::new(Conv::default())).unwrap();
        model
            .add_edge(OutletId::new(ker, 0), InletId::new(conv, 1))
            .unwrap();
        model.analyse().unwrap();
        assert_eq!(model.nodes().len(), 3);

        let input = [1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let t_input = Tensor::from(arr3(&[[input]]));

        let model = model.into_optimized().unwrap();
        assert_eq!(model.nodes().len(), 2);
        let plan = ::plan::SimplePlan::new(&model).unwrap();
        let outputs = plan.run(tvec!(t_input.clone())).unwrap();

        let pulse = 4;
        let pulsed = PulsifiedModel::new(&model, pulse).unwrap();
        assert_eq!(pulsed.model.nodes().len(), 3); // source - delay - conv
        assert_eq!(pulsed.facts[&OutletId::new(2, 0)].delay, 2);

        let pulsed_plan = ::plan::SimplePlan::new(pulsed.model).unwrap();
        let mut state = ::plan::SimpleState::new(&pulsed_plan).unwrap();
        let mut got: Vec<f32> = vec![];

        for p in 0..(input.len() / pulse) {
            let chunk = &input[(p * pulse)..((p + 1) * pulse)];
            state.reset().unwrap();
            state
                .set_input(0, Tensor::f32s(&[1, 1, 4], chunk).unwrap())
                .unwrap();
            state.eval_all_in_order().unwrap();
            got.extend(state.take_outputs().unwrap()[0].as_f32s().unwrap().iter());
        }

        assert_eq!(&got[2..], outputs[0].as_f32s().unwrap().as_slice().unwrap());
    }
}
