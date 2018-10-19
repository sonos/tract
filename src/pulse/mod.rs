use ops::source::Source;
use std::collections::HashMap;

use model::dsl::*;
use ops::prelude::*;

pub mod delay;

#[derive(Clone, Debug)]
pub struct PulsedTensorFact {
    pub fact: TensorFact,
    pub actual_size: usize,
    pub delay: usize,
}

impl PulsedTensorFact {
    pub fn axis(&self) -> TfdResult<usize> {
        self.fact
            .stream_info()?
            .map(|si| si.axis)
            .ok_or("PulsedTensorFact not pulsed".into())
    }

    pub fn stream_input_shape(&self) -> TfdResult<Vec<TDim>> {
        self.fact.shape.concretize().ok_or("Can not pulsify an untyped graph".into())
    }
}

pub struct PulsifiedOp {
    pub op: Box<Op>,
    pub outputs: TVec<PulsedTensorFact>,
}

impl PulsifiedOp {
    pub fn op(op: Box<Op>, fact: PulsedTensorFact) -> PulsifiedOp {
        PulsifiedOp {
            op,
            outputs: tvec!(fact),
        }
    }
}

pub struct PulsifiedModel {
    pub model: Model,
}

impl PulsifiedModel {
    pub fn new(old: &Model, pulse: usize) -> TfdResult<PulsifiedModel> {
        let mut model = Model::default();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        let mut facts: HashMap<OutletId, PulsedTensorFact> = HashMap::new();
        for old_id in old.eval_order()? {
            debug!("pulsify {:?}", old.node(old_id));
            if let Some(source) = old.node(old_id).op_as::<Source>() {
                let node = old.node(old_id);
                let mut fact = node.outputs[0].fact.clone();
                let axis = fact
                    .stream_info()?
                    .ok_or("Found a non straming source in pulsify")?
                    .axis;
                fact.shape.dims[axis] = pulse.to_dim().into();
                let id = model.add_source_fact(node.name.clone(), fact.clone())?;
                let pulsed_fact = PulsedTensorFact {
                    fact,
                    actual_size: pulse,
                    delay: 0,
                };
                facts.insert(OutletId::new(id, 0), pulsed_fact);
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let inputs = old
                    .node(old_id)
                    .inputs
                    .iter()
                    .map(|i| &facts[&mapping[i]])
                    .collect();
                trace!("  inputs: {:?}", inputs);
                let pulsed = old.node(old_id).op().pulsify(inputs)?;
                let new_id = model.add_node(old.node(old_id).name.clone(), pulsed.op)?;
                for (ix, input) in old.node(old_id).inputs.iter().enumerate() {
                    model.add_edge(mapping[&input], InletId::new(new_id, ix))?;
                }
                for ix in 0..model.node(new_id).op().noutputs() {
                    mapping.insert(OutletId::new(old_id, ix), OutletId::new(new_id, ix));
                }
                model.analyse_one(new_id)?;
            }
            trace!("  mappings: {:?}", mapping);
            trace!("  facts: {:?}", facts);
        }
        Ok(PulsifiedModel { model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = Model::default();
        let _a = model
            .add_source_fact("a", TensorFact::shape(vec![1, 2, 3]))
            .unwrap();
        assert!(PulsifiedModel::new(&model, 4).is_err());

        let mut model = Model::default();
        let _a = model
            .add_source_fact(
                "a",
                TensorFact::shape(vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            ).unwrap();
        let mut pulse = PulsifiedModel::new(&model, 4).unwrap();
        pulse.model.analyse().unwrap();
        assert_eq!(
            pulse.model.fact(OutletId::new(0, 0)).unwrap(),
            &TensorFact::shape(vec!(1, 4, 3))
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = Model::default();
        let _a = model
            .add_source_fact(
                "a",
                TensorFact::shape(vec![TDim::s(), 2.to_dim(), 3.to_dim()]),
            ).unwrap();
        let add = model
            .chain("add", Box::new(::ops::math::Add::default()))
            .unwrap();
        let b = model.add_const("b", Tensor::from(12)).unwrap();
        model
            .add_edge(OutletId::new(b, 0), InletId::new(add, 1))
            .unwrap();

        let pulse = PulsifiedModel::new(&model, 4).unwrap();

        assert_eq!(
            pulse.model.input_fact().unwrap(),
            &TensorFact::shape(vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.model.output_fact().unwrap(),
            &TensorFact::shape(vec!(4, 2, 3))
        );
    }
}
