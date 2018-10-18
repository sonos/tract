use std::collections::HashMap;

use model::dsl::*;
use ops::prelude::*;

pub mod delay;

pub struct PulsifiedOp {
    pub op: Box<Op>,
}

impl PulsifiedOp {
    pub fn op(op: Box<Op>) -> PulsifiedOp {
        PulsifiedOp { op }
    }
}

pub struct PulsifiedModel {
    pub model: Model,
}

impl PulsifiedModel {
    pub fn new(old: &Model, pulse: usize) -> TfdResult<PulsifiedModel> {
        let mut model = Model::default();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        for old_id in old.eval_order()? {
            let PulsifiedOp { op, .. } = {
                let (in_fact, out_fact) = old.facts(old_id)?;
                old.node(old_id).op().pulsify(in_fact, out_fact, pulse)?
            };
            let new_id = model.add_node(old.node(old_id).name.clone(), op)?;
            for (ix, input) in old.node(old_id).inputs.iter().enumerate() {
                model.add_edge(mapping[&input], InletId::new(new_id, ix))?;
            }
            for ix in 0..model.node(new_id).op().noutputs() {
                mapping.insert(OutletId::new(old_id, ix), OutletId::new(new_id, ix));
            }
            model.analyse_one(new_id)?;
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
