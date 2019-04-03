use std::borrow::BorrowMut;
use std::collections::BTreeSet;

use crate::model::*;
use crate::internal::*;

pub mod types;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
#[macro_use]
pub mod rules;

/// A graph analyser, along with its current state.
pub struct Analyser<M: BorrowMut<InferenceModel>> {
    model: M,
}

impl<M: BorrowMut<InferenceModel>> Analyser<M> {
    pub fn new(model: M) -> TractResult<Analyser<M>> {
        Ok(Analyser { model })
    }

    /// Runs the entire analysis at once. Will not stop on error if obstinate is
    /// true.
    pub fn analyse_obstinate(&mut self, obstinate: bool) -> TractResult<()> {
        let mut nodes_to_visit: BTreeSet<usize> =
            self.model.borrow().eval_order()?.iter().cloned().collect();
        let mut first_error = None;
        loop {
            trace!("Remaining nodes {}", nodes_to_visit.len());
            let node = match nodes_to_visit.iter().next() {
                None => break,
                Some(n) => *n,
            };
            match self.analyse_one(node) {
                Ok(changed_edges) => {
                    for (edge, _fact) in changed_edges {
                        trace!("Changed edge: {:?}", edge);
                        for dst in self.model.borrow().nodes()[edge.node].outputs[edge.slot]
                            .successors
                            .iter()
                        {
                            if dst.node != edge.node {
                                trace!("Inserting node dn {:?}", dst.node);
                                nodes_to_visit.insert(dst.node);
                            }
                        }
                        if edge.node != node {
                            trace!("Inserting node up {}", edge.node);
                            nodes_to_visit.insert(edge.node);
                        }
                    }
                }
                Err(e) => {
                    let e = format!("Analysing node {:?}, {:?}", node, e);
                    if !obstinate {
                        return Err(e.into());
                    }
                    debug!("{:?}", e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
            nodes_to_visit.remove(&node);
        }
        trace!("analyse done");
        if let Some(e) = first_error {
            Err(e)?
        }
        Ok(())
    }

    /// Tries to run a single step of the analysis, and returns whether
    /// there was any additional information gained during the step.
    pub fn analyse_one(&mut self, node: usize) -> TractResult<Vec<(OutletId, TensorFact)>> {
        let mut changed_edges = vec![];
        {
            let node = &self.model.borrow().nodes()[node];
            debug!("Starting step for #{} {} ({})", node.id, node.name, node.op.name(),);

            let (inputs, outputs) = self.model.borrow().node_facts(node.id)?;

            let inferred = node
                .op
                .infer(inputs, outputs)
                .map_err(|e| format!("while running inference on {} : {}", node, e))?;

            for (ix, &outlet) in node.inputs.iter().enumerate() {
                let inferred_fact = &inferred.0[ix];
                let old_fact = self.model.borrow().outlet_fact(outlet)?;
                let unified = inferred_fact
                    .unify(&old_fact)
                    .map_err(|e| format!("while unifying inputs of {} : {}", node, e))?;

                if &unified != old_fact {
                    debug!("  Refined {:?}: {:?} -> {:?}", outlet, old_fact, unified);
                    changed_edges.push((outlet, unified));
                }
            }

            for (ix, inferred_fact) in inferred.1.iter().enumerate() {
                let old_fact = self.model.borrow().outlet_fact(OutletId::new(node.id, ix))?;
                let unified = old_fact.unify(inferred_fact)?;

                if &unified != old_fact {
                    let outlet = OutletId::new(node.id, ix);
                    debug!("  Refined {:?}: {:?} -> {:?}", outlet, old_fact, unified);
                    changed_edges.push((outlet, unified));
                }
            }
        }
        for (outlet, fact) in &changed_edges {
            self.model.borrow_mut().set_outlet_fact(*outlet, fact.clone())?;
        }
        Ok(changed_edges)
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn unify_same_datum_type() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_only() {
        let dt1 = TypeFact::Only(DatumType::DT_FLOAT);
        let dt2 = TypeFact::Only(DatumType::DT_DOUBLE);
        assert!(unify_datum_type(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_datum_types_any_left() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&TypeFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_any_right() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &TypeFact::Any).unwrap(), dt);
    }

    #[test]
    fn unify_same_shape_1() {
        let s = ShapeFact::closed(vec![]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_2() {
        use super::DimFact::*;
        let s = ShapeFact::closed(vec![Any]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_3() {
        use super::DimFact::*;
        let s = ShapeFact::closed(vec![Only(1), Only(2)]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_different_shapes_1() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_2() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_3() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_4() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Any]);
        let s2 = ShapeFact::closed(vec![Any]);
        let sr = ShapeFact::closed(vec![Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_5() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Any]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        let sr = ShapeFact::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_6() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        let sr = ShapeFact::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_7() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![Any, Only(2)]);
        let s2 = ShapeFact::closed(vec![Only(1), Any, Any]);
        let sr = ShapeFact::closed(vec![Only(1), Only(2), Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_same_value() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_only() {
        use ndarray::prelude::*;
        let dt1 = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        let dt2 = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[2]))));
        assert!(unify_value(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_values_any_left() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&ValueFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_any_right() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &ValueFact::Any).unwrap(), dt);
    }
}
