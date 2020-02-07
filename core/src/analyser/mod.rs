//! Model graph type inference.
use std::borrow::BorrowMut;
use std::collections::BTreeSet;

use crate::internal::*;
use crate::model::*;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
#[macro_use]
pub mod rules;

/// A graph analyser, along with its current state.
#[derive(new)]
pub struct Analyser<M: BorrowMut<InferenceModel>> {
    model: M,
}

impl<M: BorrowMut<InferenceModel>> Analyser<M> {
    /// Runs the entire analysis at once. Will not stop on error if obstinate is
    /// true.
    pub fn analyse_obstinate(&mut self, obstinate: bool) -> TractResult<bool> {
        let mut nodes_to_visit: BTreeSet<usize> =
            self.model.borrow().eval_order()?.iter().cloned().collect();
        let mut observed_outlets: HashMap<usize, Vec<OutletId>> = HashMap::new();
        let mut observers: HashMap<OutletId, TVec<usize>> = HashMap::new();
        for node in self.model.borrow().nodes() {
            if !nodes_to_visit.contains(&node.id) {
                nodes_to_visit.insert(node.id);
            }
            let observed = node.op.observe_outlets(self.model.borrow(), node)?;
            for outlet in &observed {
                observers.entry(*outlet).or_insert(tvec!()).push(node.id);
            }
            observed_outlets.insert(node.id, observed);
        }
        let mut first_error = None;
        let mut did_something = false;
        loop {
            trace!("Remaining nodes {}", nodes_to_visit.len());
            let node = match nodes_to_visit.iter().next() {
                None => break,
                Some(n) => *n,
            };
            match self.analyse_one(node) {
                Ok(changed_edges) => {
                    for (edge, _fact) in changed_edges {
                        did_something = true;
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
                        if let Some(observers) = observers.get(&edge) {
                            for observer in observers {
                                nodes_to_visit.insert(*observer);
                            }
                        }
                    }
                }
                Err(e) => {
                    let e = e.chain_err(|| {
                        format!("Failed analyse for node {}", self.model.borrow().node(node))
                    });
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
        Ok(did_something)
    }

    /// Tries to run a single step of the analysis, and returns whether
    /// there was any additional information gained during the step.
    pub fn analyse_one(&mut self, node: usize) -> TractResult<Vec<(OutletId, InferenceFact)>> {
        let mut changed_edges = vec![];
        {
            debug!("Starting step for {}", self.model.borrow().node(node));
            let observed_outlets: Vec<OutletId> = {
                let model = self.model.borrow();
                let node = model.node(node);
                node.op.observe_outlets(&model, &node)?
            };

            let inferred = {
                let (inputs, outputs) = self.model.borrow().node_facts(node)?;
                if outputs.len() != self.model.borrow().node(node).op.nboutputs().unwrap() {
                    bail!(
                        "Wrong number of outputs. Op says {}, node says {}.",
                        self.model.borrow().node(node).op.nboutputs().unwrap(),
                        outputs.len(),
                    )
                }
                let inputs: TVec<InferenceFact> = inputs.into_iter().cloned().collect();
                let outputs: TVec<InferenceFact> = outputs.into_iter().cloned().collect();
                let observed: TVec<(OutletId, InferenceFact)> = {
                    let model = self.model.borrow();
                    let node = model.node(node);
                    node.op
                        .observe_outlets(&model, &node)?
                        .iter()
                        .map(|o| model.outlet_fact(*o).map(|f| (*o, f.clone())))
                        .collect::<TractResult<_>>()?
                };
                if log_enabled!(log::Level::Trace) {
                    for (ix, i) in inputs.iter().enumerate() {
                        trace!("  Input  #{}: {:?}", ix, i);
                    }
                    for (ix, o) in outputs.iter().enumerate() {
                        trace!("  Output #{}: {:?}", ix, o);
                    }
                }

                let inputs: TVec<&InferenceFact> = inputs.iter().collect();
                let outputs: TVec<&InferenceFact> = outputs.iter().collect();
                let observed: TVec<&InferenceFact> = observed.iter().map(|p| &p.1).collect();

                self.model.borrow_mut().node_mut(node).op.infer(inputs, outputs, observed)?
            };

            let node = self.model.borrow().node(node);
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

            for (ix, &outlet) in observed_outlets.iter().enumerate() {
                let old_fact = self.model.borrow().outlet_fact(outlet)?;
                let new_fact = &inferred.2[ix];
                let unified = old_fact.unify(new_fact)?;
                if &unified != old_fact {
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
        let dt = TypeFactoid::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_only() {
        let dt1 = TypeFactoid::Only(DatumType::DT_FLOAT);
        let dt2 = TypeFactoid::Only(DatumType::DT_DOUBLE);
        assert!(unify_datum_type(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_datum_types_any_left() {
        let dt = TypeFactoid::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&TypeFactoid::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_any_right() {
        let dt = TypeFactoid::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &TypeFactoid::Any).unwrap(), dt);
    }

    #[test]
    fn unify_same_shape_1() {
        let s = ShapeFactoid::closed(vec![]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_2() {
        use super::DimFact::*;
        let s = ShapeFactoid::closed(vec![Any]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_3() {
        use super::DimFact::*;
        let s = ShapeFactoid::closed(vec![Only(1), Only(2)]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_different_shapes_1() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFactoid::closed(vec![Only(1)]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_2() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFactoid::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_3() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::open(vec![Only(1), Only(2)]);
        let s2 = ShapeFactoid::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_4() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::closed(vec![Any]);
        let s2 = ShapeFactoid::closed(vec![Any]);
        let sr = ShapeFactoid::closed(vec![Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_5() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::closed(vec![Any]);
        let s2 = ShapeFactoid::closed(vec![Only(1)]);
        let sr = ShapeFactoid::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_6() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::open(vec![]);
        let s2 = ShapeFactoid::closed(vec![Only(1)]);
        let sr = ShapeFactoid::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_7() {
        use super::DimFact::*;
        let s1 = ShapeFactoid::open(vec![Any, Only(2)]);
        let s2 = ShapeFactoid::closed(vec![Only(1), Any, Any]);
        let sr = ShapeFactoid::closed(vec![Only(1), Only(2), Any]);
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
