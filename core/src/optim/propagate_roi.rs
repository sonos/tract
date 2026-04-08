use crate::internal::*;
use crate::optim::OptimizerSession;

/// Backward pass that propagates `region_of_interest` annotations by
/// calling `TypedOp::input_roi` on each node.
///
/// Ops can **introduce** ROIs (e.g. Iff reads its mask's uniform_tdim and
/// creates a ROI on the scores input) or **bubble** them (e.g. element-wise
/// ops pass an output ROI through to their inputs).
///
/// When multiple consumers of a wire produce different ROIs, they are merged
/// via boolean OR using De Morgan: `a ∨ b = a + b - a * b`.
/// If any consumer returns `None` for a wire (needs all positions), that wire
/// gets no ROI.
///
/// The pass iterates to fixpoint: introductions may enable further bubbling.
#[derive(Clone, Debug, Default)]
pub struct PropagateRoi;

/// Merge two ROI expressions via boolean OR: `a ∨ b = a + b - a * b`.
fn roi_union(a: &TDim, b: &TDim) -> TDim {
    if a == b {
        return a.clone();
    }
    // De Morgan: a + b - a*b
    a.clone() + b.clone() - a.clone() * b.clone()
}

impl super::TypedPass for PropagateRoi {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        _model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn run_direct(&mut self, model: &mut TypedModel) -> TractResult<bool> {
        let mut any_changed = false;
        loop {
            let order = model.eval_order()?;
            let mut changed = false;

            // Collect ROI demands from all nodes. For each outlet, track
            // whether all consumers agree (Some) or any needs everything (None).
            // We use Option<Option<TDim>>:
            //   - absent from map: no consumer has spoken yet
            //   - Some(Some(roi)): all consumers so far agree on a ROI
            //   - Some(None): at least one consumer needs everything
            let mut demands: HashMap<OutletId, Option<TDim>> = HashMap::new();

            for &node_id in &order {
                let node = &model.nodes()[node_id];
                let Some(input_rois) = node.op.as_typed().unwrap().input_roi(model, node)? else {
                    continue;
                };
                for (ix, roi) in input_rois.into_iter().enumerate() {
                    let outlet = node.inputs[ix];
                    match (demands.get(&outlet), &roi) {
                        // This consumer needs everything → wire has no ROI
                        (_, None) => {
                            demands.insert(outlet, None);
                        }
                        // First consumer with a ROI
                        (Option::None, Some(roi)) => {
                            demands.insert(outlet, Some(roi.clone()));
                        }
                        // Previous consumer already cancelled ROI
                        (Some(None), Some(_)) => {}
                        // Merge with existing ROI via OR
                        (Some(Some(existing)), Some(new)) => {
                            demands.insert(outlet, Some(roi_union(existing, new)));
                        }
                    }
                }
            }

            // Apply demands to model facts.
            for (outlet, demand) in demands {
                if let Some(roi) = demand {
                    let fact = &mut model.nodes_mut()[outlet.node].outputs[outlet.slot].fact;
                    if fact.region_of_interest.as_ref() != Some(&roi) {
                        fact.region_of_interest = Some(roi);
                        changed = true;
                    }
                }
            }

            any_changed |= changed;
            if !changed {
                break;
            }
        }
        Ok(any_changed)
    }
}
