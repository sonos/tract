use crate::internal::*;
use crate::ops::logic::sym_to_coord_axis;
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
    a.clone() + b.clone() - a.clone() * b.clone()
}

/// Check whether a TDim expression references coordinate symbol 🎯{axis}.
fn roi_mentions_axis(roi: &TDim, axis: usize) -> bool {
    roi.symbols().iter().any(|s| sym_to_coord_axis(s) == Some(axis))
}

/// Bubble output ROI to inputs for ops with natural axes mapping
/// (axis i in output corresponds to axis i in each input).
///
/// For broadcast dims (input dim = 1, output dim > 1): if the ROI expression
/// mentions that axis's coordinate symbol, we can't propagate (return None
/// for that input). Otherwise the ROI passes through.
pub fn bubble_roi_natural(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TVec<Option<TDim>>>> {
    let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
    let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };

    let mut result = tvec![];
    for input in &node.inputs {
        let input_fact = model.outlet_fact(*input)?;
        let can_propagate = input_fact
            .shape
            .iter()
            .zip(output_fact.shape.iter())
            .enumerate()
            .all(|(a, (idim, odim))| idim == odim || !roi_mentions_axis(roi, a));
        result.push(if can_propagate { Some(roi.clone()) } else { None });
    }
    Ok(Some(result))
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

            // Collect ROI demands from all nodes.
            let mut demands: HashMap<OutletId, Option<TDim>> = HashMap::new();

            for &node_id in &order {
                let node = &model.nodes()[node_id];
                let Some(input_rois) = node.op.as_typed().unwrap().input_roi(model, node)? else {
                    continue;
                };
                for (ix, roi) in input_rois.into_iter().enumerate() {
                    let outlet = node.inputs[ix];
                    match (demands.get(&outlet), &roi) {
                        (_, None) => {
                            demands.insert(outlet, None);
                        }
                        (Option::None, Some(roi)) => {
                            demands.insert(outlet, Some(roi.clone()));
                        }
                        (Some(None), Some(_)) => {}
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
