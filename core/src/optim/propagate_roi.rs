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

/// Bubble output ROI to inputs using the op's axes_mapping.
///
/// For each input, builds a coordinate substitution map from the axes mapping:
/// each output axis that appears in this input gets 🎯{out_pos} → 🎯{in_pos}.
/// If any ROI coordinate symbol has no corresponding input axis (contracted,
/// broadcast from dim=1, or absent), returns None for that input.
pub fn bubble_roi(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TVec<Option<TDim>>>> {
    let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
    rule_if_some!(roi = &output_fact.region_of_interest);

    let input_facts: TVec<&TypedFact> =
        node.inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<_>>()?;
    let output_facts = tvec![output_fact];
    let inputs_ref: Vec<&TypedFact> = input_facts.iter().copied().collect();
    let outputs_ref: Vec<&TypedFact> = output_facts.iter().copied().collect();
    let mapping = node.op.as_typed().unwrap().axes_mapping(&inputs_ref, &outputs_ref)?;

    // Collect ROI coordinate symbols and their output axis positions.
    let roi_coord_syms: Vec<(usize, Symbol)> =
        roi.symbols().into_iter().filter_map(|s| sym_to_coord_axis(&s).map(|k| (k, s))).collect();

    let remap_for_input = |input_ix: usize| -> Option<TDim> {
        let mut sub_map: HashMap<Symbol, TDim> = HashMap::new();
        for (out_pos, sym) in &roi_coord_syms {
            let logical = mapping
                .iter_all_axes()
                .find(|a| a.outputs.first().is_some_and(|o| o.contains(out_pos)))?;
            if logical.inputs[input_ix].is_empty() {
                return None;
            }
            let in_pos = logical.inputs[input_ix][0];
            if input_facts[input_ix].shape[in_pos] != output_fact.shape[*out_pos] {
                return None;
            }
            if in_pos != *out_pos {
                let scope = sym.scope()?;
                sub_map.insert(sym.clone(), TDim::Sym(scope.coord_sym(in_pos)));
            }
        }
        if sub_map.is_empty() { Some(roi.clone()) } else { roi.substitute_all(&sub_map).ok() }
    };
    let result: TVec<Option<TDim>> = (0..node.inputs.len()).map(|ix| remap_for_input(ix)).collect();
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

        Ok(changed)
    }
}
