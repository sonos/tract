//! Slice a `TypedModel` into standalone sub-models.
//!
//! Composes the existing graph primitives (`set_input_outlets`,
//! `select_output_outlets`, [`IntoTranslator::translate_model`], `compact`) and
//! adds the boundary validation that makes cutting safe: a sub-graph whose
//! declared inputs do not dominate its outputs is rejected instead of silently
//! producing an under-specified model. Adds no ops.

use std::collections::HashSet;

use crate::internal::*;
use crate::model::translator::{IntoTranslator, Translate};
use crate::ops::konst::Const;

/// Extract the sub-graph of `model` computing `outputs`, with that cone bounded
/// by `inputs`: every interior outlet listed in `inputs` becomes a fresh
/// `Source` carrying its fact, nodes off every `inputs`→`outputs` path are
/// pruned, consumed consts are copied in, and the `SymbolScope` is carried over.
///
/// Returns a standalone runnable `TypedModel`. Errors if the `outputs` cone
/// reaches a `Source` (or other input-less non-const op) that is not listed in
/// `inputs` — i.e. a boundary tensor the caller failed to declare (the mask /
/// positional-encoding / external-KV trap, and skip connections that span the
/// cut). Symbol *resolution* stays the caller's responsibility: feed the
/// extracted model shapes consistent with the parent's.
pub fn extract_subgraph(
    model: &TypedModel,
    inputs: &[OutletId],
    outputs: &[OutletId],
) -> TractResult<TypedModel> {
    validate_boundary(model, inputs, outputs)?;
    let mut m = model.clone();
    m.set_input_outlets(inputs)?;
    m.select_output_outlets(outputs)?;
    let mut sub = IntoTranslator.translate_model(&m)?;
    sub.compact()?;
    Ok(sub)
}

/// Split `model` into a linear pipeline of `boundaries.len() + 1` stages.
/// `boundaries[i]` is the set of outlets crossing the cut between stage `i` and
/// stage `i + 1` (usually a single residual outlet). Stage 0 keeps the model's
/// inputs; the last stage keeps its outputs.
///
/// Each stage is built with [`extract_subgraph`], so a skip connection that
/// spans more than one stage surfaces as an "undeclared boundary input" error on
/// the stage that needs it, rather than a silently wrong split.
pub fn split_pipeline(
    model: &TypedModel,
    boundaries: &[Vec<OutletId>],
) -> TractResult<Vec<TypedModel>> {
    let model_inputs = model.input_outlets()?.to_vec();
    let model_outputs = model.output_outlets()?.to_vec();
    let mut stages = Vec::with_capacity(boundaries.len() + 1);
    for i in 0..=boundaries.len() {
        let ins: &[OutletId] = if i == 0 { &model_inputs } else { &boundaries[i - 1] };
        let outs: &[OutletId] = if i == boundaries.len() { &model_outputs } else { &boundaries[i] };
        stages.push(extract_subgraph(model, ins, outs)?);
    }
    Ok(stages)
}

/// Reject a boundary whose `inputs` do not dominate `outputs`: walk the outputs'
/// dependency cone backward, stopping at declared inputs; any input-less node
/// reached that is not a `Const` is a required boundary the caller omitted.
fn validate_boundary(
    model: &TypedModel,
    inputs: &[OutletId],
    outputs: &[OutletId],
) -> TractResult<()> {
    let declared: HashSet<usize> = inputs.iter().map(|o| o.node).collect();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = outputs.iter().map(|o| o.node).collect();
    while let Some(n) = stack.pop() {
        if declared.contains(&n) || !seen.insert(n) {
            continue;
        }
        let node = model.node(n);
        if node.inputs.is_empty() && !node.op_is::<Const>() {
            bail!(
                "cannot extract sub-graph: outputs depend on `{}` ({}), which is not a declared \
                 boundary input; add its outlet to `inputs`",
                node.name,
                node.op().name()
            );
        }
        stack.extend(node.inputs.iter().map(|i| i.node));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid_chain(k: usize) -> (TypedModel, Vec<OutletId>) {
        let mut m = TypedModel::default();
        let mut wire = m.add_source("x", f32::fact([4])).unwrap();
        let mut outs = vec![wire];
        for i in 0..k {
            wire = m.wire_node(format!("s{i}"), crate::ops::nn::sigmoid(), &[wire]).unwrap()[0];
            outs.push(wire);
        }
        m.select_output_outlets(&[wire]).unwrap();
        (m.into_decluttered().unwrap(), outs)
    }

    fn run(m: &TypedModel, x: &Tensor) -> Tensor {
        m.clone().into_runnable().unwrap().run(tvec!(x.clone().into_tvalue())).unwrap()[0]
            .clone()
            .into_tensor()
    }

    #[test]
    fn pipeline_matches_whole_at_every_cut() {
        let x = tensor1(&[0.1f32, -0.7, 2.0, 0.3]);
        let (full, outs) = sigmoid_chain(5);
        let whole = run(&full, &x);
        // Cut after each interior sigmoid; the 2-stage pipeline must reproduce it.
        for cut in 1..outs.len() - 1 {
            let stages = split_pipeline(&full, &[vec![outs[cut]]]).unwrap();
            assert_eq!(stages.len(), 2);
            let mid = run(&stages[0], &x);
            let got = run(&stages[1], &mid);
            assert_eq!(got, whole, "mismatch cutting after node index {cut}");
        }
    }

    #[test]
    fn extract_prunes_to_the_slice() {
        let (full, outs) = sigmoid_chain(6);
        // Slice [after s1 .. s4]: a standalone 3-sigmoid model fed at the s1 boundary.
        let slice = extract_subgraph(&full, &[outs[2]], &[outs[5]]).unwrap();
        assert!(slice.nodes().len() < full.nodes().len());
        let _ = run(&slice, &tensor1(&[0.2f32, 0.2, 0.2, 0.2]));
    }

    #[test]
    fn undeclared_boundary_input_is_rejected() {
        let mut m = TypedModel::default();
        let a = m.add_source("a", f32::fact([4])).unwrap();
        let b = m.add_source("b", f32::fact([4])).unwrap();
        let y = m.wire_node("y", crate::ops::math::add(), &[a, b]).unwrap()[0];
        m.select_output_outlets(&[y]).unwrap();
        let m = m.into_decluttered().unwrap();
        // Declaring only `a` leaves `b` (a Source) undeclared → error.
        let a = m.node_by_name("a").unwrap().id;
        assert!(
            extract_subgraph(&m, &[OutletId::from((a, 0))], &m.output_outlets().unwrap().to_vec())
                .is_err()
        );
    }
}
