//! Split a `TypedModel` at chosen activation edges into standalone sub-models.
//!
//! No dedicated splitter exists in tract; this composes `set_input_outlets` /
//! `select_output_outlets` / `IntoTranslator::translate_model` / `compact`.
//! Each stage receives its own copy of only the consts/weights its slice needs
//! — that per-stage weight ownership is the memory-capacity win.

use anyhow::Result;
use tract_core::model::split_pipeline;
use tract_core::prelude::*;

use crate::protocol::TensorContract;

/// One partition: a standalone runnable sub-model plus its pinned I/O contracts.
pub struct Stage {
    pub model: TypedModel,
    pub inputs: Vec<TensorContract>,
    pub outputs: Vec<TensorContract>,
}

/// Capture a fact as a wire contract (dtype + per-dim strings, symbols kept).
pub fn contract_of(fact: &TypedFact) -> TensorContract {
    TensorContract {
        dt: format!("{:?}", fact.datum_type),
        shape: fact.shape.to_tvec().iter().map(|d| d.to_string()).collect(),
    }
}

/// Split `full` at `cut_nodes`, each naming the node whose slot-0 output is a
/// boundary activation. Cuts must be given in eval order. Returns
/// `cut_nodes.len() + 1` stages, in pipeline order. Delegates the graph surgery
/// and cut validation to `tract_core::model::split_pipeline`.
pub fn partition(full: &TypedModel, cut_nodes: &[impl AsRef<str>]) -> Result<Vec<Stage>> {
    let mut boundaries: Vec<Vec<OutletId>> = Vec::with_capacity(cut_nodes.len());
    for n in cut_nodes {
        let id = full.node_by_name(n.as_ref())?.id;
        boundaries.push(vec![OutletId::from((id, 0))]);
    }
    let contracts = |sub: &TypedModel, outlets: &[OutletId]| {
        outlets.iter().map(|o| contract_of(sub.outlet_fact(*o).unwrap())).collect()
    };
    Ok(split_pipeline(full, &boundaries)?
        .into_iter()
        .map(|sub| {
            let inputs = contracts(&sub, sub.input_outlets().unwrap());
            let outputs = contracts(&sub, sub.output_outlets().unwrap());
            Stage { model: sub, inputs, outputs }
        })
        .collect())
}

/// Bytes a const tensor really occupies. Block-quant weights are stored packed, so
/// their element count times the (dequantized) datum type overstates them ~8x for
/// q40 — measure the packed blob instead.
pub(crate) fn tensor_bytes(t: &Tensor) -> usize {
    if let Some(bq) = t.storage_as::<tract_core::tract_linalg::block_quant::BlockQuantStorage>() {
        return bq.value().len();
    }
    t.len() * t.datum_type().size_of()
}

/// Total bytes of constant (weight) tensors a stage owns — the metric that
/// makes the memory split concrete: each stage holds only its slice's weights.
pub fn const_bytes(model: &TypedModel) -> usize {
    model
        .nodes()
        .iter()
        .flat_map(|n| n.outputs.iter())
        .filter_map(|o| o.fact.konst.as_ref())
        .map(|t| tensor_bytes(t))
        .sum()
}
