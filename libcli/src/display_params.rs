use crate::model::Model;
use tract_core::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Io {
    None,
    #[default]
    Short,
    Long,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisplayParams {
    pub konst: bool,
    pub invariants: bool,
    pub quiet: bool,
    pub natural_order: bool,
    pub opt_ram_order: bool,
    pub debug_op: bool,
    pub cost: bool,
    pub tmp_mem_usage: bool,
    pub profile: bool,
    pub folded: bool,
    pub node_ids: Option<Vec<TVec<(usize, String)>>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub expect_core: bool,
    pub outlet_labels: bool,
    pub io: Io,
    pub json: bool,
    pub info: bool,
    pub left_column_width: usize,
    pub mm: bool,
}

impl DisplayParams {
    pub fn filter(
        &self,
        model: &dyn Model,
        scope: &[(usize, String)],
        node_id: usize,
    ) -> TractResult<bool> {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.iter().any(|n| {
                n.len() == scope.len() + 1
                    && &n[0..scope.len()] == scope
                    && n.last().unwrap().0 == node_id
            }));
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(model.node_name(node_id).starts_with(node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(model.node_op_name(node_id).starts_with(op_name));
        }
        /*
        if let Some(successor) = self.successors {
        return Ok(model.node_inputs(node_id).iter().any(|i| i.node == successor));
        }
        */
        Ok(!model.node_const(node_id) || self.konst)
    }

    pub fn should_draw(&self) -> bool {
        !self.natural_order
    }

    pub fn order(&self, model: &dyn Model) -> TractResult<Vec<usize>> {
        if self.natural_order {
            Ok((0..model.nodes_len()).collect())
        } else if self.opt_ram_order {
            model.eval_order_opt_ram()
        } else {
            model.eval_order()
        }
    }
}
