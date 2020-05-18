use crate::errors::*;
use tract_core::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Io {
    None,
    Short,
    Long,
}

impl Default for Io {
    fn default() -> Io {
        Io::Short
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisplayParams {
    pub konst: bool,
    pub invariants: bool,
    pub quiet: bool,
    pub natural_order: bool,
    pub debug_op: bool,
    pub cost: bool,
    pub profile: bool,
    pub node_ids: Option<Vec<TVec<(usize, String)>>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub expect_canonic: bool,
    pub outlet_labels: bool,
    pub io: Io,
    pub info: bool,
    pub left_column_width: usize,
}

impl DisplayParams {
    pub fn filter(
        &self,
        model: &dyn Model,
        scope: &[(usize, String)],
        node_id: usize,
    ) -> CliResult<bool> {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.iter().any(|n| {
                n.len() == scope.len() + 1
                    && &n[0..scope.len()] == scope
                    && n.last().unwrap().0 == node_id
            }));
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(model.node_name(node_id).starts_with(&*node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(model.node_op(node_id).name().starts_with(op_name));
        }
        /*
        if let Some(successor) = self.successors {
        return Ok(model.node_inputs(node_id).iter().any(|i| i.node == successor));
        }
        */
        Ok(model.node_op(node_id).name() != "Const" || self.konst)
    }

    pub fn should_draw(&self) -> bool {
        !self.natural_order
    }
}
