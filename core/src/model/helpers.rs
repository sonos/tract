use crate::ops::binary::{BinMiniOp, TypedBinOp};
use crate::ops::konst::Const;
use crate::prelude::*;
use tract_data::internal::Approximation;

pub trait TypedModelHelpers {
    fn next_node(&self, node: &TypedNode) -> Option<&TypedNode>;
    fn previous_node(&self, node: &TypedNode) -> Option<&TypedNode>;
    fn previous_nodes(&self, node: &TypedNode) -> TVec<&TypedNode>;
    fn collect_const_inputs<'a>(&'a self, node: &TypedNode) -> TVec<&'a Const>;
    fn single_prev_node_as<O: TypedOp>(&self, node: &TypedNode) -> Option<(usize, &TypedNode)>;
    fn matches_single_input_const(&self, node: &TypedNode, konst: f32) -> bool;
    fn find_succ_bin_with_const<B: BinMiniOp>(
        &self,
        node: &TypedNode,
        konst: f32,
    ) -> Option<&TypedNode>;
    fn find_succ_bin_with_outlet<B: BinMiniOp>(
        &self,
        node: &TypedNode,
        outlet_id: &OutletId,
    ) -> Option<&TypedNode>;
}

impl TypedModelHelpers for TypedModel {
    fn next_node(&self, node: &TypedNode) -> Option<&TypedNode> {
        if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return None;
        }
        let succ = node.outputs[0].successors[0];
        Some(&self.nodes()[succ.node])
    }

    fn previous_node(&self, node: &TypedNode) -> Option<&TypedNode> {
        if node.inputs.len() != 1 {
            return None;
        }
        Some(&self.nodes()[node.inputs[0].node])
    }

    fn previous_nodes(&self, node: &TypedNode) -> TVec<&TypedNode> {
        node.inputs.iter().map(|n| &self.nodes()[n.node]).collect()
    }

    fn collect_const_inputs<'a>(&'a self, node: &TypedNode) -> TVec<&'a Const> {
        node.inputs
            .iter()
            .filter_map(|i| {
                let prec = &self.nodes()[i.node];
                prec.op_as::<Const>()
            })
            .collect::<TVec<_>>()
    }

    fn single_prev_node_as<O: TypedOp>(&self, node: &TypedNode) -> Option<(usize, &TypedNode)> {
        let prev_nodes = node
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(in_idx, i)| {
                let prec = &self.nodes()[i.node];
                prec.op_is::<O>().then_some((in_idx, prec))
            })
            .collect::<TVec<_>>();

        if prev_nodes.len() != 1 { None } else { Some(prev_nodes[0]) }
    }

    fn matches_single_input_const(&self, node: &TypedNode, konst: f32) -> bool {
        let consts = self.collect_const_inputs(node);
        if consts.len() != 1 {
            return false;
        }
        let Ok(in_const) = consts[0].val().cast_to_dt(DatumType::F32) else {
            return false;
        };
        let Ok(in_const) = in_const.to_scalar_tensor() else {
            return false;
        };
        in_const
            .close_enough(&tract_data::prelude::tensor0(konst), Approximation::Approximate)
            .is_ok()
    }

    fn find_succ_bin_with_const<B: BinMiniOp>(
        &self,
        node: &TypedNode,
        konst: f32,
    ) -> Option<&TypedNode> {
        let succ = self.single_succ(node.id).ok()??;
        let succ_op = succ.op_as::<TypedBinOp>()?;
        (succ_op.0.is::<B>() && self.matches_single_input_const(succ, konst)).then_some(succ)
    }

    fn find_succ_bin_with_outlet<B: BinMiniOp>(
        &self,
        node: &TypedNode,
        outlet_id: &OutletId,
    ) -> Option<&TypedNode> {
        let succ = self.single_succ(node.id).ok()??;
        let succ_op = succ.op_as::<TypedBinOp>()?;
        (succ_op.0.is::<B>() && succ.inputs.contains(outlet_id)).then_some(succ)
    }
}
