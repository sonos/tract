use std::fmt::Formatter;
use std::ops::Deref;

use tract_itertools::{izip, multiunzip};
use tract_linalg::block_quant::PackedBlockQuantFormat;
use tract_linalg::pack::PackedFormat;

use super::*;
use crate::ops::cast::cast;
use crate::ops::math::add;
use crate::ops::matmul::ModePicker;
use crate::ops::matmul::optimized::{
    AddMatMulGeometry, MapOutputAxisToInput, OptMatMul, ProtoFusedSpec,
};
use crate::ops::matmul::pack::{OptMatMulPack, OptSimpleMatMulPack};
use crate::ops::matmul::quant::{
    combine_scales, compensate_zero_points, requant, wire_ensure_q8_flavour,
};
use crate::ops::nn::{Reduce, Reducer};

pub fn merge_consecutive_same_role_axes(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("merge-same-role-axes", merge_same_role_axes_rule)
        .rewrite(&(), model)
}

fn merge_same_role_axes_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    // Only handle 2-input EinSums (matmul-like)
    rule_if!(node.inputs.len() == 2);

    // Compute role signature for each axis: (in_input_0, in_input_1, in_output)
    type Role = (bool, bool, bool);
    let axes: Vec<(char, Role)> = op
        .axes
        .iter_all_axes()
        .map(|a| {
            (a.repr, (!a.inputs[0].is_empty(), !a.inputs[1].is_empty(), !a.outputs[0].is_empty()))
        })
        .collect();

    // For each input/output slot, get the axis order
    let a_order: Vec<char> = op.axes.axes(InOut::In(0)).map(|a| a.repr).collect();
    let b_order: Vec<char> = op.axes.axes(InOut::In(1)).map(|a| a.repr).collect();
    let c_order: Vec<char> = op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();

    // Find first group of 2+ same-role axes that are consecutive in all inputs.
    // Scan each input's axis order for runs of same-role axes.
    let role_map: std::collections::HashMap<char, Role> = axes.iter().cloned().collect();
    let mut best_group: Option<Vec<char>> = None;

    // Try each input order as the primary scan order
    let all_orders = [&a_order, &b_order];
    for (primary_idx, primary_order) in all_orders.iter().enumerate() {
        let mut i = 0;
        while i < primary_order.len() {
            let first = primary_order[i];
            let first_role = role_map[&first];
            let mut group = vec![first];
            let mut j = i + 1;
            while j < primary_order.len() {
                let candidate = primary_order[j];
                if role_map[&candidate] != first_role {
                    break;
                }
                // Check consecutive in the OTHER input too
                let consecutive_in_others = all_orders
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != primary_idx)
                    .all(|(_, order)| {
                        let positions: Vec<usize> = group
                            .iter()
                            .chain(std::iter::once(&candidate))
                            .filter_map(|c| order.iter().position(|x| x == c))
                            .collect();
                        if positions.len() <= 1 {
                            return true;
                        }
                        let mut sorted = positions.clone();
                        sorted.sort();
                        sorted == positions
                            && sorted.last().unwrap() - sorted.first().unwrap() == sorted.len() - 1
                    });
                if !consecutive_in_others {
                    break;
                }
                group.push(candidate);
                j += 1;
            }
            if group.len() >= 2 && best_group.as_ref().map_or(true, |bg| group.len() > bg.len()) {
                best_group = Some(group);
            }
            i = j;
        }
    }

    if let Some(group) = best_group {
        // Found a mergeable group. Emit the patch.
        let input_facts = model.node_input_facts(node.id)?;
        let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
        let output_shape = super::eval::output_shape(&op.axes, &input_shapes)?;

        let drop_set: Vec<char> = group[1..].to_vec();

        let mut patch = TypedModelPatch::default();
        let mut wires: TVec<OutletId> = patch.taps(model, &node.inputs)?;

        // Reshape each input to merge the group
        for (slot, order) in [(0, &a_order), (1, &b_order)] {
            let positions: Vec<usize> =
                group.iter().filter_map(|c| order.iter().position(|x| x == c)).collect();
            if positions.len() < 2 {
                continue;
            }
            let start = positions[0];
            let from_dims: TVec<TDim> =
                positions.iter().map(|&p| input_shapes[slot][p].clone()).collect();
            let merged: TDim = from_dims.iter().product();
            wires[slot] = patch.wire_node(
                format!("{node_name}.merge_in{slot}"),
                AxisOp::Reshape(start, from_dims, tvec![merged]),
                &[wires[slot]],
            )?[0];
        }

        // If group axes aren't consecutive in C, reorder the EinSum output
        let c_positions: Vec<usize> =
            group.iter().filter_map(|c| c_order.iter().position(|x| x == c)).collect();
        let c_needs_reorder = c_positions.len() >= 2 && {
            let mut sorted = c_positions.clone();
            sorted.sort();
            sorted.last().unwrap() - sorted.first().unwrap() != sorted.len() - 1
                || sorted != c_positions
        };
        let mut adjusted_c_order = c_order.clone();
        if c_needs_reorder {
            // Move group axes together (put second next to first)
            for k in 1..c_positions.len() {
                let cur_pos = adjusted_c_order.iter().position(|&c| c == group[k]).unwrap();
                let target_pos =
                    adjusted_c_order.iter().position(|&c| c == group[k - 1]).unwrap() + 1;
                if cur_pos != target_pos {
                    let removed = adjusted_c_order.remove(cur_pos);
                    let insert_at = if cur_pos < target_pos { target_pos - 1 } else { target_pos };
                    adjusted_c_order.insert(insert_at, removed);
                }
            }
        }

        // Rebuild EinSum formula with adjusted output and dropped axes
        let in0: String = a_order.iter().collect();
        let in1: String = b_order.iter().collect();
        let out: String = adjusted_c_order.iter().collect();
        let expr = format!("{in0},{in1}->{out}");
        let mut new_axes: AxesMapping = expr.parse()?;
        for &drop in &drop_set {
            new_axes = new_axes.remove_axis(drop)?;
        }
        let new_op =
            EinSum { axes: new_axes, operating_dt: op.operating_dt, q_params: op.q_params };
        let mut result = patch.wire_node(node_name, new_op, &wires)?;

        // Reshape output to split the merged axis back
        let merged_c_positions: Vec<usize> =
            group.iter().filter_map(|c| adjusted_c_order.iter().position(|x| x == c)).collect();
        if merged_c_positions.len() >= 2 {
            let start = merged_c_positions[0];
            // Use original output dims for the group axes
            let original_c_positions: Vec<usize> =
                group.iter().filter_map(|c| c_order.iter().position(|x| x == c)).collect();
            let original_dims: TVec<TDim> =
                original_c_positions.iter().map(|&p| output_shape[p].clone()).collect();
            let merged: TDim = original_dims.iter().product();
            result[0] = patch.wire_node(
                format!("{node_name}.unmerge_out"),
                AxisOp::Reshape(start, tvec![merged], original_dims),
                &[result[0]],
            )?[0];
        }

        // Restore original output order if we reordered
        if c_needs_reorder {
            // After unmerge, axes are in adjusted_c_order (but with group expanded).
            // Need to permute back to c_order.
            // Build the unmerged adjusted order
            let mut unmerged_adj: Vec<char> = Vec::new();
            for &c in &adjusted_c_order {
                if c == group[0] {
                    unmerged_adj.extend(&group);
                } else if !group.contains(&c) {
                    unmerged_adj.push(c);
                }
            }
            // Find what moves are needed to get from unmerged_adj to c_order
            for target_pos in 0..c_order.len() {
                let cur_pos = unmerged_adj.iter().position(|&c| c == c_order[target_pos]).unwrap();
                if cur_pos != target_pos {
                    result[0] = patch.wire_node(
                        format!("{node_name}.restore_out_{target_pos}"),
                        AxisOp::Move(cur_pos, target_pos),
                        &[result[0]],
                    )?[0];
                    let removed = unmerged_adj.remove(cur_pos);
                    unmerged_adj.insert(target_pos, removed);
                }
            }
        }

        patch.shunt_outside(model, node.id.into(), result[0])?;
        return Ok(Some(patch));
    }

    // Second pass: look for same-role pairs separated by exactly one k-like axis
    // in a single input. Insert a MoveAxis to push the separator to the end.
    let k_role: Role = (true, true, false); // present in both inputs, absent from output
    let role_of = |c: char| axes.iter().find(|(ch, _)| *ch == c).map(|(_, r)| *r);

    for (slot, order) in [(0usize, &a_order), (1, &b_order)] {
        // Find three consecutive axes in this input where the outer two share a role
        // and the middle one is a k-axis
        for w in order.windows(3) {
            let (left, mid, right) = (w[0], w[1], w[2]);
            let left_role = role_of(left);
            let mid_role = role_of(mid);
            let right_role = role_of(right);
            if left_role != right_role || mid_role != Some(k_role) {
                continue;
            }
            // left and right must also be consecutive in other inputs
            // (output order is handled by the EinSum formula)
            let other_input_orders: Vec<&Vec<char>> = [(0, &a_order), (1, &b_order)]
                .iter()
                .filter(|(s, _)| *s != slot)
                .map(|(_, o)| *o)
                .collect();
            let consecutive_elsewhere = other_input_orders.iter().all(|order| {
                let lp = order.iter().position(|&c| c == left);
                let rp = order.iter().position(|&c| c == right);
                match (lp, rp) {
                    (Some(l), Some(r)) => r == l + 1,
                    _ => true, // one or both absent — no constraint
                }
            });
            if !consecutive_elsewhere {
                continue;
            }

            // Move the k-axis to the inner (last) position in inputs and
            // make left,right adjacent in the output too.
            let mid_pos = order.iter().position(|&c| c == mid).unwrap();
            let end_pos = order.len() - 1;
            if mid_pos == end_pos {
                continue;
            }

            // Use change_axes to update the EinSum formula for the input move
            let move_op = AxisOp::Move(mid_pos, end_pos);
            let Some(AxisChangeConsequence { substitute_op, .. }) =
                op.change_axes(model, node, InOut::In(slot), &move_op)?
            else {
                continue;
            };
            let mut current_op = *substitute_op
                .unwrap()
                .downcast::<EinSum>()
                .map_err(|_| anyhow!("expected EinSum"))?;

            // Also make left,right adjacent in the output if needed
            let new_c: Vec<char> = current_op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();
            let left_c = new_c.iter().position(|&c| c == left);
            let right_c = new_c.iter().position(|&c| c == right);
            let need_output_fix = matches!((left_c, right_c), (Some(l), Some(r)) if r != l + 1);
            if need_output_fix {
                let r_pos = right_c.unwrap();
                let l_pos = left_c.unwrap();
                let target = if r_pos < l_pos { l_pos } else { l_pos + 1 };
                if let Some(AxisChangeConsequence { substitute_op, .. }) = current_op.change_axes(
                    model,
                    node,
                    InOut::Out(0),
                    &AxisOp::Move(r_pos, target),
                )? {
                    current_op = *substitute_op
                        .unwrap()
                        .downcast::<EinSum>()
                        .map_err(|_| anyhow!("expected EinSum"))?;
                }
            }

            let mut patch = TypedModelPatch::default();
            let mut wires: TVec<OutletId> = patch.taps(model, &node.inputs)?;

            wires[slot] =
                patch.wire_node(format!("{node_name}.move_k_in{slot}"), move_op, &[wires[slot]])?
                    [0];

            let final_c: Vec<char> = current_op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();
            let mut result = patch.wire_node(node_name, current_op, &wires)?;

            // Restore original output order
            if need_output_fix {
                let r_cur = final_c.iter().position(|&c| c == right).unwrap();
                let r_orig = c_order.iter().position(|&c| c == right).unwrap();
                if r_cur != r_orig {
                    result[0] = patch.wire_node(
                        format!("{node_name}.restore_out"),
                        AxisOp::Move(r_cur, r_orig),
                        &[result[0]],
                    )?[0];
                }
            }

            patch.shunt_outside(model, node.id.into(), result[0])?;
            return Ok(Some(patch));
        }
    }

    Ok(None)
}

pub fn detect_all(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default().with_rule_for("detect-matmul-einsum", detect_rule).rewrite(&(), model)
}

pub fn flatten_all(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default().with_rule_for("flatten-matmul-einsum", flatten_rule).rewrite(&(), model)
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct EinSumMatMul {
    pub op: EinSum,
    pub m_axis: char,
    pub k_axis: char,
    pub n_axis: char,
    pub m: TDim,
    pub k: TDim,
    pub n: TDim,
}

impl EinSumMatMul {
    pub fn m_axis(&self) -> &Axis {
        self.op.axes.axis(self.m_axis).unwrap()
    }
    pub fn k_axis(&self) -> &Axis {
        self.op.axes.axis(self.k_axis).unwrap()
    }
    pub fn n_axis(&self) -> &Axis {
        self.op.axes.axis(self.n_axis).unwrap()
    }
    pub fn a_m(&self) -> usize {
        self.m_axis().inputs[0][0]
    }
    pub fn a_k(&self) -> usize {
        self.k_axis().inputs[0][0]
    }
    pub fn b_k(&self) -> usize {
        self.k_axis().inputs[1][0]
    }
    pub fn b_n(&self) -> usize {
        self.n_axis().inputs[1][0]
    }
    pub fn c_m(&self) -> Option<usize> {
        self.m_axis().outputs[0].first().cloned()
    }
    pub fn c_n(&self) -> Option<usize> {
        self.n_axis().outputs[0].first().cloned()
    }

    fn new(
        op: EinSum,
        m_axis: char,
        k_axis: char,
        n_axis: char,
        m: TDim,
        k: TDim,
        n: TDim,
    ) -> Self {
        Self { op, m_axis, k_axis, n_axis, m, k, n }
    }
}

impl Debug for EinSumMatMul {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "EinsumMatMul: {} {:?} m: {}={}; k: {}={}; n: {}={}",
            self.op.axes,
            self.op.operating_dt,
            self.m_axis,
            self.m,
            self.k_axis,
            self.k,
            self.n_axis,
            self.n
        )
    }
}

impl Deref for EinSumMatMul {
    type Target = EinSum;
    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl Op for EinSumMatMul {
    fn name(&self) -> StaticName {
        "EinSumMatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for EinSumMatMul {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        self.op.eval_with_session(node_id, session, inputs)
    }
}

impl TypedOp for EinSumMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.op.output_facts(inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        // deal with parametric quantization extra inputs
        if node.inputs.len() == 9 {
            ensure!(self.op.q_params.is_some());
            return dequant(model, node, self).map(Some);
        }
        ensure!(node.inputs.len() == 2);
        let (a, b) = model.node_input_facts(node.id)?.into_iter().collect_tuple().unwrap();
        // at this stage a and b must NOT be packed yet. if they are exotic, we can assume it's just compression
        let must_transpose = if let Some(of) = a.exotic_fact() {
            ensure!(of.is::<BlockQuantFact>());
            false
        } else if let Some(of) = b.exotic_fact() {
            ensure!(of.is::<BlockQuantFact>());
            true
        } else if self.m == self.n {
            false
        } else {
            match (self.m.as_i64(), self.n.as_i64()) {
                (Some(m), Some(n)) => m < n,
                (None, Some(n)) => n >= 8,
                (Some(_), _) => false,
                _ => (self.n.clone() - &self.m).prove_positive_or_zero(),
            }
        };
        if must_transpose {
            let mut op = self.clone();
            op.op.axes.iter_all_axes_mut().for_each(|axis| axis.inputs.swap(0, 1));
            std::mem::swap(&mut op.m_axis, &mut op.n_axis);
            std::mem::swap(&mut op.m, &mut op.n);
            return TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[1], node.inputs[0]],
                op,
            )
            .map(|p| Some(p.with_context("transposing")));
        }
        // opt mat mul assumes we have at least one m or n
        if self.c_m().is_some() || self.c_n().is_some() {
            return optimized_mat_mul(model, node, self)
                .map(|opt| opt.map(|p| p.with_context("optimizing")));
        }
        Ok(None)
    }

    as_op!();
}

pub(crate) fn detect_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.inputs.len() == (2 + op.q_params.is_some() as usize * 7));
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let output_shape = super::eval::output_shape(&op.axes, &input_shapes)?;
    let k_axes: TVec<&Axis> = op
        .axes
        .iter_all_axes()
        // Filter possible candidates (should be one time in each inputs but not in output)
        .filter(|a| a.inputs[0].len() == 1 && a.inputs[1].len() == 1 && a.outputs[0].is_empty())
        .collect();

    let non_trivial_k_axis = k_axes
        .iter()
        .filter(|a| {
            !input_shapes[0][a.inputs[0][0]].is_one() || !input_shapes[1][a.inputs[1][0]].is_one()
        })
        .copied()
        .collect::<TVec<_>>();

    let k_axis = if non_trivial_k_axis.len() > 1 {
        return regroup_k_axes(op, model, node, non_trivial_k_axis);
    } else {
        non_trivial_k_axis.first().or_else(|| k_axes.first()).copied()
    };
    let Some(k_axis) = k_axis else { return inject_k_axis(op, model, node).map(Some) };

    let mut possible_m_axes: Vec<_> = op
        .axes
        .iter_all_axes()
        .filter(|a| {
            a.inputs[0].len() == 1
                && (a.inputs[1].is_empty() || input_shapes[1][a.inputs[1][0]].is_one())
                && (a.outputs[0].len() == 1
                    || (input_shapes[0][a.inputs[0][0]].is_one() && a.inputs[1].is_empty()))
        })
        .collect();

    // Prioritize obvious m-axes
    if possible_m_axes.iter().any(|a| !a.outputs[0].is_empty()) {
        possible_m_axes.retain(|a| !a.outputs[0].is_empty());
    }

    let m_axis = possible_m_axes
        .into_iter()
        .max_by_key(|a| input_shapes[0][a.inputs[0][0]].as_i64().unwrap_or(i64::MAX));

    let Some(m_axis) = m_axis else {
        return inject_m_or_n_axis(op, model, node, false).map(Some);
    };

    let n_axis = op
        .axes
        .iter_all_axes()
        .filter(|a| {
            (a.inputs[0].is_empty() || input_shapes[0][a.inputs[0][0]].is_one())
                && a.inputs[1].len() == 1
                && a.outputs[0].len() == 1
                && *a != m_axis
        })
        .max_by_key(|a| input_shapes[1][a.inputs[1][0]].as_i64().unwrap_or(i64::MAX));
    let Some(n_axis) = n_axis else {
        return inject_m_or_n_axis(op, model, node, true).map(Some);
    };
    for axis in op.axes.iter_all_axes() {
        let one = TDim::one();
        let in_left =
            axis.inputs[0].first().map(|pos| &input_shapes[0][*pos]).unwrap_or(&one) != &one;
        let in_right =
            axis.inputs[1].first().map(|pos| &input_shapes[1][*pos]).unwrap_or(&one) != &one;
        let in_out = axis.outputs[0].first().map(|pos| &output_shape[*pos]).unwrap_or(&one) != &one;
        if (in_left ^ in_right) && !in_out {
            return Ok(None);
            // return Ok(AxesOrPatch::NotAMatMul(
            //     "non trivial single-side disappearing axis",
            //     vec![axis],
            // ));
        }
    }
    let m = input_shapes[0][m_axis.inputs[0][0]].clone();
    let k = input_shapes[0][k_axis.inputs[0][0]].clone();
    let n = input_shapes[1][n_axis.inputs[1][0]].clone();
    TypedModelPatch::replace_single_op(
        model,
        node,
        &node.inputs,
        EinSumMatMul::new(op.clone(), m_axis.repr, k_axis.repr, n_axis.repr, m, k, n),
    )
    .map(Some)
}

pub(super) fn inject_k_axis(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<TypedModelPatch> {
    let mut new_axes = op.axes.clone();
    let name = &node.name;
    let mut patch = TypedModelPatch::new("inject k axis");
    let mut wire = patch.taps(model, &node.inputs)?;
    let repr = new_axes.available_label();
    new_axes = new_axes.with_extra_axis(repr, InOut::In(0), 0)?.with_extra_axis_occurency(
        repr,
        InOut::In(1),
        0,
    )?;
    wire[0] = patch.wire_node(format!("{name}.add_k.0"), AxisOp::Add(0), &[wire[0]])?[0];
    wire[1] = patch.wire_node(format!("{name}.add_k.1"), AxisOp::Add(0), &[wire[1]])?[0];
    wire = patch.wire_node(&node.name, EinSum { axes: new_axes, ..op.clone() }, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(patch)
}

pub(super) fn regroup_k_axes(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    mut k_axes: TVec<&Axis>,
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let contig_in_a = k_axes
        .iter()
        .map(|axis| axis.inputs[0][0])
        .sorted()
        .tuple_windows()
        .all(|(a, b)| a + 1 == b);
    if contig_in_a {
        k_axes.sort_by_key(|ax| ax.inputs[0][0]);
    } else {
        k_axes.sort_by_key(|ax| ax.inputs[1][0]);
    }
    let k_dims: TVec<_> =
        k_axes.iter().map(|ax| input_shapes[0][ax.inputs[0][0]].clone()).collect();
    let k: TDim = k_dims.iter().product();
    let mut patch = TypedModelPatch::default();
    let mut wires = patch.taps(model, &node.inputs)?;
    let mut exprs: Vec<String> =
        (0..2).map(|slot| op.axes.axes(InOut::In(slot)).map(|ax| ax.repr).join("")).collect();
    for slot in 0..2 {
        if k_axes.iter().map(|ax| ax.inputs[slot][0]).tuple_windows().any(|(a, b)| a + 1 != b) {
            let after = op
                .axes
                .axes(InOut::In(slot))
                .filter(|ax| !k_axes.contains(ax))
                .chain(k_axes.iter().copied())
                .map(|ax| ax.repr)
                .join("");
            let transpose =
                AxesMapping::from_strs(&[&exprs[slot]], &[&after])?.translate_to_axis_ops()?;
            for (ix, op) in transpose.into_iter().enumerate() {
                wires[slot] = patch.wire_node(
                    format!("{}.transpose_input_{}.{}", &node.name, slot, ix),
                    op,
                    &[wires[slot]],
                )?[0];
            }
            exprs[slot] = after;
        }
        let pos = exprs[slot].chars().position(|c| k_axes[0].repr == c).unwrap();
        wires[slot] = patch.wire_node(
            format!("{}.fold_k_in_input_{}", &node.name, slot),
            AxisOp::Reshape(pos, k_dims.clone(), tvec!(k.clone())),
            &[wires[slot]],
        )?[0];
        exprs[slot] =
            exprs[slot].chars().filter(|c| !k_axes.iter().any(|k| k.repr == *c)).collect();
        exprs[slot].insert(pos, k_axes[0].repr);
    }
    let old = op.axes.to_string();
    let (iexpr, oexpr) = old.split_once("->").unwrap();
    let mut expr: String = exprs.iter().join(",");
    if node.inputs.len() > 2 {
        expr = expr + "," + &iexpr.split(",").skip(2).join(",");
    }
    expr = expr + "->" + oexpr;
    let wire = patch.wire_node(
        &node.name,
        EinSum { axes: expr.parse().unwrap(), ..op.clone() },
        &wires,
    )?[0];
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

pub(super) fn inject_m_or_n_axis(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    is_n: bool,
) -> TractResult<TypedModelPatch> {
    let input_to_fix = is_n as usize;
    let label = if is_n { "n" } else { "m" };
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Injecting m or n axis");
    let mut wire = patch.taps(model, &node.inputs)?;
    let repr = op.axes.available_label();
    let new_axes = op
        .axes
        .clone()
        .with_extra_axis(repr, InOut::In(input_to_fix), 0)?
        .with_extra_axis_occurency(repr, InOut::Out(0), 0)?;
    wire[input_to_fix] =
        patch.wire_node(format!("{name}.add_{label}"), AxisOp::Add(0), &[wire[input_to_fix]])?[0];
    wire = patch.wire_node(name, EinSum { axes: new_axes, ..op.clone() }, &wire)?;
    wire = patch.wire_node(&node.name, AxisOp::Rm(0), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(patch)
}

fn wire_axes_fix(
    patch: &mut TypedModelPatch,
    name: &str,
    var: &str,
    mapping: &AxesMapping,
    mut outlet: TVec<OutletId>,
) -> TractResult<TVec<OutletId>> {
    for (ix, axis_op) in mapping.translate_to_axis_ops()?.into_iter().enumerate() {
        outlet = patch.wire_node(format!("{name}.fix_{var}.{ix})"), axis_op, &outlet)?;
    }
    Ok(outlet)
}

fn dequant(
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSumMatMul,
) -> TractResult<TypedModelPatch> {
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Dequantizing einsum");

    let k_axis = op.k_axis();

    let mut taps = patch.taps(model, &node.inputs)?;
    for ab in [0, 1] {
        let scale_input = 4 + ab * 2;
        if !patch.outlet_fact(taps[scale_input])?.shape.volume().is_one() {
            let q_axis_in_output = op.axes.axis((InOut::In(scale_input), 0))?.outputs[0][0];
            let output_rank = node.outputs[0].fact.rank();
            for i in 1..(output_rank - q_axis_in_output) {
                taps[scale_input] = patch.wire_node(
                    format!("{name}.scale_input{ab}_axis_fix_{i}"),
                    AxisOp::Add(i),
                    &[taps[scale_input]],
                )?[0];
            }
        }
    }

    let [mut a, mut b, bias, mut a0, a_scale, mut b0, b_scale, c0, c_scale] = *taps else {
        bail!("Expect exactly 9 inputs")
    };

    wire_ensure_q8_flavour(&mut patch, &node.name, &mut a, "a", &mut a0, i8::datum_type())?;
    wire_ensure_q8_flavour(&mut patch, &node.name, &mut b, "b", &mut b0, i8::datum_type())?;

    let mut output = patch.wire_node(
        &node.name,
        EinSum {
            q_params: None,
            axes: op.axes.extract_sub_mapping(&[0, 1], &[0])?,
            operating_dt: op.operating_dt,
        },
        &[a, b],
    )?;

    let a_i32 = patch.wire_node(format!("{name}.a_as_i32"), cast(i32::datum_type()), &[a])?[0];
    let b_i32 = patch.wire_node(format!("{name}.b_as_i32"), cast(i32::datum_type()), &[b])?[0];
    let sum_a = patch.wire_node(
        format!("{name}.sum_a"),
        Reduce::new(tvec!(k_axis.inputs[0][0]), Reducer::Sum),
        &[a_i32],
    )?;
    let sum_b = patch.wire_node(
        format!("{name}.sum_b"),
        Reduce::new(tvec!(k_axis.inputs[1][0]), Reducer::Sum),
        &[b_i32],
    )?;

    let sum_a =
        wire_axes_fix(&mut patch, name, "sum_a", &op.axes.extract_sub_mapping(&[0], &[0])?, sum_a)?;
    let sum_b =
        wire_axes_fix(&mut patch, name, "sum_b", &op.axes.extract_sub_mapping(&[1], &[0])?, sum_b)?;
    let bias = tvec!(bias);
    let bias =
        wire_axes_fix(&mut patch, name, "bias", &op.axes.extract_sub_mapping(&[2], &[0])?, bias)?;

    let abc_scale = combine_scales(&mut patch, name, a_scale, b_scale, c_scale)?;

    output = patch.wire_node(format!("{name}.add_bias"), add(), &[output[0], bias[0]])?;

    let k = model.outlet_fact(node.inputs[0])?.shape[k_axis.inputs[0][0]].clone();
    let output = compensate_zero_points(&mut patch, name, output[0], k, a0, b0, sum_a[0], sum_b[0])
        .context("Zero point compensation")?;
    let output = requant(&mut patch, name, output, op.q_params.unwrap(), abc_scale, c0)?;
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(patch)
}

fn flatten_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &EinSumMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    TypedModelPatch::replace_single_op(model, node, &node.inputs, op.op.clone()).map(Some)
}

fn optimized_mat_mul(
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSumMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    let (mode_picker, left_pack, impls) = kernel_selection::strategize(model, node, op)?;
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let prefix = &node.name;

    let mut patch = TypedModelPatch::new("Einsum to OptMatMul");
    let taps = patch.taps(model, &node.inputs)?;
    let name = &node.name;

    let pack_a: Box<dyn TypedOp> = if input_facts[0].konst.is_some() {
        if let Some(pf) = left_pack.downcast_ref::<PackedFormat>() {
            Box::new(OptMatMulPack {
                packers: vec![pf.clone()],
                mode_picker: ModePicker::Single,
                k_axis: op.a_k(),
                mn_axis: op.a_m(),
            })
        } else if let Some(packed_format) =
            left_pack.downcast_ref::<PackedBlockQuantFormat>().cloned()
        {
            Box::new(OptSimpleMatMulPack {
                packed_format,
                k: input_shapes[0][op.a_k()].to_usize().unwrap(),
                m: input_shapes[0][op.a_m()].to_usize().unwrap(),
            })
        } else {
            bail!("Unexpected static input format {left_pack:?}");
        }
    } else {
        Box::new(OptMatMulPack {
            packers: impls
                .iter()
                .map(|(mmm, p, pe)| {
                    pe.as_ref()
                        .map(|pe| &pe.from)
                        .unwrap_or(&mmm.packings()[*p].0)
                        .downcast_ref::<PackedFormat>()
                        .unwrap()
                        .clone()
                })
                .collect(),
            mode_picker: mode_picker.clone(),
            k_axis: op.a_k(),
            mn_axis: op.a_m(),
        })
    };
    let pa = patch.wire_node(format!("{prefix}.pack_a"), pack_a, &[taps[0]])?[0];

    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        OptMatMulPack {
            k_axis: op.b_k(),
            mn_axis: op.b_n(),
            packers: impls
                .iter()
                .map(|(mmm, p, _)| {
                    mmm.packings()[*p].1.downcast_ref::<PackedFormat>().unwrap().clone()
                })
                .collect(),
            mode_picker: mode_picker.clone(),
        },
        &[taps[1]],
    )?[0];

    let mut c_to_a_axis_mapping = tvec!();
    let mut c_to_b_axis_mapping = tvec!();
    for axis in op
        .op
        .axes
        .iter_all_axes()
        .filter(|&axis| ![op.m_axis, op.k_axis, op.n_axis].contains(&axis.repr))
    {
        if let (&[c], &[a]) = (&*axis.outputs[0], &*axis.inputs[0])
            && input_shapes[0][a] != 1.to_dim()
        {
            let a = a - (a > op.a_m()) as usize - (a > op.a_k()) as usize;
            c_to_a_axis_mapping.push((c, a));
        }
        if let (&[c], &[b]) = (&*axis.outputs[0], &*axis.inputs[1])
            && input_shapes[1][b] != 1.to_dim()
        {
            let b = b - (b > op.b_n()) as usize - (b > op.b_k()) as usize;
            c_to_b_axis_mapping.push((c, b));
        }
    }

    let c_fact = op.output_facts(&input_facts)?.remove(0);
    let geo = AddMatMulGeometry {
        k: op.k.clone(),
        c_to_a_axis_mapping: MapOutputAxisToInput(c_to_a_axis_mapping),
        c_to_b_axis_mapping: MapOutputAxisToInput(c_to_b_axis_mapping),
    };
    let (mmms, packings, extractor): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(impls);
    let outputs = mmms.iter().map(|mmm| unsafe { mmm.c_view(op.c_m(), op.c_n()) }).collect();
    let trivial_packing = mmms.len() == 1
        && packings[0] == 0
        && extractor[0].is_none()
        && input_facts[0].exotic_fact.is_none();
    let opt = OptMatMul::new(
        mmms,
        mode_picker,
        c_fact,
        op.c_m(),
        op.c_n(),
        vec![
            ProtoFusedSpec::AddMatMul {
                geo,
                a: 0,
                b: 1,
                packings: izip!(packings, extractor).collect_vec(),
            },
            ProtoFusedSpec::Store(outputs),
        ],
        trivial_packing,
    )
    .context("Creating OptMatMul")?;
    let output = patch.wire_node(name, opt, &[pa, pb])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
