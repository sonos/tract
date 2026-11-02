use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::par_half_cast;
use crate::ops::math::Sub;
use crate::ops::nn::{Reduce, Reducer, RmsNorm};
use tract_itertools::Itertools;

/// Group normalization, without the affine part.
///
/// The `axis` dimension (channels) is split into `groups` contiguous groups. Each group is
/// normalized to zero mean and unit variance over its own channels and every axis in
/// `reduced_axes`, independently for each index of the remaining axes:
/// `y = (x - mean) / sqrt(var + eps)`. Statistics are computed in F32 and the output keeps the
/// input datum type.
///
/// Detected from the `Reshape(C -> [groups, rest]) -> Reduce<Mean> -> Sub -> RmsNorm` chain
/// that ONNX exporters emit (see `detect_group_norm`). `change_axes` follows the channel and
/// reduced axes, so a layout change (channel-last) can be propagated through it.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GroupNorm {
    pub axis: usize,
    pub groups: usize,
    pub reduced_axes: TVec<usize>,
    pub eps: Arc<Tensor>,
}

impl Op for GroupNorm {
    fn name(&self) -> StaticName {
        "GroupNorm".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis: {}, groups: {}, reduced_axes: {:?}, eps: {:?}",
            self.axis, self.groups, self.reduced_axes, self.eps
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for GroupNorm {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let in_dt = input.datum_type();
        let rank = input.rank();
        let kept = (0..rank).filter(|ax| *ax != self.axis && !self.reduced_axes.contains(ax));
        let order: TVec<usize> = kept
            .chain(std::iter::once(self.axis))
            .chain(self.reduced_axes.iter().copied())
            .collect();
        let shape = input.shape().to_vec();
        let x = if in_dt == f32::datum_type() {
            input.into_tensor()
        } else if let Some(x) = par_half_cast(&input, f32::datum_type())? {
            x
        } else {
            input.cast_to::<f32>()?.into_owned()
        };
        let x = x.permute_axes(&order)?;
        let permuted_shape = x.shape().to_vec();
        let rows = shape.iter().enumerate().filter(|(ax, _)| !self.reduced_axes.contains(ax));
        let rows = rows.map(|(_, d)| d).product::<usize>() / shape[self.axis] * self.groups;
        let len = x.len();
        let mut x = x.into_shape(&[rows, len / rows.max(1)])?;
        let eps = self.eps.cast_to_scalar::<f32>()?;
        let row_len = x.shape()[1];
        if row_len > 0 {
            let mut view = x.try_as_plain_ram_mut()?;
            let data = view.as_slice_mut::<f32>()?;
            let total = data.len();
            tract_linalg::multithread::par_chunks_mut(data, row_len, total, |_, chunk| {
                for row in chunk.chunks_mut(row_len) {
                    let mean = row.iter().sum::<f32>() / row_len as f32;
                    let var =
                        row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row_len as f32;
                    let inv = (var + eps).sqrt().recip();
                    row.iter_mut().for_each(|v| *v = (*v - mean) * inv);
                }
                Ok(())
            })?;
        }
        let mut inverse = tvec!(0; rank);
        order.iter().enumerate().for_each(|(pos, ax)| inverse[*ax] = pos);
        let y = x.into_shape(&permuted_shape)?.permute_axes(&inverse)?;
        let y = if in_dt == f32::datum_type() {
            y
        } else if let Some(y) = par_half_cast(&y, in_dt)? {
            y
        } else {
            y.cast_to_dt(in_dt)?.into_owned()
        };
        Ok(tvec!(y.into_tvalue()))
    }
}

impl TypedOp for GroupNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let rank = inputs[0].rank();
        ensure!(self.eps.rank() == 0, "GroupNorm: eps must be a rank-0 tensor");
        ensure!(self.axis < rank && self.reduced_axes.iter().all(|ax| *ax < rank));
        ensure!(!self.reduced_axes.contains(&self.axis));
        ensure!(self.reduced_axes.iter().tuple_windows().all(|(a, b)| a < b));
        if let Ok(c) = inputs[0].shape[self.axis].to_usize() {
            ensure!(
                self.groups > 0 && c % self.groups == 0,
                "GroupNorm: {c} channels, {} groups",
                self.groups
            );
        }
        Ok(tvec!(inputs[0].without_value()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural_for_rank(1, 1, inputs[0].rank())
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        rule_if!(matches!(
            change.canonical().as_ref(),
            AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Move(..)
        ));
        rule_if_some!(axis = change.transform_axis(self.axis));
        let reduced_axes =
            self.reduced_axes.iter().filter_map(|ax| change.transform_axis(*ax)).sorted().collect();
        let op = GroupNorm { axis, reduced_axes, ..self.clone() };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = inputs[0].shape.iter().product();
        Ok(tvec!((Cost::FMA(inputs[0].datum_type), count * 4)))
    }

    as_op!();
}

/// Search pattern => GroupNorm(X), from
/// `R = Reshape(X, C -> [G, C/G * rest]); RmsNorm(R - Mean(R))` over the merged axis.
pub fn detect_group_norm(
    op: &RmsNorm,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(sub = model.single_prev_node_as::<TypedBinOp>(node).map(|(_, n)| n));
    rule_if!(sub.op_as::<TypedBinOp>().unwrap().0.is::<Sub>());
    let (centered, mean) = (sub.inputs[0], sub.inputs[1]);
    let mean_node = model.node(mean.node);
    rule_if_some!(reduce = mean_node.op_as::<Reduce>());
    rule_if!(reduce.reducer == Reducer::Mean && *reduce.axes == [op.axis]);
    rule_if!(mean_node.inputs[0] == centered);
    let reshape_node = model.node(centered.node);
    rule_if_some!(AxisOp::Reshape(at, from, to) = reshape_node.op_as::<AxisOp>());
    let x_fact = model.outlet_fact(reshape_node.inputs[0])?;
    rule_if!(*at + 1 == op.axis && to.len() == 2 && at + from.len() == x_fact.rank());
    rule_if_some!(groups = to[0].to_usize().ok());
    rule_if_some!(channels = from[0].to_usize().ok());
    rule_if!(groups > 0 && channels % groups == 0);
    let group_norm = GroupNorm {
        axis: *at,
        groups,
        reduced_axes: (at + 1..x_fact.rank()).collect(),
        eps: op.eps.clone(),
    };
    let mut patch = TypedModelPatch::default();
    let x = patch.tap_model(model, reshape_node.inputs[0])?;
    let y = patch.wire_node(format!("{}.group_norm", node.name), group_norm, &[x])?;
    let y = patch.wire_node(
        format!("{}.group_norm_reshape", node.name),
        AxisOp::Reshape(*at, from.clone(), to.clone()),
        &y,
    )?;
    patch.shunt_outside(model, node.id.into(), y[0])?;
    Ok(Some(patch))
}
