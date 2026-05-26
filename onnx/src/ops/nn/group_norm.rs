use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::ops::math::{add, mul, rsqrt, square, sub};
use tract_hir::ops::nn::{Reduce, Reducer};

pub fn group_normalization(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let num_groups: usize = node.get_attr("num_groups")?;
    // Before opset 21, `scale`/`bias` are per-group (shape [num_groups]); from opset 21 on they
    // are per-channel (shape [C]), matching the other normalization operators.
    let per_channel_affine = ctx.onnx_operator_set_version >= 21;
    Ok((expand(GroupNorm { epsilon, num_groups, per_channel_affine }), vec![]))
}

#[derive(Debug, Clone, new)]
struct GroupNorm {
    epsilon: f32,
    num_groups: usize,
    per_channel_affine: bool,
}

// Broadcast a 1-D parameter [K] to rank `target_rank` with K on axis 1: [1, K, 1, .., 1].
fn broadcast_to_channel_axis(
    model: &mut TypedModel,
    base: &str,
    outlet: OutletId,
    target_rank: usize,
) -> TractResult<OutletId> {
    let mut wire = model.wire_node(format!("{base}.ax0"), AxisOp::Add(0), &[outlet])?;
    for ax in 2..target_rank {
        wire = model.wire_node(format!("{base}.ax{ax}"), AxisOp::Add(ax), &wire)?;
    }
    Ok(wire[0])
}

impl Expansion for GroupNorm {
    fn name(&self) -> StaticName {
        "GroupNorm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?.clone();
        let rank = fact.rank();
        let dt = fact.datum_type;
        ensure!(rank >= 2, "GroupNormalization expects rank >= 2, got {rank}");
        let c = fact.shape[1].clone();
        let groups = self.num_groups.to_dim();
        let channels_per_group = c.clone().div_ceil(self.num_groups as u64);

        // Per the ONNX spec (`stash_type`, default 1=FLOAT), mean/variance are computed in f32;
        // this matters for f16/bf16 inputs. Cast in, normalize in f32, cast back before the affine.
        let stash = DatumType::F32;
        let x = model.wire_node(format!("{prefix}.cast_in"), cast(stash), &inputs[0..1])?;

        // Split the channel axis: (N, C, *spatial) -> (N, G, C/G, *spatial), rank + 1.
        let grouped = model.wire_node(
            format!("{prefix}.split"),
            AxisOp::Reshape(1, tvec![c.clone()], tvec![groups.clone(), channels_per_group.clone()]),
            &x,
        )?;

        // Normalize each group over C/G and all spatial dims (axes 2..=rank in the split tensor).
        let red_axes: Vec<i64> = (2..=rank as i64).collect();
        let mean = Reduce::new(Some(red_axes.clone()), true, Reducer::Mean).wire(
            &format!("{prefix}.mean"),
            model,
            &grouped,
        )?;
        let diff = wire_with_rank_broadcast(
            format!("{prefix}.diff"),
            model,
            sub(),
            &[grouped[0], mean[0]],
        )?;
        let sq = model.wire_node(format!("{prefix}.sq"), square(), &diff)?;
        let var = Reduce::new(Some(red_axes), true, Reducer::Mean).wire(
            &format!("{prefix}.var"),
            model,
            &sq,
        )?;
        let eps = model.add_const(
            format!("{prefix}.eps"),
            tensor0(self.epsilon).cast_to_dt(stash)?.into_owned(),
        )?;
        let var_eps =
            wire_with_rank_broadcast(format!("{prefix}.var_eps"), model, add(), &[var[0], eps])?;
        let inv = model.wire_node(format!("{prefix}.rsqrt"), rsqrt(), &var_eps)?;
        let normed_f32 =
            wire_with_rank_broadcast(format!("{prefix}.normed"), model, mul(), &[diff[0], inv[0]])?;
        // Back to the input dtype before the (dtype-native) scale/bias affine.
        let normed = model.wire_node(format!("{prefix}.cast_out"), cast(dt), &normed_f32)?;

        let merge = |model: &mut TypedModel, name: String, wire: &[OutletId]| {
            model.wire_node(
                name,
                AxisOp::Reshape(
                    1,
                    tvec![groups.clone(), channels_per_group.clone()],
                    tvec![c.clone()],
                ),
                wire,
            )
        };

        if self.per_channel_affine {
            // scale/bias are [C]: merge back to (N, C, *spatial), then apply per-channel affine.
            let merged = merge(model, format!("{prefix}.merge"), &normed)?;
            let scale =
                broadcast_to_channel_axis(model, &format!("{prefix}.scale"), inputs[1], rank)?;
            let scaled = wire_with_rank_broadcast(
                format!("{prefix}.scaled"),
                model,
                mul(),
                &[merged[0], scale],
            )?;
            let bias =
                broadcast_to_channel_axis(model, &format!("{prefix}.bias"), inputs[2], rank)?;
            wire_with_rank_broadcast(prefix, model, add(), &[scaled[0], bias])
        } else {
            // scale/bias are [num_groups]: apply on the grouped tensor, then merge back.
            let gr_rank = rank + 1;
            let scale =
                broadcast_to_channel_axis(model, &format!("{prefix}.scale"), inputs[1], gr_rank)?;
            let scaled = wire_with_rank_broadcast(
                format!("{prefix}.scaled"),
                model,
                mul(),
                &[normed[0], scale],
            )?;
            let bias =
                broadcast_to_channel_axis(model, &format!("{prefix}.bias"), inputs[2], gr_rank)?;
            let biased = wire_with_rank_broadcast(
                format!("{prefix}.biased"),
                model,
                add(),
                &[scaled[0], bias],
            )?;
            merge(model, prefix.to_string(), &biased)
        }
    }
}
