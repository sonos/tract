use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::array::{Gather, Slice, TypedConcat};
use tract_core::ops::math::{add, mul, sub};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

// ONNX RotaryEmbedding (opset 23). Mirrors onnx/reference/ops/op_rotary_embedding.py:
//   * normalize input to [batch, seq, heads, head_size]
//   * gather cos/sin caches by position_ids (when provided)
//   * rotate the first `rotary_embedding_dim` channels (NeoX halves, or GPT-J interleaved pairs)
//   * concatenate the untouched tail back and restore the original layout
pub fn rotary_embedding(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let interleaved = node.get_attr_opt::<i64>("interleaved")?.unwrap_or(0) != 0;
    let num_heads = node.get_attr_opt::<i64>("num_heads")?.unwrap_or(0) as usize;
    let rotary_embedding_dim =
        node.get_attr_opt::<i64>("rotary_embedding_dim")?.unwrap_or(0) as usize;
    Ok((expand(RotaryEmbedding { interleaved, num_heads, rotary_embedding_dim }), vec![]))
}

#[derive(Debug, Clone, new)]
struct RotaryEmbedding {
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
}

impl Expansion for RotaryEmbedding {
    fn name(&self) -> StaticName {
        "RotaryEmbedding".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        ensure!(
            inputs.len() == 3 || inputs.len() == 4,
            "RotaryEmbedding expects 3 or 4 inputs, got {}",
            inputs.len()
        );
        check_output_arity(outputs, 1)?;
        // Output keeps the input tensor's type and shape.
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
        let in_fact = model.outlet_fact(inputs[0])?.clone();
        let in_rank = in_fact.rank();
        ensure!(in_rank == 3 || in_rank == 4, "RotaryEmbedding expects rank 3 or 4, got {in_rank}");
        let has_position_ids = inputs.len() == 4;
        let two = 2usize.to_dim();

        // 1. Normalize input to [batch, seq, heads, head_size].
        let x = if in_rank == 4 {
            // [B, N, S, H] -> [B, S, N, H]
            model.wire_node(format!("{prefix}.to_bsnh"), AxisOp::Move(1, 2), &[inputs[0]])?[0]
        } else {
            // [B, S, hidden] -> [B, S, N, H]
            ensure!(self.num_heads > 0, "RotaryEmbedding with a 3D input requires num_heads");
            let hidden = in_fact.shape[2].clone();
            let head_size = hidden.clone().div_ceil(self.num_heads as u64);
            model.wire_node(
                format!("{prefix}.split_heads"),
                AxisOp::Reshape(2, tvec![hidden], tvec![self.num_heads.to_dim(), head_size]),
                &[inputs[0]],
            )?[0]
        };

        let head_size = model.outlet_fact(x)?.shape[3].clone();
        let rotary_dim = if self.rotary_embedding_dim == 0 {
            head_size.clone()
        } else {
            self.rotary_embedding_dim.to_dim()
        };
        let half = rotary_dim.clone().div_ceil(2);

        // 2. Split off the rotary part (and the pass-through tail when partial).
        let x_rotate = model.wire_node(
            format!("{prefix}.rotary"),
            Slice::new(3, 0, rotary_dim.clone()),
            &[x],
        )?[0];
        let passthrough = if rotary_dim != head_size {
            Some(
                model.wire_node(
                    format!("{prefix}.passthrough"),
                    Slice::new(3, rotary_dim.clone(), head_size.clone()),
                    &[x],
                )?[0],
            )
        } else {
            None
        };

        // 3. Prepare cos/sin as [B, S, 1, half] (gathering by position_ids if present).
        let mut prep = |tag: &str, cache: usize| -> TractResult<OutletId> {
            let gathered = if has_position_ids {
                model.wire_node(
                    format!("{prefix}.{tag}_gather"),
                    Gather::new(0),
                    &[inputs[cache], inputs[3]],
                )?[0]
            } else {
                inputs[cache]
            };
            Ok(model.wire_node(format!("{prefix}.{tag}_unsqueeze"), AxisOp::Add(2), &[gathered])?
                [0])
        };
        let cos = prep("cos", 1)?;
        let sin = prep("sin", 2)?;

        // 4. Extract the two rotated components.
        let (x1, x2) = if self.interleaved {
            // [.., rotary_dim] -> [.., half, 2], then take the even/odd lanes.
            let pairs = model.wire_node(
                format!("{prefix}.pairs"),
                AxisOp::Reshape(3, tvec![rotary_dim.clone()], tvec![half.clone(), two.clone()]),
                &[x_rotate],
            )?[0];
            let even = model.wire_node(format!("{prefix}.even"), Slice::new(4, 0, 1), &[pairs])?[0];
            let x1 = model.wire_node(format!("{prefix}.even_sq"), AxisOp::Rm(4), &[even])?[0];
            let odd = model.wire_node(format!("{prefix}.odd"), Slice::new(4, 1, 2), &[pairs])?[0];
            let x2 = model.wire_node(format!("{prefix}.odd_sq"), AxisOp::Rm(4), &[odd])?[0];
            (x1, x2)
        } else {
            let x1 = model.wire_node(
                format!("{prefix}.x1"),
                Slice::new(3, 0, half.clone()),
                &[x_rotate],
            )?[0];
            let x2 = model.wire_node(
                format!("{prefix}.x2"),
                Slice::new(3, half.clone(), rotary_dim.clone()),
                &[x_rotate],
            )?[0];
            (x1, x2)
        };

        // 5. real = cos*x1 - sin*x2 ; imag = sin*x1 + cos*x2
        let cos_x1 =
            wire_with_rank_broadcast(format!("{prefix}.cos_x1"), model, mul(), &[cos, x1])?[0];
        let sin_x2 =
            wire_with_rank_broadcast(format!("{prefix}.sin_x2"), model, mul(), &[sin, x2])?[0];
        let real =
            wire_with_rank_broadcast(format!("{prefix}.real"), model, sub(), &[cos_x1, sin_x2])?[0];
        let sin_x1 =
            wire_with_rank_broadcast(format!("{prefix}.sin_x1"), model, mul(), &[sin, x1])?[0];
        let cos_x2 =
            wire_with_rank_broadcast(format!("{prefix}.cos_x2"), model, mul(), &[cos, x2])?[0];
        let imag =
            wire_with_rank_broadcast(format!("{prefix}.imag"), model, add(), &[sin_x1, cos_x2])?[0];

        // 6. Reassemble the rotated channels.
        let rotated = if self.interleaved {
            let real5 = model.wire_node(format!("{prefix}.real_unsq"), AxisOp::Add(4), &[real])?[0];
            let imag5 = model.wire_node(format!("{prefix}.imag_unsq"), AxisOp::Add(4), &[imag])?[0];
            let interleaved = model.wire_node(
                format!("{prefix}.interleave"),
                TypedConcat::new(4),
                &[real5, imag5],
            )?[0];
            model.wire_node(
                format!("{prefix}.merge_pairs"),
                AxisOp::Reshape(3, tvec![half.clone(), two.clone()], tvec![rotary_dim.clone()]),
                &[interleaved],
            )?[0]
        } else {
            model.wire_node(
                format!("{prefix}.concat_halves"),
                TypedConcat::new(3),
                &[real, imag],
            )?[0]
        };

        // 7. Re-attach the pass-through tail.
        let out_bsnh = if let Some(pt) = passthrough {
            model.wire_node(format!("{prefix}.concat_tail"), TypedConcat::new(3), &[rotated, pt])?
                [0]
        } else {
            rotated
        };

        // 8. Restore the original layout.
        let out = if in_rank == 4 {
            // [B, S, N, H] -> [B, N, S, H]
            model.wire_node(prefix.to_string(), AxisOp::Move(2, 1), &[out_bsnh])?
        } else {
            let hidden = in_fact.shape[2].clone();
            let head_size = hidden.clone().div_ceil(self.num_heads as u64);
            model.wire_node(
                prefix.to_string(),
                AxisOp::Reshape(2, tvec![self.num_heads.to_dim(), head_size], tvec![hidden]),
                &[out_bsnh],
            )?
        };
        Ok(out)
    }
}
