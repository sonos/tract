use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::array::{Gather, MultiBroadcastTo, TypedConcat};
use tract_core::ops::cast::cast;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::math::{floor, mul, sub};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

/// com.microsoft GatherBlockQuantized: a `Gather` over a block-wise quantized table,
/// dequantized on the way out.
///
/// `data` is `[rows, cols / 2]` uint8 holding two 4-bit values per byte, low nibble first,
/// quantized along the last axis in blocks of `block_size`. `scales` is
/// `[rows, cols / block_size]` and the optional `zero_points` packs one 4-bit zero per
/// block the same way; absent, the zero is 8. Output is `(value - zero) * scale`.
///
/// The rows are gathered *before* being dequantized, so the table stays 4-bit in memory
/// and only the selected rows are widened — dequantizing the whole table up front would
/// cost eight times its size for a vocabulary-sized embedding.
///
/// Scoped to what the ORT-GenAI exports emit: a rank-2 table, `bits=4`, uint8 storage,
/// `gather_axis` 0 (which the operator requires for uint8) and `quantize_axis` the last.
pub fn gather_block_quantized(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let bits: usize = node.get_attr_opt("bits")?.unwrap_or(4);
    ensure!(bits == 4, "GatherBlockQuantized: only bits=4 is supported (got {bits})");
    let block_size: usize = node.get_attr_opt("block_size")?.unwrap_or(128);
    ensure!(
        block_size >= 16 && block_size.is_power_of_two(),
        "GatherBlockQuantized: block_size must be a power of two of at least 16, got {block_size}"
    );
    let gather_axis: i64 = node.get_attr_opt("gather_axis")?.unwrap_or(0);
    ensure!(gather_axis == 0, "GatherBlockQuantized: only gather_axis=0 is supported");
    let quantize_axis: i64 = node.get_attr_opt("quantize_axis")?.unwrap_or(1);
    ensure!(
        quantize_axis == 1 || quantize_axis == -1,
        "GatherBlockQuantized: only the last axis can be the quantized one, got {quantize_axis}"
    );
    let has_zero_points = node.input.len() > 3 && !node.input[3].is_empty();
    Ok((expand(GatherBlockQuantized { block_size, has_zero_points }), vec![]))
}

#[derive(Debug, Clone)]
struct GatherBlockQuantized {
    block_size: usize,
    has_zero_points: bool,
}

/// Splits every byte of `x` into its low then high nibble along the last axis, doubling it.
/// `x` is float-typed and integral in `0..256`, where `/16` and the remainder are exact.
fn unpack_nibbles(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    packed: &TDim,
) -> TractResult<OutletId> {
    let rank = model.outlet_fact(x)?.rank();
    let sixteenth = model.add_const(format!("{prefix}.sixteenth"), tensor0(1f32 / 16.))?;
    let sixteen = model.add_const(format!("{prefix}.sixteen"), tensor0(16f32))?;
    let scaled =
        wire_with_rank_broadcast(format!("{prefix}.scaled"), model, mul(), &[x, sixteenth])?[0];
    let high = model.wire_node(format!("{prefix}.high"), floor(), &[scaled])?[0];
    let high_16 =
        wire_with_rank_broadcast(format!("{prefix}.high_16"), model, mul(), &[high, sixteen])?[0];
    let low = wire_with_rank_broadcast(format!("{prefix}.low"), model, sub(), &[x, high_16])?[0];

    let low = model.wire_node(format!("{prefix}.low_axis"), AxisOp::Add(rank), &[low])?[0];
    let high = model.wire_node(format!("{prefix}.high_axis"), AxisOp::Add(rank), &[high])?[0];
    let pairs =
        model.wire_node(format!("{prefix}.pairs"), TypedConcat::new(rank), &[low, high])?[0];
    Ok(model.wire_node(
        format!("{prefix}.merge"),
        AxisOp::Reshape(rank - 1, tvec![packed.clone(), 2.to_dim()], tvec![packed.clone() * 2]),
        &[pairs],
    )?[0])
}

/// Repeats each of the `blocks` per-block values `block_size` times along the last axis.
fn expand_blocks(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    blocks: &TDim,
    block_size: usize,
) -> TractResult<OutletId> {
    let rank = model.outlet_fact(x)?.rank();
    let per_block = model.wire_node(format!("{prefix}.axis"), AxisOp::Add(rank), &[x])?[0];
    let mut shape: TVec<TDim> = model.outlet_fact(per_block)?.shape.to_tvec();
    shape[rank] = block_size.to_dim();
    let broadcast = model.wire_node(
        format!("{prefix}.broadcast"),
        MultiBroadcastTo::new(ShapeFact::from_dims(shape)),
        &[per_block],
    )?[0];
    Ok(model.wire_node(
        format!("{prefix}.merge"),
        AxisOp::Reshape(
            rank - 1,
            tvec![blocks.clone(), block_size.to_dim()],
            tvec![blocks.clone() * block_size],
        ),
        &[broadcast],
    )?[0])
}

impl Expansion for GatherBlockQuantized {
    fn name(&self) -> StaticName {
        "GatherBlockQuantized".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, if self.has_zero_points { 4 } else { 3 })?;
        check_output_arity(outputs, 1)?;
        // Output takes the scales' type, and the indices' shape with the table's row
        // replaced by the dequantized column count.
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, data, indices| {
            let mut shape: TVec<TDim> = indices;
            shape.push(data[1].clone() * 2);
            s.equals(&outputs[0].shape, ShapeFactoid::from(shape))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let data_fact = model.outlet_fact(inputs[0])?.clone();
        ensure!(
            data_fact.rank() == 2,
            "GatherBlockQuantized: expected a rank 2 table, got rank {}",
            data_fact.rank()
        );
        ensure!(
            data_fact.datum_type == u8::datum_type(),
            "GatherBlockQuantized: only uint8 storage is supported, got {:?}",
            data_fact.datum_type
        );
        let packed = data_fact.shape[1].clone();
        let cols = packed.clone() * 2;
        let blocks = model.outlet_fact(inputs[2])?.shape[1].clone();
        ensure!(
            blocks.clone() * self.block_size == cols,
            "GatherBlockQuantized: {blocks} scales of block {} do not cover {cols} columns",
            self.block_size
        );
        let dt = model.outlet_fact(inputs[2])?.datum_type;

        let rows =
            model.wire_node(format!("{prefix}.rows"), Gather::new(0), &[inputs[0], inputs[1]])?[0];
        let rows =
            model.wire_node(format!("{prefix}.rows_f32"), cast(f32::datum_type()), &[rows])?[0];
        let values = unpack_nibbles(model, &format!("{prefix}.values"), rows, &packed)?;

        let scales =
            model.wire_node(format!("{prefix}.scales"), Gather::new(0), &[inputs[2], inputs[1]])?
                [0];
        let scales =
            model.wire_node(format!("{prefix}.scales_f32"), cast(f32::datum_type()), &[scales])?[0];
        let scales =
            expand_blocks(model, &format!("{prefix}.scales_x"), scales, &blocks, self.block_size)?;

        let zeros = if self.has_zero_points {
            let packed_zeros = model.outlet_fact(inputs[3])?.shape[1].clone();
            let z = model.wire_node(
                format!("{prefix}.zeros"),
                Gather::new(0),
                &[inputs[3], inputs[1]],
            )?[0];
            let z =
                model.wire_node(format!("{prefix}.zeros_f32"), cast(f32::datum_type()), &[z])?[0];
            unpack_nibbles(model, &format!("{prefix}.zeros_u"), z, &packed_zeros)?
        } else {
            model.add_const(format!("{prefix}.zeros"), tensor0(8f32))?
        };
        let zeros = if self.has_zero_points {
            expand_blocks(model, &format!("{prefix}.zeros_x"), zeros, &blocks, self.block_size)?
        } else {
            zeros
        };

        let centered =
            wire_with_rank_broadcast(format!("{prefix}.centered"), model, sub(), &[values, zeros])?
                [0];
        let out = wire_with_rank_broadcast(
            format!("{prefix}.dequant"),
            model,
            mul(),
            &[centered, scales],
        )?[0];
        Ok(tvec!(model.wire_node(prefix.to_string(), cast(dt), &[out])?[0]))
    }
}
