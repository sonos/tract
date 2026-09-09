use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::array::{Gather, MultiBroadcastTo, TypedConcat};
use tract_core::ops::cast::cast;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::konst::Const;
use tract_core::ops::math::{add, floor, mul, sub};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};

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
/// There are two lowerings for that. `block_size` 32 over constant inputs is exactly
/// tract's `Q4_0`, so the table becomes a single block-quant constant and the gather is
/// core's `Gather`, which dequantizes the selected rows in `eval_bq`. Everything else
/// (a runtime table, another block size) keeps the table as its original uint8 tensor and
/// spells the dequantization out in the graph.
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

/// The value of an input that is a genuine initializer.
///
/// `TypedFact::konst` is not enough on its own: the ad-hoc model `Expansion::eval` builds
/// makes every input a `Source` carrying its value, so a runtime table would look constant
/// there and take the block-quant route at eval time only — the same node would then compute
/// something slightly different depending on which pass ran it.
fn initializer(model: &TypedModel, outlet: OutletId) -> Option<Arc<Tensor>> {
    if !model.node(outlet.node).op_is::<Const>() {
        return None;
    }
    model.outlet_fact(outlet).ok()?.konst.clone()
}

/// Reads the `blk`-th 4-bit zero point of a row, low nibble first.
fn zero_point_at(zeros: &[u8], row_bytes: usize, row: usize, blk: usize) -> u8 {
    let byte = zeros[row * row_bytes + blk / 2];
    if blk.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 }
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

        let out = match self.wire_block_quant(prefix, model, inputs, &blocks)? {
            Some(out) => out,
            None => self.wire_dequant_graph(prefix, model, inputs, &packed, &blocks)?,
        };
        Ok(tvec!(model.wire_node(prefix.to_string(), cast(dt), &[out])?[0]))
    }
}

impl GatherBlockQuantized {
    /// Lowers to a `Q4_0` constant read by core's `Gather`, which dequantizes only the rows
    /// the indices select. Returns `None` when the table does not fit `Q4_0` — another block
    /// size, a symbolic shape, or inputs that are not constant — leaving the graph lowering
    /// to handle it.
    ///
    /// `Q4_0` fixes the zero point at 8 and rounds the scale to f16, so an asymmetric table
    /// is split the way #2648 splits a `MatMulNBits` weight:
    /// `(q - z) * s == (q - 8) * s + (8 - z) * s`. The first term is exactly `Q4_0`, the
    /// second is constant within a block and rides along as a `[rows, blocks]` table gathered
    /// with the same indices. Both terms use the *rounded* scale and are exact in f32, so the
    /// result is exactly `(q - z) * f16(s)`: the f16 rounding of the scale is the only
    /// departure from the graph lowering, which is bit-exact. The ORT-GenAI exports quantize
    /// in f16 and store the scale widened to f32, so for those there is no departure at all.
    fn wire_block_quant(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        blocks: &TDim,
    ) -> TractResult<Option<OutletId>> {
        if self.block_size != Q4_0.block_len() {
            return Ok(None);
        }
        let Some(shape) = model.outlet_fact(inputs[0])?.shape.as_concrete().map(|s| s.to_vec())
        else {
            return Ok(None);
        };
        let (rows, cols) = (shape[0], shape[1] * 2);
        let Some(blocks) = blocks.as_i64().map(|b| b as usize) else { return Ok(None) };
        let (Some(data_k), Some(scales_k)) =
            (initializer(model, inputs[0]), initializer(model, inputs[2]))
        else {
            return Ok(None);
        };
        let zeros_k = if self.has_zero_points {
            let Some(k) = initializer(model, inputs[3]) else { return Ok(None) };
            Some(k)
        } else {
            None
        };

        let data_plain = data_k.try_as_plain()?;
        let data: &[u8] = data_plain.as_slice()?;
        let scales_f = scales_k.cast_to::<f32>()?;
        let scales_plain = scales_f.try_as_plain()?;
        let scales: &[f32] = scales_plain.as_slice()?;

        // The table is already in the nibble layout Q4_0 packs from, so it goes over as is —
        // pack_prequantized_nibbles handles the GGML interleave and the f16 scale.
        let weights = Q4_0.pack_prequantized_nibbles(data, scales, rows, cols)?;
        let bqs = BlockQuantStorage::new(Box::new(Q4_0), rows, cols, Arc::new(weights))?;
        let fact = Box::new(BlockQuantFact::new(Box::new(Q4_0), tvec!(rows, cols)));
        let table = model.wire_node(
            format!("{prefix}.table"),
            Const::new_with_exotic_fact(
                Arc::new(bqs.into_tensor_with_shape(f32::datum_type(), &[rows, cols])),
                fact,
            )?,
            &[],
        )?[0];
        let mut out = model.wire_node(
            format!("{prefix}.gather"),
            Gather { axis: 0, output_type: Some(f32::datum_type()) },
            &[table, inputs[1]],
        )?[0];

        // (8 - z) * f16(s), zero unless the export carries zero points that are not all 8.
        if let Some(zeros_k) = zeros_k {
            let zeros_plain = zeros_k.try_as_plain()?;
            let zeros: &[u8] = zeros_plain.as_slice()?;
            let row_bytes = zeros_k.shape()[1];
            let mut correction = vec![0f32; rows * blocks];
            for row in 0..rows {
                for blk in 0..blocks {
                    let z = zero_point_at(zeros, row_bytes, row, blk) as f32;
                    let scale = f16::from_f32(scales[row * blocks + blk]).to_f32();
                    correction[row * blocks + blk] = (8. - z) * scale;
                }
            }
            if correction.iter().any(|c| *c != 0.) {
                let correction = model.add_const(
                    format!("{prefix}.correction"),
                    Tensor::from_shape(&[rows, blocks], &correction)?,
                )?;
                let gathered = model.wire_node(
                    format!("{prefix}.correction_rows"),
                    Gather::new(0),
                    &[correction, inputs[1]],
                )?[0];
                let gathered = expand_blocks(
                    model,
                    &format!("{prefix}.correction_x"),
                    gathered,
                    &blocks.to_dim(),
                    self.block_size,
                )?;
                out = wire_with_rank_broadcast(
                    format!("{prefix}.recentered"),
                    model,
                    add(),
                    &[out, gathered],
                )?[0];
            }
        }
        Ok(Some(out))
    }

    /// Spells the dequantization out in the graph, on the gathered rows: split the nibbles,
    /// broadcast the per-block scales and zeros back over their block, and `(value - zero) *
    /// scale`. Exact — the values are integral and below 256, where `/16` and the remainder
    /// are exact in f32, and the scales are used as they come.
    fn wire_dequant_graph(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        packed: &TDim,
        blocks: &TDim,
    ) -> TractResult<OutletId> {
        let rows =
            model.wire_node(format!("{prefix}.rows"), Gather::new(0), &[inputs[0], inputs[1]])?[0];
        let rows =
            model.wire_node(format!("{prefix}.rows_f32"), cast(f32::datum_type()), &[rows])?[0];
        let values = unpack_nibbles(model, &format!("{prefix}.values"), rows, packed)?;

        let scales =
            model.wire_node(format!("{prefix}.scales"), Gather::new(0), &[inputs[2], inputs[1]])?
                [0];
        let scales =
            model.wire_node(format!("{prefix}.scales_f32"), cast(f32::datum_type()), &[scales])?[0];
        let scales =
            expand_blocks(model, &format!("{prefix}.scales_x"), scales, blocks, self.block_size)?;

        let zeros = if self.has_zero_points {
            let packed_zeros = model.outlet_fact(inputs[3])?.shape[1].clone();
            let z = model.wire_node(
                format!("{prefix}.zeros"),
                Gather::new(0),
                &[inputs[3], inputs[1]],
            )?[0];
            let z =
                model.wire_node(format!("{prefix}.zeros_f32"), cast(f32::datum_type()), &[z])?[0];
            let z = unpack_nibbles(model, &format!("{prefix}.zeros_u"), z, &packed_zeros)?;
            expand_blocks(model, &format!("{prefix}.zeros_x"), z, blocks, self.block_size)?
        } else {
            model.add_const(format!("{prefix}.zeros"), tensor0(8f32))?
        };

        let centered =
            wire_with_rank_broadcast(format!("{prefix}.centered"), model, sub(), &[values, zeros])?
                [0];
        Ok(wire_with_rank_broadcast(
            format!("{prefix}.dequant"),
            model,
            mul(),
            &[centered, scales],
        )?[0])
    }
}
