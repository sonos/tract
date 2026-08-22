use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::konst::Const;
use tract_core::ops::math::add;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage, Q4_0};

// com.microsoft MatMulNBits: Y = A @ dequant(B)^T (+ bias)
//   A:           float [.., K]
//   B (Q4):      uint8 [N, n_blocks, blob]   (blob = block_size/2, two 4-bit weights per byte)
//   scales:      float [N * n_blocks]
//   zero_points: uint8 [N * ceil(n_blocks/2)] packed (optional; default 8)
//   bias:        float [N] (optional)
// block_size 32 + symmetric + K a multiple of 32 keeps the weight as a Q4_0 block-quant
// constant (dequantized inside the matmul packer); any other shape dequantizes the constant
// to a plain float [N, K] weight and emits a plain matmul (EinSum).
pub fn mat_mul_nbits(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let k: usize = node.get_attr("K")?;
    let n: usize = node.get_attr("N")?;
    let bits: usize = node.get_attr_opt("bits")?.unwrap_or(4);
    let block_size: usize = node.get_attr("block_size")?;
    ensure!(bits == 4, "MatMulNBits: only bits=4 is supported (got {bits})");
    let mut opt = crate::model::optional_inputs(node).skip(3);
    let zp_input = opt.next().unwrap();
    let gidx_input = opt.next().unwrap();
    let bias_input = opt.next().unwrap();
    ensure!(gidx_input.is_none(), "MatMulNBits: g_idx (act-order) is unsupported");
    Ok((expand(MatMulNBits { k, n, block_size, zp_input, bias_input }), vec![]))
}

/// Evaluates the constant cone feeding `outlet`, so a weight reachable only through constant
/// ops reads back as a constant. `None` when the cone is not entirely constant: a node that is
/// not stateless, or one whose inputs are not themselves constant, stops the walk.
///
/// This reads the graph and never edits it. A `ModelPatch` would be the idiomatic edit, but it
/// cannot be applied from here: `wire` runs mid-translation, and `ModelPatch::apply` finishes
/// by collecting every node the shunt left without a successor. Against a model that has no
/// outputs yet, nothing anchors a constant whose remaining consumers are still to be
/// translated -- the shared lookup table of a sub-4-bit export is collected while folding the
/// first layer's weight, and the next layer then wires from a retired outlet. Leaving the cone
/// in place costs nothing: it is dead once the weight is read, and the compaction after
/// translation drops it.
fn fold_const_cone(model: &TypedModel, outlet: OutletId) -> TractResult<Option<Arc<Tensor>>> {
    let order =
        tract_core::model::order::eval_order_for_nodes(&model.nodes, &[], &[outlet.node], &[])?;
    let mut values: HashMap<OutletId, Arc<Tensor>> = HashMap::default();
    for n in order {
        let node = model.node(n);
        for (slot, o) in node.outputs.iter().enumerate() {
            if let Some(k) = o.fact.konst.as_ref().filter(|k| k.is_plain()) {
                values.insert(OutletId::new(n, slot), k.clone());
            }
        }
        if !node.op.is_stateless() || node.inputs.is_empty() {
            continue;
        }
        let Some(inputs) = node
            .inputs
            .iter()
            .map(|i| values.get(i).cloned().map(|t| t.into_tvalue()))
            .collect::<Option<TVec<_>>>()
        else {
            continue;
        };
        let Ok(res) = node.op.eval_with_session(n, &TurnState::default(), inputs) else {
            continue;
        };
        for (slot, output) in res.into_iter().enumerate() {
            values.insert(OutletId::new(n, slot), output.into_arc_tensor());
        }
    }
    Ok(values.remove(&outlet))
}

#[derive(Debug, Clone)]
struct MatMulNBits {
    k: usize,
    n: usize,
    block_size: usize,
    zp_input: Option<usize>,
    bias_input: Option<usize>,
}

impl Expansion for MatMulNBits {
    fn name(&self) -> StaticName {
        "MatMulNBits".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        let n = self.n.to_dim();
        s.given(&inputs[0].rank, move |s, rank| {
            let rank = rank as usize;
            s.equals(&outputs[0].rank, rank as i64)?;
            for ax in 0..rank - 1 {
                s.equals(&outputs[0].shape[ax], &inputs[0].shape[ax])?;
            }
            s.equals(&outputs[0].shape[rank - 1], n.clone())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let (k, n, block_size) = (self.k, self.n, self.block_size);
        let n_blocks = k.div_ceil(block_size);
        let blob = block_size.div_ceil(2);
        let zp_blob = n_blocks.div_ceil(2);

        // Read the constant quantized weight, scales and (optional) zero points. The weight
        // need not be an initializer: tied-embedding exports reach it through a reshape, and
        // 2-bit ones through a lookup table, both constant. The model is still being
        // translated, so it has no outputs to drive a whole-graph fold; patch this input's
        // own cone instead.
        let b_k = match model.outlet_fact(inputs[1])?.konst.clone() {
            Some(k) => k,
            None => fold_const_cone(model, inputs[1])?
                .context("MatMulNBits: quantized weight B must be a constant")?,
        };
        let b_plain = b_k.try_as_plain()?;
        let b: &[u8] = b_plain.as_slice()?;
        let scales_k = model
            .outlet_fact(inputs[2])?
            .konst
            .clone()
            .context("MatMulNBits: scales must be a constant")?;
        let scales_f = scales_k.cast_to::<f32>()?;
        let scales_plain = scales_f.try_as_plain()?;
        let scales: &[f32] = scales_plain.as_slice()?;
        let zp_k = if let Some(i) = self.zp_input {
            Some(
                model
                    .outlet_fact(inputs[i])?
                    .konst
                    .clone()
                    .context("MatMulNBits: zero_points must be a constant")?,
            )
        } else {
            None
        };
        let zp_plain = match &zp_k {
            Some(t) => Some(t.try_as_plain()?),
            None => None,
        };
        let zp: Option<&[u8]> = match &zp_plain {
            Some(p) => Some(p.as_slice::<u8>()?),
            None => None,
        };

        // Dequantize to a [N, K] float weight; also keep the raw nibbles (logical row-major)
        // so the block-quant path can reuse the original int4 values without re-quantizing.
        let mut w = vec![0f32; n * k];
        let mut q_logical = vec![0u8; n * k];
        for col in 0..n {
            for blk in 0..n_blocks {
                let scale = scales[col * n_blocks + blk];
                let zero = match zp {
                    Some(zp) => {
                        let byte = zp[col * zp_blob + blk / 2];
                        if blk % 2 == 0 { byte & 0x0F } else { byte >> 4 }
                    }
                    None => 8,
                } as f32;
                let base = col * n_blocks * blob + blk * blob;
                for i in 0..block_size {
                    let kk = blk * block_size + i;
                    if kk >= k {
                        break;
                    }
                    let byte = b[base + i / 2];
                    let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    q_logical[col * k + kk] = nib;
                    w[col * k + kk] = (nib as f32 - zero) * scale;
                }
            }
        }
        // Y = A @ W^T, contracting K. Computed in f32, then cast to the input dtype.
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let rank = model.outlet_fact(inputs[0])?.rank();
        let a =
            model.wire_node(format!("{prefix}.cast_a"), cast(f32::datum_type()), &[inputs[0]])?[0];
        let lead: String = "abcdefgh".chars().take(rank - 1).collect();

        // block_size 32 + symmetric (no zero points) + K a multiple of 32 maps onto tract's
        // Q4_0 block-quant format: the weight stays int4 in memory and is dequantized inside the
        // matmul packer, instead of materializing a full f32 weight. Anything else (other block
        // sizes, asymmetric zero points, a partial last block) falls back to the f32 weight.
        let y = if block_size == 32 && self.zp_input.is_none() && k % 32 == 0 {
            let weights = Q4_0.pack_prequantized(&q_logical, scales, n, k)?;
            let bqs = BlockQuantStorage::new(Box::new(Q4_0), n, k, Arc::new(weights))?;
            let fact = Box::new(BlockQuantFact::new(Box::new(Q4_0), tvec!(1, n, k)));
            let wq = model.wire_node(
                format!("{prefix}.weight_bq"),
                Const::new_with_exotic_fact(
                    Arc::new(bqs.into_tensor_with_shape(f32::datum_type(), &[1, n, k])),
                    fact,
                )?,
                &[],
            )?[0];
            let axes = AxesMapping::from_strs(
                &[format!("{lead}k"), "gnk".to_string()],
                &[format!("{lead}n")],
            )?;
            model.wire_node(
                format!("{prefix}.matmul"),
                EinSum::new(axes, f32::datum_type()),
                &[a, wq],
            )?[0]
        } else {
            let w =
                model.add_const(format!("{prefix}.weight"), Tensor::from_shape(&[n, k], &w)?)?;
            let axes = AxesMapping::from_strs(
                &[format!("{lead}k"), "nk".to_string()],
                &[format!("{lead}n")],
            )?;
            model.wire_node(
                format!("{prefix}.matmul"),
                EinSum::new(axes, f32::datum_type()),
                &[a, w],
            )?[0]
        };
        let mut y = model.wire_node(format!("{prefix}.cast_y"), cast(dt), &[y])?[0];

        if let Some(i) = self.bias_input {
            y = wire_with_rank_broadcast(format!("{prefix}.bias"), model, add(), &[y, inputs[i]])?
                [0];
        }
        Ok(tvec!(y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed_weight(n: usize, n_blocks: usize, blob: usize) -> TractResult<Tensor> {
        let mut t = Tensor::zero::<u8>(&[n, n_blocks, blob])?;
        for (ix, b) in t.try_as_plain_mut()?.as_slice_mut::<u8>()?.iter_mut().enumerate() {
            *b = (ix % 251) as u8;
        }
        Ok(t)
    }

    /// The weight reaches `wire` through a constant expression instead of an initializer, as
    /// it does in a tied-embedding or sub-4-bit export. On a real model the analyser gives up
    /// on a cone over `CONST_FOLD_MEM_BUDGET`, which would mean shipping megabytes of weights
    /// to reproduce; a hand-built typed model arrives with the same unresolved fact for any
    /// cone over sixteen elements, and — as during translation — has no outputs yet.
    #[test]
    fn weight_behind_a_constant_expression() -> TractResult<()> {
        let (k, n, block_size) = (32usize, 8usize, 32usize);
        let (n_blocks, blob) = (k.div_ceil(block_size), block_size.div_ceil(2));
        let op = MatMulNBits { k, n, block_size, zp_input: None, bias_input: None };
        let weight = packed_weight(n, n_blocks, blob)?;
        let scales = Tensor::from_shape(
            &[n * n_blocks],
            &(0..n * n_blocks).map(|i| 0.1 + i as f32 / 32.).collect::<Vec<f32>>(),
        )?;

        let run = |through_a_cone: bool| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact([2, k]))?;
            let b = if through_a_cone {
                let flat = model
                    .add_const("b.flat", weight.clone().into_shape(&[n * n_blocks * blob])?)?;
                let wire = model.wire_node(
                    "b",
                    AxisOp::Reshape(
                        0,
                        tvec![(n * n_blocks * blob).to_dim()],
                        tvec![n.to_dim(), n_blocks.to_dim(), blob.to_dim()],
                    ),
                    &[flat],
                )?[0];
                assert!(model.outlet_fact(wire)?.konst.is_none(), "cone folded too early");
                wire
            } else {
                model.add_const("b", weight.clone())?
            };
            let s = model.add_const("scales", scales.clone())?;
            let y = op.wire("mmnb", &mut model, &[a, b, s])?;
            model.select_output_outlets(&y)?;
            let input = Tensor::from_shape(
                &[2, k],
                &(0..2 * k).map(|i| (i as f32 / 7.).sin()).collect::<Vec<f32>>(),
            )?;
            let mut out = model.into_optimized()?.into_runnable()?.run(tvec!(input.into()))?;
            Ok(out.remove(0).into_tensor())
        };

        let expected = run(false)?;
        let folded = run(true)?;
        expected.close_enough(&folded, false)?;
        Ok(())
    }

    /// Two weights folded off one shared constant, the second wired only after the first has
    /// been folded -- the topology of a sub-4-bit export, where every layer unpacks through the
    /// same lookup table and the later layers are still untranslated when the first one folds.
    ///
    /// This is why the fold reads the graph instead of patching it. `ModelPatch::apply` ends by
    /// collecting every node its shunt left without a successor, and mid-translation the model
    /// has no outputs to anchor anything, so folding the first weight retires the shared table
    /// and the next layer wires from a dead outlet.
    #[test]
    fn shared_cone_source_survives_the_fold() -> TractResult<()> {
        let (k, n, block_size) = (32usize, 8usize, 32usize);
        let (n_blocks, blob) = (k.div_ceil(block_size), block_size.div_ceil(2));
        let op = MatMulNBits { k, n, block_size, zp_input: None, bias_input: None };
        let weight = packed_weight(n, n_blocks, blob)?;
        let flat_shape = n * n_blocks * blob;

        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([2, k]))?;
        let flat = model.add_const("b.flat", weight.clone().into_shape(&[flat_shape])?)?;
        let scales = model.add_const(
            "scales",
            Tensor::from_shape(&[n * n_blocks], &vec![0.25f32; n * n_blocks])?,
        )?;
        let reshape = || {
            AxisOp::Reshape(
                0,
                tvec![flat_shape.to_dim()],
                tvec![n.to_dim(), n_blocks.to_dim(), blob.to_dim()],
            )
        };

        // first consumer: folds, and its patch retires the cone
        let b0 = model.wire_node("b0", reshape(), &[flat])?[0];
        op.wire("mmnb0", &mut model, &[a, b0, scales])?;

        // second consumer, translated only now, still needs the shared constant
        let b1 = model.wire_node("b1", reshape(), &[flat])?[0];
        op.wire("mmnb1", &mut model, &[a, b1, scales])?;
        Ok(())
    }
}
