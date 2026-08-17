use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::konst::Const;
use tract_core::ops::math::add;
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage, Q2_0_T, Q4_0};

// com.microsoft MatMulNBits: Y = A @ dequant(B)^T (+ bias)
//   A:           float [.., K]
//   B (Q4):      uint8 [N, n_blocks, blob]   (blob = block_size/2, two 4-bit weights per byte)
//   scales:      float [N * n_blocks]
//   zero_points: uint8 [N * ceil(n_blocks/2)] packed (optional; default 8)
//   bias:        float [N] (optional)
// block_size 32 + K a multiple of 32 keeps the weight as a Q4_0 block-quant constant
// (dequantized inside the matmul packer), with a zero point carried as a separate low-rank
// correction; any other shape dequantizes the constant to a plain float [N, K] weight and
// emits a plain matmul (EinSum).
pub fn mat_mul_nbits(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let k: usize = node.get_attr("K")?;
    let n: usize = node.get_attr("N")?;
    let bits: usize = node.get_attr_opt("bits")?.unwrap_or(4);
    let block_size: usize = node.get_attr("block_size")?;
    ensure!(
        bits == 4 || bits == 2,
        "MatMulNBits: only bits=4 and bits=2 are supported (got {bits})"
    );
    let mut opt = crate::model::optional_inputs(node).skip(3);
    let zp_input = opt.next().unwrap();
    let gidx_input = opt.next().unwrap();
    let bias_input = opt.next().unwrap();
    ensure!(gidx_input.is_none(), "MatMulNBits: g_idx (act-order) is unsupported");
    Ok((expand(MatMulNBits { k, n, bits, block_size, zp_input, bias_input }), vec![]))
}

#[derive(Debug, Clone)]
struct MatMulNBits {
    k: usize,
    n: usize,
    bits: usize,
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
        let (k, n, bits, block_size) = (self.k, self.n, self.bits, self.block_size);
        let n_blocks = k.div_ceil(block_size);
        let per_byte = 8 / bits;
        let blob = block_size.div_ceil(per_byte);
        let zp_blob = n_blocks.div_ceil(per_byte);
        let mask = ((1u16 << bits) - 1) as u8;
        // The code that dequantizes to zero when the export is symmetric, and the base the
        // block-quant formats bake in: 8 for Q4_0's `(code - 8) * s`, 1 for Q2_0_T's
        // `(code - 1) * s`.
        let (default_zero, base) = if bits == 4 { (8f32, 8f32) } else { (2f32, 1f32) };

        // Read the constant quantized weight, scales and (optional) zero points.
        let b_k = model
            .outlet_fact(inputs[1])?
            .konst
            .clone()
            .context("MatMulNBits: quantized weight B must be a constant")?;
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

        // Unpack the nibbles (logical row-major) and the per-block zero point.
        let mut q_logical = vec![0u8; n * k];
        let mut zeros = vec![0f32; n * n_blocks];
        for col in 0..n {
            for blk in 0..n_blocks {
                zeros[col * n_blocks + blk] = match zp {
                    Some(zp) => {
                        let byte = zp[col * zp_blob + blk / per_byte];
                        ((byte >> (bits * (blk % per_byte))) & mask) as f32
                    }
                    None => default_zero,
                };
                let row = col * n_blocks * blob + blk * blob;
                for i in 0..block_size {
                    let kk = blk * block_size + i;
                    if kk >= k {
                        break;
                    }
                    let byte = b[row + i / per_byte];
                    q_logical[col * k + kk] = (byte >> (bits * (i % per_byte))) & mask;
                }
            }
        }
        // Y = A @ W^T, contracting K. Computed in f32, then cast to the input dtype.
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let rank = model.outlet_fact(inputs[0])?.rank();
        let a =
            model.wire_node(format!("{prefix}.cast_a"), cast(f32::datum_type()), &[inputs[0]])?[0];
        let lead: String = "abcdefgh".chars().take(rank - 1).collect();

        // block_size 32 + K a multiple of 32 maps onto tract's Q4_0 block-quant format: the
        // weight stays int4 in memory and is dequantized inside the matmul packer, instead of
        // materializing a full f32 weight. Anything else (other block sizes, a partial last
        // block) falls back to the f32 weight.
        //
        // Q4_0 fixes the zero point at 8, so an asymmetric weight is split as
        // `(q - z) * s == (q - 8) * s + (8 - z) * s`. The second term is constant within a
        // block, so it contributes `sum_b (8 - z) * s * sum_{k in b} A[k]`: the block sums of
        // the activations against a `[N, K/32]` constant. Both terms use the f16-rounded
        // scale the packer stores, so the split is exact against the rounded weight.
        let y = if block_size % 32 == 0 && k % 32 == 0 && (bits == 2 || block_size == 32) {
            // Both formats block by 32. A wider quantization block holds one scale for
            // several of them, so it is split by repeating that scale, which changes nothing
            // about what the weight dequantizes to.
            let per_32: Vec<f32> = if block_size == 32 {
                scales.to_vec()
            } else {
                scales.iter().flat_map(|s| std::iter::repeat_n(*s, block_size / 32)).collect()
            };
            let bq: Box<dyn tract_linalg::block_quant::BlockQuant> =
                if bits == 4 { Box::new(Q4_0) } else { Box::new(Q2_0_T) };
            let weights = if bits == 4 {
                Q4_0.pack_prequantized(&q_logical, &per_32, n, k)?
            } else {
                Q2_0_T.pack_prequantized(&q_logical, &per_32, n, k)?
            };
            let bqs = BlockQuantStorage::new(bq.clone(), n, k, Arc::new(weights))?;
            let fact = Box::new(BlockQuantFact::new(bq, tvec!(1, n, k)));
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
            let y = model.wire_node(
                format!("{prefix}.matmul"),
                EinSum::new(axes, f32::datum_type()),
                &[a, wq],
            )?[0];
            let offsets: Vec<f32> = zeros
                .iter()
                .zip(scales)
                .map(|(z, s)| (base - z) * f16::from_f32(*s).to_f32())
                .collect();
            // Zero only when the export's zero point already coincides with the format's,
            // which a symmetric int4 export does and a 2-bit one (whose default is 2, not
            // Q2_0_T's 1) does not.
            if offsets.iter().any(|o| *o != 0.) {
                let offsets = model.add_const(
                    format!("{prefix}.zp_offsets"),
                    Tensor::from_shape(&[n, n_blocks], &offsets)?,
                )?;
                let blocked = model.wire_node(
                    format!("{prefix}.a_blocked"),
                    AxisOp::Reshape(
                        rank - 1,
                        tvec![k.to_dim()],
                        tvec![n_blocks.to_dim(), block_size.to_dim()],
                    ),
                    &[a],
                )?[0];
                let summed = model.wire_node(
                    format!("{prefix}.a_block_sums"),
                    Reduce::new(tvec![rank], Reducer::Sum),
                    &[blocked],
                )?[0];
                let summed =
                    model.wire_node(format!("{prefix}.a_sums"), AxisOp::Rm(rank), &[summed])?[0];
                let axes = AxesMapping::from_strs(
                    &[format!("{lead}z"), "nz".to_string()],
                    &[format!("{lead}n")],
                )?;
                let correction = model.wire_node(
                    format!("{prefix}.zp_correction"),
                    EinSum::new(axes, f32::datum_type()),
                    &[summed, offsets],
                )?[0];
                wire_with_rank_broadcast(
                    format!("{prefix}.unbias"),
                    model,
                    add(),
                    &[y, correction],
                )?[0]
            } else {
                y
            }
        } else {
            let mut w = vec![0f32; n * k];
            for col in 0..n {
                for blk in 0..n_blocks {
                    let scale = scales[col * n_blocks + blk];
                    let zero = zeros[col * n_blocks + blk];
                    for i in 0..block_size {
                        let kk = blk * block_size + i;
                        if kk >= k {
                            break;
                        }
                        w[col * k + kk] = (q_logical[col * k + kk] as f32 - zero) * scale;
                    }
                }
            }
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
