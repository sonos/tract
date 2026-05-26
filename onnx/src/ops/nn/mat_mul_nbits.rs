use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::add;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

// com.microsoft MatMulNBits: Y = A @ dequant(B)^T (+ bias)
//   A:           float [.., K]
//   B (Q4):      uint8 [N, n_blocks, blob]   (blob = block_size/2, two 4-bit weights per byte)
//   scales:      float [N * n_blocks]
//   zero_points: uint8 [N * ceil(n_blocks/2)] packed (optional; default 8)
//   bias:        float [N] (optional)
// The quantized weight is constant, so we dequantize it to a float [N, K] weight in Rust and
// emit a plain matmul (EinSum). A fused int4 kernel would be a follow-up perf optimization.
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

        // Dequantize to a [N, K] float weight.
        let mut w = vec![0f32; n * k];
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
                    let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 } as f32;
                    w[col * k + kk] = (q - zero) * scale;
                }
            }
        }
        let w = model.add_const(format!("{prefix}.weight"), Tensor::from_shape(&[n, k], &w)?)?;

        // Y = A @ W^T, contracting K. Computed in f32, then cast to the input dtype.
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let rank = model.outlet_fact(inputs[0])?.rank();
        let a =
            model.wire_node(format!("{prefix}.cast_a"), cast(f32::datum_type()), &[inputs[0]])?[0];
        let lead: String = "abcdefgh".chars().take(rank - 1).collect();
        let axes =
            AxesMapping::from_strs(&[format!("{lead}k"), "nk".to_string()], &[format!("{lead}n")])?;
        let y = model.wire_node(
            format!("{prefix}.matmul"),
            EinSum::new(axes, f32::datum_type()),
            &[a, w],
        )?[0];
        let mut y = model.wire_node(format!("{prefix}.cast_y"), cast(dt), &[y])?[0];

        if let Some(i) = self.bias_input {
            y = wire_with_rank_broadcast(format!("{prefix}.bias"), model, add(), &[y, inputs[i]])?
                [0];
        }
        Ok(tvec!(y))
    }
}
