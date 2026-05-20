use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::array::Slice;
use tract_hir::tract_core::ops::einsum::EinSum;
use tract_hir::tract_core::ops::rec::OptGru;

use super::common::CommonRec;
use super::common::WireBody;

pub fn gru(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let gru = GRU {
        f: Box::new(ops::nn::sigmoid()),
        g: Box::new(ops::math::tanh()),
        linear_before_reset: pb.get_attr("linear_before_reset").unwrap_or(false),
    };
    let common = CommonRec::from_node_and_options(pb, 3, 0, Box::new(gru))?;

    Ok((expand(common), vec![]))
}

#[derive(Debug, Clone)]
pub struct GRU {
    pub f: Box<dyn TypedOp>,
    pub g: Box<dyn TypedOp>,
    pub linear_before_reset: bool,
}

impl WireBody for GRU {
    fn name(&self) -> &'static str {
        "GRU"
    }

    fn w_b_multipliers(&self) -> (usize, usize) {
        (3, 6)
    }

    fn have_extra_c_state(&self) -> bool {
        false
    }

    /// Lower this GRU directly to a fused `OptGru` op when the attribute
    /// combination is supported (Phase 1: f32 only, concrete hidden_size,
    /// concrete input_size, no peepholes, no sequence_lens, no extra C state
    /// — all of which hold for DFN3's GRU layers).
    fn try_wire_fused(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        common: &CommonRec,
        inputs: &[OutletId],
        dir: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        // OPT-IN until the perf work lands. The current OptGru eval uses
        // ndarray matmuls and is ~4x slower than the Scan path (the fused-op
        // win requires swapping in tract-linalg MMM/MMV kernels — Phase 2).
        // Correctness is proven (see onnx/tests/opt_gru_vs_scan.rs + the
        // df_dec reference comparison), so the op is enabled explicitly via
        // TRACT_ENABLE_OPT_GRU=1 for benchmarking / further development.
        // Do NOT default-enable until OptGru is at least as fast as Scan.
        if std::env::var("TRACT_ENABLE_OPT_GRU").is_err() {
            return Ok(None);
        }

        // Phase 1 support gate. Bail to Scan path for anything outside scope.
        let x_fact = target.outlet_fact(inputs[0])?.clone();
        if x_fact.datum_type != f32::datum_type() {
            return Ok(None);
        }
        if common.optional_sequence_lens_input.is_some() {
            // Non-trivial seq_lens not supported in fused path.
            return Ok(None);
        }
        if common.optional_p_input.is_some() {
            return Ok(None);
        }

        let r_fact = target.outlet_fact(inputs[2])?.clone();
        let Ok(hidden_size) = r_fact.shape[2].to_usize() else {
            return Ok(None);
        };
        let Ok(input_size) = x_fact.shape[2].to_usize() else {
            return Ok(None);
        };

        // Batch axis index (in raw ONNX input X layout).
        let batch_axis_in_x = if common.batch_first { 0 } else { 1 };
        let batch_dim = x_fact.shape[batch_axis_in_x].clone();

        // ── Prepare OptGru inputs ──────────────────────────────────────

        // X: [seq_len, batch, input_size] (transpose if batch_first).
        let x = if common.batch_first {
            // [batch, seq, input] → [seq, batch, input]
            target.wire_node(
                format!("{prefix}.opt_gru.x_seq_first"),
                AxisOp::Move(0, 1),
                &[inputs[0]],
            )?[0]
        } else {
            inputs[0]
        };

        // W: [num_dir, 3*hidden, input_size] → [3*hidden, input_size]
        let w_dir = target.wire_node(
            format!("{prefix}.opt_gru.w_dir"),
            Slice::new(0, dir, dir + 1),
            &[inputs[1]],
        )?[0];
        let w = target.wire_node(format!("{prefix}.opt_gru.w"), AxisOp::Rm(0), &[w_dir])?[0];

        // R: [num_dir, 3*hidden, hidden] → [3*hidden, hidden]
        let r_dir = target.wire_node(
            format!("{prefix}.opt_gru.r_dir"),
            Slice::new(0, dir, dir + 1),
            &[inputs[2]],
        )?[0];
        let r = target.wire_node(format!("{prefix}.opt_gru.r"), AxisOp::Rm(0), &[r_dir])?[0];

        // B: optional [num_dir, 6*hidden] → [6*hidden]; else zeros.
        let b = if let Some(slot) = common.optional_bias_input {
            let b_dir = target.wire_node(
                format!("{prefix}.opt_gru.b_dir"),
                Slice::new(0, dir, dir + 1),
                &[inputs[slot]],
            )?[0];
            target.wire_node(format!("{prefix}.opt_gru.b"), AxisOp::Rm(0), &[b_dir])?[0]
        } else {
            target.add_const(
                format!("{prefix}.opt_gru.b_zero"),
                Tensor::zero::<f32>(&[6 * hidden_size])?,
            )?
        };

        // h0: optional [num_dir, batch, hidden] → [batch, hidden]; else zeros.
        let h0 = if let Some(slot) = common.optional_initial_h_input {
            let mut input = inputs[slot];
            if common.batch_first {
                // [batch, num_dir, hidden] → [num_dir, batch, hidden]
                input = target.wire_node(
                    format!("{prefix}.opt_gru.h0_dir_first"),
                    AxisOp::Move(1, 0),
                    &[input],
                )?[0];
            }
            let h_dir = target.wire_node(
                format!("{prefix}.opt_gru.h0_dir"),
                Slice::new(0, dir, dir + 1),
                &[input],
            )?[0];
            target.wire_node(format!("{prefix}.opt_gru.h0"), AxisOp::Rm(0), &[h_dir])?[0]
        } else {
            // Need concrete batch for the zero const. If symbolic, fall back to Scan.
            let Ok(batch) = batch_dim.to_usize() else {
                return Ok(None);
            };
            target.add_const(
                format!("{prefix}.opt_gru.h0_zero"),
                Tensor::zero::<f32>(&[batch, hidden_size])?,
            )?
        };

        // ── Emit OptGru ────────────────────────────────────────────────

        let opt_gru =
            OptGru { hidden_size, input_size, linear_before_reset: self.linear_before_reset };
        let gru_outputs =
            target.wire_node(format!("{prefix}.opt_gru"), opt_gru, &[x, w, r, b, h0])?;
        // OptGru emits Y in [batch, seq, hidden] (Scan-accumulator layout)
        // and Y_h in [batch, 1, hidden] (Scan-body-output layout). Apply the
        // SAME post-Scan wire ops as `common.rs`'s Scan flow, so the optimizer
        // sees identical downstream-op chains for both lowerings.
        let y_scan_layout = gru_outputs[0];
        let y_h_scan_layout = gru_outputs[1];

        // ── Wire outputs into ONNX layout (mirrors common.rs Scan branch) ──

        let mut result = tvec!();
        if let Some(_slot) = common.optional_y_output {
            let y = if common.batch_first {
                // ONNX: Y.shape = [batch, seq, num_directions, hidden]
                // OptGru emits [batch, seq, hidden]; insert num_dir at axis 2.
                target.wire_node(
                    format!("{prefix}.opt_gru.y_add_dir"),
                    AxisOp::Add(2),
                    &[y_scan_layout],
                )?[0]
            } else {
                // ONNX: Y.shape = [seq, num_directions, batch, hidden]
                // OptGru emits [batch, seq, hidden]; Move(1,0) → [seq, batch, hidden],
                // Add(1) → [seq, 1, batch, hidden]. Same ops as Scan path.
                let y_batch_middle = target.wire_node(
                    format!("{prefix}.opt_gru.y_batch_middle"),
                    AxisOp::Move(1, 0),
                    &[y_scan_layout],
                )?[0];
                target.wire_node(
                    format!("{prefix}.opt_gru.y_add_dir"),
                    AxisOp::Add(1),
                    &[y_batch_middle],
                )?[0]
            };
            result.push(y);
        }
        if let Some(_slot) = common.optional_y_h_output {
            let y_h = if common.batch_first {
                // ONNX Y_h: [batch, num_dir, hidden]. OptGru emits [batch, 1, hidden].
                // Already in batch_first layout — no transform needed.
                y_h_scan_layout
            } else {
                // ONNX Y_h: [num_dir, batch, hidden]. OptGru emits [batch, 1, hidden].
                // Move(1, 0) → [1, batch, hidden]. Same as Scan path.
                target.wire_node(
                    format!("{prefix}.opt_gru.y_h_swap"),
                    AxisOp::Move(1, 0),
                    &[y_h_scan_layout],
                )?[0]
            };
            result.push(y_h);
        }

        Ok(Some(result))
    }

    #[allow(non_snake_case)]
    fn wire_body(&self, prefix: &str, body: &mut TypedModel) -> TractResult<()> {
        use tract_hir::ops::{array, math};
        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = body.wire_node(
                    format!("{}.{}", prefix, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        }

        let Xt: OutletId = body.node_by_name("Xt").unwrap().id.into();
        let W: OutletId = body.node_by_name("W").unwrap().id.into();
        let R: OutletId = body.node_by_name("R").unwrap().id.into();
        let Ht_1: OutletId = body.node_by_name("Ht_1").unwrap().id.into();
        let b: Option<OutletId> = body.node_by_name("b").ok().map(|n| n.id.into());

        let h_size = body.outlet_fact(R)?.shape[1].clone();

        wire!(Rz = array::Slice::new(0, 0.to_dim() * &h_size, 1.to_dim() * &h_size), R);
        wire!(Rr = array::Slice::new(0, 1.to_dim() * &h_size, 2.to_dim() * &h_size), R);
        wire!(Rh = array::Slice::new(0, 2.to_dim() * &h_size, 3.to_dim() * &h_size), R);

        wire!(Wz = array::Slice::new(0, 0.to_dim() * &h_size, 1.to_dim() * &h_size), W);
        wire!(Wr = array::Slice::new(0, 1.to_dim() * &h_size, 2.to_dim() * &h_size), W);
        wire!(Wh = array::Slice::new(0, 2.to_dim() * &h_size, 3.to_dim() * &h_size), W);

        let dt = body.outlet_fact(Xt)?.datum_type;
        let matmul_t = EinSum::new("mk,nk->mn".parse()?, dt);

        // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        wire!(Xt_WzT = matmul_t.clone(), Xt, Wz);
        wire!(Ht_1_RzT = matmul_t.clone(), Ht_1, Rz);
        wire!(zt0 = math::add(), Xt_WzT, Ht_1_RzT);
        let mut zt0 = zt0;
        if let Some(b) = b {
            wire!(Wbz = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), b);
            wire!(Rbz = array::Slice::new(1, 3.to_dim() * &h_size, 4.to_dim() * &h_size), b);
            wire!(Wbz_Rbz = math::add(), Wbz, Rbz);
            wire!(zt0_biased = math::add(), zt0, Wbz_Rbz);
            zt0 = zt0_biased
        };
        wire!(zt = self.f.clone(), zt0);

        // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        wire!(Xt_WrT = matmul_t.clone(), Xt, Wr);
        wire!(Ht_1_RrT = matmul_t.clone(), Ht_1, Rr);
        wire!(rt0 = math::add(), Xt_WrT, Ht_1_RrT);
        let mut rt0 = rt0;
        if let Some(b) = b {
            wire!(Wbr = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), b);
            wire!(Rbr = array::Slice::new(1, 4.to_dim() * &h_size, 5.to_dim() * &h_size), b);
            wire!(Wbr_Rbr = math::add(), Wbr, Rbr);
            wire!(rt0_biased = math::add(), rt0, Wbr_Rbr);
            rt0 = rt0_biased
        };
        wire!(rt = self.f.clone(), rt0);

        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        wire!(Xt_WhT = matmul_t.clone(), Xt, Wh);
        let rt_Ht_1_RhT_Rbh = if self.linear_before_reset {
            // rt (.) (Ht-1*(Rh^T) + Rbh)
            wire!(Ht_1_RhT = matmul_t, Ht_1, Rh);
            let Ht_1_RhT_Rbh = if let Some(b) = b {
                wire!(Rbh = array::Slice::new(1, 5.to_dim() * &h_size, 6.to_dim() * &h_size), b);
                wire!(Ht_1_RhT_Rbh = math::add(), Ht_1_RhT, Rbh);
                Ht_1_RhT_Rbh
            } else {
                Ht_1_RhT
            };
            wire!(rt_Ht_1_RhT_Rbh = math::mul(), rt, Ht_1_RhT_Rbh);
            rt_Ht_1_RhT_Rbh
        } else {
            // (rt (.) Ht-1)*(Rh^T) + Rbh
            wire!(rt_Ht_1 = math::mul(), rt, Ht_1);
            wire!(rt_Ht_1_RhT = matmul_t, rt_Ht_1, Rh);
            if let Some(b) = b {
                wire!(Rbh = array::Slice::new(1, 5.to_dim() * &h_size, 6.to_dim() * &h_size), b);
                wire!(rt_Ht_1_RhT_Rbh = math::add(), rt_Ht_1_RhT, Rbh);
                rt_Ht_1_RhT_Rbh
            } else {
                rt_Ht_1_RhT
            }
        };
        wire!(ht0 = math::add(), Xt_WhT, rt_Ht_1_RhT_Rbh);
        let mut ht0 = ht0;
        if let Some(b) = b {
            wire!(Wbh = array::Slice::new(1, 2.to_dim() * &h_size, 3.to_dim() * &h_size), b);
            wire!(ht0_biased = math::add(), ht0, Wbh);
            ht0 = ht0_biased
        }
        wire!(ht = self.g.clone(), ht0);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        let one: OutletId =
            body.add_const("one", tensor2(&[[1f32]]).cast_to_dt(dt)?.into_owned())?;
        wire!(one_sub_zt = math::sub(), one, zt);
        wire!(one_sub_zt_ht = math::mul(), one_sub_zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(Ht = math::add(), one_sub_zt_ht, zt_Ht_1);

        wire!(y_h = AxisOp::Add(1), Ht);
        body.select_output_outlets(&[y_h])?;
        Ok(())
    }
}
