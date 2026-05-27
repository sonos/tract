use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::einsum::EinSum;

use super::common::CommonRec;
use super::common::WireBody;

pub fn lstm(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let lstm = LSTM {
        f: Box::new(ops::nn::sigmoid()),
        g: Box::new(ops::math::tanh()),
        h: Box::new(ops::math::tanh()),
    };
    let common = CommonRec::from_node_and_options(pb, 3, 0, Box::new(lstm))?;
    Ok((expand(common), vec![]))
}

#[derive(Debug, Clone)]
pub struct LSTM {
    pub f: Box<dyn TypedOp>,
    pub g: Box<dyn TypedOp>,
    pub h: Box<dyn TypedOp>,
}

impl WireBody for LSTM {
    fn name(&self) -> &'static str {
        "LSTM"
    }

    fn w_b_multipliers(&self) -> (usize, usize) {
        (4, 8)
    }

    fn have_extra_c_state(&self) -> bool {
        true
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
        let Ct_1: OutletId = body.node_by_name("Ct_1").unwrap().id.into();
        let b: Option<OutletId> = body.node_by_name("b").ok().map(|n| n.id.into());
        let peepholes: Option<OutletId> = body.node_by_name("peepholes").ok().map(|n| n.id.into());

        let h_size = body.outlet_fact(R)?.shape[1].clone();
        let dt = body.outlet_fact(R)?.datum_type;

        // Gate fusion: compute all four gate pre-activations with a SINGLE matmul
        // per side (Xt·Wᵀ → [batch, 4*h], Ht-1·Rᵀ → [batch, 4*h]) and slice the
        // result per gate, instead of slicing the weights into 4 and doing 4
        // separate matmuls. This gives fewer + larger kernel invocations, one B
        // pack instead of four, and a contiguous weight stream — mirroring how
        // TFLite/XNNPACK run an LSTM as one FC per gate group. ONNX gate order
        // in W/R is i, o, f, c.
        let matmul_t = EinSum::new("mk,nk->mn".parse()?, dt);
        wire!(Xt_WT = matmul_t.clone(), Xt, W);
        wire!(Ht_1_RT = matmul_t.clone(), Ht_1, R);
        wire!(Xt_WiT = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), Xt_WT);
        wire!(Xt_WoT = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), Xt_WT);
        wire!(Xt_WfT = array::Slice::new(1, 2.to_dim() * &h_size, 3.to_dim() * &h_size), Xt_WT);
        wire!(Xt_WcT = array::Slice::new(1, 3.to_dim() * &h_size, 4.to_dim() * &h_size), Xt_WT);
        wire!(Ht_1_RiT = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), Ht_1_RT);
        wire!(Ht_1_RoT = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), Ht_1_RT);
        wire!(Ht_1_RfT = array::Slice::new(1, 2.to_dim() * &h_size, 3.to_dim() * &h_size), Ht_1_RT);
        wire!(Ht_1_RcT = array::Slice::new(1, 3.to_dim() * &h_size, 4.to_dim() * &h_size), Ht_1_RT);

        let biases = if let Some(b) = b {
            wire!(Wbi = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), b);
            wire!(Wbo = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), b);
            wire!(Wbf = array::Slice::new(1, 2.to_dim() * &h_size, 3.to_dim() * &h_size), b);
            wire!(Wbc = array::Slice::new(1, 3.to_dim() * &h_size, 4.to_dim() * &h_size), b);

            wire!(Rbi = array::Slice::new(1, 4.to_dim() * &h_size, 5.to_dim() * &h_size), b);
            wire!(Rbo = array::Slice::new(1, 5.to_dim() * &h_size, 6.to_dim() * &h_size), b);
            wire!(Rbf = array::Slice::new(1, 6.to_dim() * &h_size, 7.to_dim() * &h_size), b);
            wire!(Rbc = array::Slice::new(1, 7.to_dim() * &h_size, 8.to_dim() * &h_size), b);

            wire!(bi = math::add(), Wbi, Rbi);
            wire!(bo = math::add(), Wbo, Rbo);
            wire!(bf = math::add(), Wbf, Rbf);
            wire!(bc = math::add(), Wbc, Rbc);

            Some((bi, bo, bf, bc))
        } else {
            None
        };

        let peepholes = if let Some(p) = peepholes {
            wire!(pi = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), p);
            wire!(po = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), p);
            wire!(pf = array::Slice::new(1, 2.to_dim() * &h_size, 3.to_dim() * &h_size), p);
            Some((pi, po, pf))
        } else {
            None
        };

        // Fused-epilogue fast path: tract's ONNX LSTM always uses standard
        // activations (f = sigmoid, g = h = tanh), so whenever there are no
        // peepholes and the hidden size is concrete we collapse the per-gate
        // sigmoid/tanh + cell/hidden elementwise chain (~15 separately
        // dispatched ops, each materialising an intermediate) into a single
        // LstmEpilogue op. Numerically identical (same activation kernels);
        // peephole / symbolic-hidden LSTMs fall through to the decomposed form.
        if peepholes.is_none()
            && let Ok(hidden) = h_size.to_usize()
        {
            use tract_hir::tract_core::ops::lstm_cell::LstmEpilogue;
            wire!(preact0 = math::add(), Xt_WT, Ht_1_RT);
            let preact = if let Some(b) = b {
                wire!(Wb_all = array::Slice::new(1, 0.to_dim() * &h_size, 4.to_dim() * &h_size), b);
                wire!(Rb_all = array::Slice::new(1, 4.to_dim() * &h_size, 8.to_dim() * &h_size), b);
                wire!(bias_all = math::add(), Wb_all, Rb_all);
                wire!(preact = math::add(), preact0, bias_all);
                preact
            } else {
                preact0
            };
            let outs = body.wire_node(
                format!("{prefix}.lstm_cell"),
                LstmEpilogue { hidden },
                &[preact, Ct_1],
            )?;
            wire!(Ht_fixed = AxisOp::Add(1), outs[0]);
            wire!(Ct_fixed = AxisOp::Add(1), outs[1]);
            body.select_output_outlets(&[Ht_fixed, Ct_fixed])?;
            return Ok(());
        }

        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        wire!(it0 = math::add(), Xt_WiT, Ht_1_RiT);
        let mut it0 = it0;
        if let Some(biases) = biases {
            wire!(it_bias = math::add(), it0, biases.0);
            it0 = it_bias;
        };
        if let Some(peephole) = peepholes {
            wire!(Pi_Ct_1 = math::mul(), peephole.0, Ct_1);
            wire!(it_peep = math::add(), Pi_Ct_1, it0);
            it0 = it_peep;
        }
        wire!(it = self.f.clone(), it0);

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        wire!(ft0 = math::add(), Xt_WfT, Ht_1_RfT);
        let mut ft0 = ft0;
        if let Some(biases) = biases {
            wire!(ft_bias = math::add(), ft0, biases.2);
            ft0 = ft_bias;
        };
        if let Some(peephole) = peepholes {
            wire!(Pf_Ct_1 = math::mul(), peephole.2, Ct_1);
            wire!(ft_peep = math::add(), Pf_Ct_1, ft0);
            ft0 = ft_peep;
        }
        wire!(ft = self.f.clone(), ft0);

        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        wire!(ct0 = math::add(), Xt_WcT, Ht_1_RcT);
        let mut ct0 = ct0;
        if let Some(biases) = biases {
            wire!(ct_bias = math::add(), ct0, biases.3);
            ct0 = ct_bias
        };
        wire!(ct = self.g.clone(), ct0);

        // Ct = ft (.) Ct-1 + it (.) ct
        wire!(ft_Ct_1 = math::mul(), ft, Ct_1);
        wire!(it_ct = math::mul(), it, ct);
        wire!(Ct = math::add(), ft_Ct_1, it_ct);

        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        wire!(ot0 = math::add(), Xt_WoT, Ht_1_RoT);
        let mut ot0 = ot0;
        if let Some(biases) = biases {
            wire!(ot_bias = math::add(), ot0, biases.1);
            ot0 = ot_bias
        };
        if let Some(peephole) = peepholes {
            wire!(Po_Ct = math::mul(), peephole.1, Ct);
            wire!(ot_peep = math::add(), Po_Ct, ot0);
            ot0 = ot_peep;
        }
        wire!(ot = self.f.clone(), ot0);

        // Ht = ot (.) h(Ct)
        wire!(h_Ct = self.h.clone(), Ct);
        wire!(Ht = math::mul(), ot, h_Ct);

        // onnx inner interface: [batch_size, input_size]
        // add sequence axis (chunk == 1)
        wire!(Ht_fixed = AxisOp::Add(1), Ht);
        wire!(Ct_fixed = AxisOp::Add(1), Ct);
        body.select_output_outlets(&[Ht_fixed, Ct_fixed])?;

        Ok(())
    }
}
