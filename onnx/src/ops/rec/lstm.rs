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

        wire!(Wi = array::Slice::new(0, 0.to_dim() * &h_size, 1.to_dim() * &h_size), W);
        wire!(Wo = array::Slice::new(0, 1.to_dim() * &h_size, 2.to_dim() * &h_size), W);
        wire!(Wf = array::Slice::new(0, 2.to_dim() * &h_size, 3.to_dim() * &h_size), W);
        wire!(Wc = array::Slice::new(0, 3.to_dim() * &h_size, 4.to_dim() * &h_size), W);

        wire!(Ri = array::Slice::new(0, 0.to_dim() * &h_size, 1.to_dim() * &h_size), R);
        wire!(Ro = array::Slice::new(0, 1.to_dim() * &h_size, 2.to_dim() * &h_size), R);
        wire!(Rf = array::Slice::new(0, 2.to_dim() * &h_size, 3.to_dim() * &h_size), R);
        wire!(Rc = array::Slice::new(0, 3.to_dim() * &h_size, 4.to_dim() * &h_size), R);

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

        let matmul_t = EinSum::new("mk,nk->mn".parse()?, f32::datum_type());

        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        wire!(Xt_WiT = matmul_t.clone(), Xt, Wi);
        wire!(Ht_1_RiT = matmul_t.clone(), Ht_1, Ri);
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
        wire!(Xt_WfT = matmul_t.clone(), Xt, Wf);
        wire!(Ht_1_RfT = matmul_t.clone(), Ht_1, Rf);
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
        wire!(Xt_WcT = matmul_t.clone(), Xt, Wc);
        wire!(Ht_1_RcT = matmul_t.clone(), Ht_1, Rc);
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
        wire!(Xt_WoT = matmul_t.clone(), Xt, Wo);
        wire!(Ht_1_RoT = matmul_t, Ht_1, Ro);
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
        body.set_output_outlets(&[Ht_fixed, Ct_fixed])?;

        Ok(())
    }
}
