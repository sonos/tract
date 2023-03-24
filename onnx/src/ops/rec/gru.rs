use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::einsum::EinSum;

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

        let matmul_t = EinSum::new("mk,nk->mn".parse()?, f32::datum_type());

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
        let one: OutletId = body.add_const("one", tensor2(&[[1f32]]))?;
        wire!(one_sub_zt = math::sub(), one, zt);
        wire!(one_sub_zt_ht = math::mul(), one_sub_zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(Ht = math::add(), one_sub_zt_ht, zt_Ht_1);

        /*
        // Ht = ht + (- (zt (.) ht) + zt (.) Ht-1)
        wire!(zt_ht = math::mul(), zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(zt_Ht_1_sub_zt_ht = math::sub(), zt_Ht_1, zt_ht);
        wire!(Ht = math::add(), ht, zt_Ht_1_sub_zt_ht);
        */

        // Ht = ht - (zt (.) ht) + zt (.) Ht-1)
        /*
        wire!(zt_ht = math::mul(), zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(ht_zt_ht = math::sub(), ht, zt_ht);
        wire!(Ht = math::add(), ht_zt_ht, zt_Ht_1);
        */

        wire!(y_h = AxisOp::Add(1), Ht);
        body.set_output_outlets(&[y_h])?;
        Ok(())
    }
}
