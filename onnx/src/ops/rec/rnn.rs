use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::einsum::EinSum;

use super::common::CommonRec;
use super::common::WireBody;

pub fn rnn(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let rnn = RNN { fore: Box::new(ops::math::tanh()), back: Box::new(ops::math::tanh()) };
    let common = CommonRec::from_node_and_options(pb, 3, 0, Box::new(rnn))?;
    Ok((expand(common), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct RNN {
    pub fore: Box<dyn TypedOp>,
    pub back: Box<dyn TypedOp>,
}

impl WireBody for RNN {
    fn name(&self) -> &'static str {
        "RNN"
    }

    fn w_b_multipliers(&self) -> (usize, usize) {
        (1, 2)
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

        let bias = if let Some(b) = b {
            wire!(Wbi = array::Slice::new(1, 0.to_dim() * &h_size, 1.to_dim() * &h_size), b);
            wire!(Rbi = array::Slice::new(1, 1.to_dim() * &h_size, 2.to_dim() * &h_size), b);
            wire!(bi = math::add(), Wbi, Rbi);
            Some(bi)
        } else {
            None
        };

        let matmul_t = EinSum::new("mk,nk->mn".parse()?, f32::datum_type());

        // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        wire!(Xt_WiT = matmul_t.clone(), Xt, W);
        wire!(Ht_1_RiT = matmul_t, Ht_1, R);

        wire!(ht0 = math::add(), Xt_WiT, Ht_1_RiT);
        let mut ht0 = ht0;
        if let Some(bias) = bias {
            wire!(ht_bias = math::add(), ht0, bias);
            ht0 = ht_bias;
        }
        wire!(Ht = self.fore.clone(), ht0);

        wire!(y_h = AxisOp::Add(1), Ht);
        body.set_output_outlets(&[y_h])?;
        Ok(())
    }
}
