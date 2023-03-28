use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::binary::wire_bin;

pub fn rem(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    if node.get_attr_opt::<i64>("fmod")? == Some(1) {
        Ok((ops::math::Rem.into_hir(), vec![]))
    } else {
        Ok((expand(RemInt), vec![]))
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct RemInt;

impl Expansion for RemInt {
    fn name(&self) -> Cow<str> {
        "Remint".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        tract_hir::ops::binary::rules(s, inputs, outputs, move |a, b| {
            a.common_super_type(b).with_context(|| format!("No super type for {a:?} and {b:?}"))
        })
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::logic::*;
        use tract_hir::ops::math::*;
        let zero = tract_hir::ops::activations::broadcast_scalar(0.0, model, inputs)?;
        let a = model.outlet_fact(inputs[0])?.datum_type;
        let b = model.outlet_fact(inputs[1])?.datum_type;
        let dt =
            a.common_super_type(b).with_context(|| format!("No super type for {a:?} and {b:?}"))?;
        let wires = tract_hir::ops::binary::wire_rank_broadcast(name, model, inputs)?;
        let wires = tract_hir::ops::binary::wire_cast(name, model, &wires, dt)?;
        if dt.is_unsigned() {
            return wire_bin(name, model, Rem, &wires);
        }
        // from onnx runtime:
        // auto res = x % y;
        // if ((res < 0 && y > 0) || (res > 0 && y < 0)) { res += y; }
        let zero = model.add_const(name.to_string() + ".zero", zero)?;
        let rem = wire_bin(name.to_string() + ".rem", model, Rem, &wires)?[0];
        let rem_is_neg = wire_bin(name.to_string() + ".rem_is_neg", model, Greater, &[zero, rem])?;
        let rem_is_pos = wire_bin(name.to_string() + ".rem_is_pos", model, Less, &[zero, rem])?;
        let b_is_neg = wire_bin(name.to_string() + ".b_is_neg", model, Greater, &[zero, wires[1]])?;
        let b_is_pos = wire_bin(name.to_string() + ".b_is_pos", model, Less, &[zero, wires[1]])?;
        let rem_is_neg_b_is_pos = wire_bin(
            name.to_string() + ".rem_is_neg_b_is_pos",
            model,
            And,
            &[rem_is_neg[0], b_is_pos[0]],
        )?;
        let rem_is_pos_b_is_neg = wire_bin(
            name.to_string() + ".rem_is_pos_b_is_neg",
            model,
            And,
            &[rem_is_pos[0], b_is_neg[0]],
        )?;
        let adjust = wire_bin(
            name.to_string() + ".adjust",
            model,
            Or,
            &[rem_is_pos_b_is_neg[0], rem_is_neg_b_is_pos[0]],
        )?;
        let adjusted = wire_bin(name.to_string() + ".adjusted", model, Add, &[rem, wires[1]])?;
        model.wire_node(
            name.to_string(),
            tract_core::ops::logic::Iff,
            &[adjust[0], adjusted[0], rem],
        )
    }
}
