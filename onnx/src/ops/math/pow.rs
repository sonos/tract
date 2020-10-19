use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn pow(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(Pow), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Pow;

tract_data::impl_dyn_hash!(Pow);

impl Expansion for Pow {
    fn name(&self) -> Cow<str> {
        "Pow".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;

        s.with(&inputs[0].shape, move |s, a_shape| {
            s.with(&inputs[1].shape, move |s, b_shape| {
                if let Ok(Some(c_shape)) =
                    tract_hir::infer::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape])
                {
                    s.equals(&outputs[0].shape, c_shape)?;
                }
                Ok(())
            })
        })?;
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use DatumType::*;
        let dta = model.outlet_fact(inputs[0])?.datum_type;
        let dtb = model.outlet_fact(inputs[1])?.datum_type;
        let mut wires = tract_hir::ops::binary::wire_rank_broadcast(name, model, inputs)?;
        if dta.is_integer() != dtb.is_integer() {
            wires = tract_hir::ops::binary::wire_cast(name, model, &wires, F64)?;
            wires = model.wire_node(format!("{}.pow", name), tract_hir::ops::math::pow::bin_typed(), &wires)?;
            model.wire_node(name, tract_hir::ops::cast(dta), &wires)
        } else {
            let dt = dta.common_super_type(dtb).unwrap();
            wires = tract_hir::ops::binary::wire_cast(name, model, &wires, dt)?;
            model.wire_node(name, tract_hir::ops::math::pow::bin_typed(), &wires)
        }
    }
}
