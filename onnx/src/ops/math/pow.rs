use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn pow(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(Pow), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Pow;

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
        if dta.is_integer() && !dtb.is_integer() {
            let f = model.wire_node(
                format!("{}.cast-to-f64", name),
                tract_hir::ops::math::Cast::new(F64),
                inputs[0],
            )?;

            let f = tract_hir::ops::math::pow::bin_typed()
        }
    }
}
