use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::ops::math::round_ties_to_even;

use crate::model::ParsingContext;
use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("FakeQuantWithMinMaxVars", fake_quant_with_min_max_vars);
}

fn fake_quant_with_min_max_vars(
    _ctx: &ParsingContext,
    node: &NodeDef,
) -> TractResult<Box<dyn InferenceOp>> {
    let narrow_range = node.get_attr_bool("narrow_range")?;
    let num_bits = node.get_attr_int("num_bits")?;
    Ok(expand(FakeQuantWithMinMaxVars::new(narrow_range, num_bits)))
}

#[derive(Clone, Debug, new, Hash)]
struct FakeQuantWithMinMaxVars {
    narrow_range: bool,
    num_bits: usize,
}



impl FakeQuantWithMinMaxVars {
    fn step(&self, min: &Tensor, max: &Tensor) -> TractResult<f32> {
        let min = min.to_scalar::<f32>()?;
        let max = max.to_scalar::<f32>()?;
        let amplitude = max - min;
        let scale_len = 2_usize.pow(self.num_bits as u32) - 1 - self.narrow_range as usize;
        Ok(amplitude / scale_len as f32)
    }
}

impl Expansion for FakeQuantWithMinMaxVars {
    fn name(&self) -> Cow<str> {
        "FakeQuantWithMinMaxVars".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[1].shape, shapefactoid!())?;
        s.equals(&inputs[2].shape, shapefactoid!())?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(min), Some(max)) = (
            target.outlet_fact(inputs[1])?.konst.as_ref(),
            target.outlet_fact(inputs[2])?.konst.as_ref(),
        ) {
            let rank = target.outlet_fact(inputs[0])?.rank();
            macro_rules! cst {
                ($id:ident, $value: expr) => {
                    let $id = tensor0($value).broadcast_into_rank(rank)?;
                    let $id = target.add_const(prefix.to_string() + "." + stringify!($id), $id)?;
                };
            }
            let step = self.step(min, max)?;
            let min = *min.to_scalar::<f32>()?;
            let max = *max.to_scalar::<f32>()?;
            let min_adj = step * round_ties_to_even(min / step);
            let max_adj = max - min + min_adj;
            let wire = inputs[0];
            cst!(min_adj, min_adj);
            cst!(max_adj, max_adj);
            cst!(step, step);
            let wire = target.wire_node(
                format!("{prefix}.clamp_min"),
                ops::math::max(),
                &[wire, min_adj],
            )?[0];
            let wire = target.wire_node(
                format!("{prefix}.clamp_max"),
                ops::math::min(),
                &[max_adj, wire],
            )?[0];
            let wire = target.wire_node(
                format!("{prefix}.sub-min"),
                ops::math::sub(),
                &[wire, min_adj],
            )?[0];
            let wire = target.wire_node(
                format!("{prefix}.div-step"),
                ops::math::div(),
                &[wire, step],
            )?[0];
            let wire = target.wire_node(
                format!("{prefix}.round"),
                ops::math::round_half_to_even(),
                &[wire],
            )?[0];
            let wire = target.wire_node(
                format!("{prefix}.mul-step"),
                ops::math::mul(),
                &[wire, step],
            )?[0];
            target.wire_node(format!("{prefix}.add-min"), ops::math::add(), &[wire, min_adj])
        } else {
            bail!("Operator can not be made a TypedOp.")
        }
    }
}
