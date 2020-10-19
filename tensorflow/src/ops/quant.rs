use tract_hir::internal::*;
use tract_hir::ops;

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

tract_data::impl_dyn_hash!(FakeQuantWithMinMaxVars);

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

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
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
            let step = self.step(&min, &max)?;
            let min = *min.to_scalar::<f32>()?;
            let wire = &inputs[0..1];
            let wire = target.wire_node(
                format!("{}.sub-min", prefix),
                ops::math::add::unary(tensor0(-min).broadcast_into_rank(rank)?.into_arc_tensor()),
                &wire,
            )?;
            let wire = target.wire_node(
                format!("{}.div-step", prefix),
                ops::math::mul::unary(
                    tensor0(step.recip()).broadcast_into_rank(rank)?.into_arc_tensor(),
                ),
                &wire,
            )?;
            let wire =
                target.wire_node(format!("{}.round", &*prefix), ops::math::round(), &wire)?;
            let wire = target.wire_node(
                format!("{}.mul-step", &*prefix),
                ops::math::mul::unary(tensor0(step).broadcast_into_rank(rank)?.into_arc_tensor()),
                &wire,
            )?;
            target.wire_node(
                format!("{}.add-min", &*prefix),
                ops::math::add::unary(tensor0(min).broadcast_into_rank(rank)?.into_arc_tensor()),
                &wire,
            )
        } else {
            bail!("Operator can not be made a TypedOp.")
        }
    }
}
