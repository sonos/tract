use tract_core::infer::*;
use tract_core::internal::*;

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
    Ok(Box::new(FakeQuantWithMinMaxVars::new(narrow_range, num_bits)))
}

#[derive(Clone, Debug, new)]
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

impl Op for FakeQuantWithMinMaxVars {
    fn name(&self) -> Cow<str> {
        "tf.FakeQuantWithMinMaxVars".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for FakeQuantWithMinMaxVars {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, min, max) = args_3!(inputs);
        let step = self.step(&min, &max)?;
        let min = min.to_scalar::<f32>()?;
        let mut tensor = input.into_tensor().into_array::<f32>()?;
        tensor.mapv_inplace(|v| ((v - min) / step).round() * step + min);
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl InferenceRulesOp for FakeQuantWithMinMaxVars {
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

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(min), Some(max)) = (
            target.outlet_fact(mapping[&node.inputs[1]])?.konst.as_ref(),
            target.outlet_fact(mapping[&node.inputs[2]])?.konst.as_ref(),
        ) {
            let rank = target.outlet_fact(mapping[&node.inputs[0]])?.rank();
            let step = self.step(&min, &max)?;
            let min = *min.to_scalar::<f32>()?;
            let bc = |v| -> TractResult<Arc<Tensor>> {
                let mut t = tensor0(v);
                while t.rank() < rank {
                    t.insert_axis(0)?;
                }
                Ok(t.into_arc_tensor())
            };
            let wire = mapping[&node.inputs[0]];
            let wire = target.wire_node(
                format!("{}-sub-min", &*node.name),
                tract_core::ops::math::add::unary(bc(-min)?),
                &[wire],
            )?[0];
            let wire = target.wire_node(
                format!("{}-div-step", &*node.name),
                tract_core::ops::math::mul::unary(bc(step.recip())?),
                &[wire],
            )?[0];
            let wire = target.wire_node(
                format!("{}-round", &*node.name),
                tract_core::ops::math::round(),
                &[wire],
            )?[0];
            let wire = target.wire_node(
                format!("{}-mul-step", &*node.name),
                tract_core::ops::math::mul::unary(bc(step)?),
                &[wire],
            )?[0];
            let wire = target.wire_node(
                format!("{}-add-min", &*node.name),
                tract_core::ops::math::add::unary(bc(min)?),
                &[wire],
            )?[0];
            return Ok(tvec!(wire));
        }
        bail!("Operator can not be made a TypedOp.")
    }

    as_op!();
}

