use tract_core::internal::*;
use tract_core::ops as tractops;
use crate::tfpb::node_def::NodeDef;
use crate::model::ParsingContext;
use crate::model::TfOpRegister;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Equal", with_T!(tractops::logic::Equals::Bin));
    reg.insert("Greater", with_T!(tractops::logic::Greater::Bin));
    reg.insert("GreaterEqual", with_T!(tractops::logic::GreaterEqual::Bin));
    reg.insert("Less", with_T!(tractops::logic::Lesser::Bin));
    reg.insert("LessEqual", with_T!(tractops::logic::LesserEqual::Bin));
    reg.insert("LogicalAnd", |_, _| Ok(Box::new(tractops::logic::And::default())));
    reg.insert("LogicalOr", |_, _| Ok(Box::new(tractops::logic::Or::default())));
    reg.insert("Merge", merge);
    reg.insert("Switch", |_, _| Ok(Box::new(Switch)));
}

#[derive(Debug, Clone)]
pub struct Switch;

impl Op for Switch {
    fn name(&self) -> Cow<str> {
        "tf.Switch".into()
    }
}

impl StatelessOp for Switch {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, pred) = args_2!(inputs);
        let null = unsafe { Tensor::null_dt(input.datum_type(), input.shape())? };
        if *pred.to_scalar::<bool>()? {
            Ok(tvec!(null.into(), input))
        } else {
            Ok(tvec!(input, null.into()))
        }
    }
}

impl InferenceRulesOp for Switch {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        s.equals(&inputs[1].datum_type, DatumType::Bool)?;
        s.equals(&inputs[1].shape, shapefact!())?;
        for i in 0..outputs.len() {
            s.equals(&inputs[0].datum_type, &outputs[i].datum_type)?;
            s.equals(&inputs[0].shape, &outputs[i].shape)?;
        }
        Ok(())
    }

    inference_op_as_op!();
}

fn merge(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let inputs = pb.get_attr_int::<i32>("N")?;
    Ok(Box::new(Merge::new(inputs as usize)))
}

#[derive(Debug, Clone, new)]
pub struct Merge {
    n: usize,
}

impl Op for Merge {
    fn name(&self) -> Cow<str> {
        "tf.Merge".into()
    }
}

impl StatelessOp for Merge {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let index =
            inputs.iter().position(|t| !t.is_null()).ok_or("No tensor received in merge")?;
        Ok(tvec!(inputs.remove(index), Tensor::from(index as i32).into()))
    }
}

impl InferenceRulesOp for Merge {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, self.n)?;
        check_output_arity(&outputs, 1)?;
        for i in 1..self.n {
            s.equals(&inputs[0].datum_type, &inputs[i].datum_type)?;
            s.equals(&inputs[0].shape, &inputs[i].shape)?;
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    inference_op_as_op!();
}
