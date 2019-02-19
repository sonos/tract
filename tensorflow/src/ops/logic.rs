use tract_core::ops as tractops;
use tract_core::ops::prelude::*;

use crate::ops::OpRegister;
use crate::tfpb::node_def::NodeDef;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Less", with_T!(tractops::logic::Lesser::Bin));
    reg.insert("Merge", merge);
    reg.insert("Switch", |_| Ok(Box::new(Switch)));
}

#[derive(Debug, Clone)]
pub struct Switch;

impl Op for Switch {
    fn name(&self) -> Cow<str> {
        "tf.Switch".into()
    }

    fn noutputs(&self) -> usize {
        2
    }
}

impl StatelessOp for Switch {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&inputs[1].datum_type, DatumType::Bool)?;
        s.equals(&inputs[1].shape, shapefact!())?;
        s.given(&outputs.len, move |s, len| {
            for i in 0..(len as usize) {
                s.equals(&inputs[0].datum_type, &outputs[i].datum_type)?;
                s.equals(&inputs[0].shape, &outputs[i].shape)?;
            }
            Ok(())
        })
    }
}

fn merge(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
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

    fn noutputs(&self) -> usize {
        2
    }
}

impl StatelessOp for Merge {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let index = inputs
            .iter()
            .position(|t| !t.is_null())
            .ok_or("No tensor received in merge")?;
        Ok(tvec!(
            inputs.remove(index),
            Tensor::from(index as i32).into()
        ))
    }
}

impl InferenceRulesOp for Merge {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, self.n as i32)?;
        s.equals(&outputs.len, 1)?;
        for i in 1..self.n {
            s.equals(&inputs[0].datum_type, &inputs[i].datum_type)?;
            s.equals(&inputs[0].shape, &inputs[i].shape)?;
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}
