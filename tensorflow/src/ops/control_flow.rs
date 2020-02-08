use tract_core::internal::*;
use tract_core::infer::*;

use crate::model::TfOpRegister;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Enter", |_, node| {
        Ok(Box::new(LoopGate(LoopGateRole::Enter(node.get_attr_str("frame_name")?))))
    });
    reg.insert("Exit", |_, _| Ok(Box::new(LoopGate(LoopGateRole::Exit))));
    reg.insert("LoopCond", |_, _| Ok(Box::new(LoopGate(LoopGateRole::LoopCond))));
}

#[derive(Debug, Clone)]
pub enum LoopGateRole {
    Enter(String),
    Exit,
    LoopCond,
}

#[derive(Debug, Clone)]
pub struct LoopGate(LoopGateRole);

impl Op for LoopGate {
    fn name(&self) -> Cow<str> {
        format!("tf.{:?}", self.0).into()
    }

    not_a_typed_op!();
}

impl StatelessOp for LoopGate {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(inputs)
    }
}

impl InferenceRulesOp for LoopGate {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    as_op!();
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NextIterationRole {
    Source,
    Sink,
}

#[derive(Debug, Clone, new)]
pub struct NextIteration {
    name: String,
    role: NextIterationRole,
}

impl Op for NextIteration {
    fn name(&self) -> Cow<str> {
        format!("{:?}({})", self.role, self.name).into()
    }

    not_a_typed_op!();
}

impl StatefullOp for NextIteration {
    fn state(
        &self,
        _state: &mut SessionState,
        _id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        unimplemented!();
    }
}

impl InferenceRulesOp for NextIteration {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        match self.role {
            NextIterationRole::Source => {
                check_input_arity(&inputs, 0)?;
                check_output_arity(&outputs, 1)?;
            }
            NextIterationRole::Sink => {
                check_input_arity(&inputs, 1)?;
                check_output_arity(&outputs, 0)?;
            }
        }
        Ok(())
    }

    as_op!();
}
