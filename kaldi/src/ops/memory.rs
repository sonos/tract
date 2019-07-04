use tract_core::internal::*;
use tract_core::ndarray;

use crate::model::ParsingContext;

#[derive(Clone, Debug, new)]
pub struct Memory {
    pub name: String,
    pub offset: isize
}

impl Op for Memory {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Memory".into()
    }
}

impl StatefullOp for Memory {
    fn state(&self, session: &mut SessionState, id:usize) -> TractResult<Option<Box<OpState>>> {
        unimplemented!()
    }
}

impl InferenceRulesOp for Memory {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        Ok(())

    }

    inference_op_as_op!();
}
