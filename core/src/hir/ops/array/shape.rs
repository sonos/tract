use crate::internal::*;
use crate::infer::*;

pub use crate::ops::array::Shape;

impl InferenceRulesOp for Shape {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 1)?;
        s.given(&inputs[0].rank, move |s, r| s.equals(&outputs[0].shape[0], r.to_dim()))?;
        s.given(&outputs[0].shape[0], move |s, r| {
            if let Ok(d) = r.to_integer() {
                s.equals(&inputs[0].rank, d)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].shape, move |s, shape| {
            if shape.iter().any(|d| d.to_integer().is_err()) {
                s.equals(&outputs[0].datum_type, DatumType::TDim)?;
                let tensor = rctensor1(&*shape);
                s.equals(&outputs[0].value, tensor)
            } else if self.dt == DatumType::I64 {
                s.equals(&outputs[0].datum_type, DatumType::I64)?;
                let tensor = rctensor1(
                    &shape.iter().map(|i| i.to_integer().unwrap() as i64).collect::<Vec<_>>(),
                );
                s.equals(&outputs[0].value, tensor)
            } else {
                s.equals(&outputs[0].datum_type, DatumType::I32)?;
                let tensor = rctensor1(
                    &shape.iter().map(|i| i.to_integer().unwrap() as i32).collect::<Vec<_>>(),
                );
                s.equals(&outputs[0].value, tensor)
            }
        })
    }

    as_op!();
    to_typed!();
}

