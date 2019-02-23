use ndarray::prelude::*;
use tract_linalg::Conv;
use crate::ops::prelude::*;

#[derive(CustomDebug, Clone, new)]
pub struct Direct {
    conv: Box<Conv<f32>>,
    input_shape: TVec<usize>,
    output_shape: TVec<usize>,
    #[debug(skip)]
    packed_filters: Tensor,
}

impl Op for Direct {
    fn name(&self) -> Cow<str> {
        "ConvDirect".into()
    }
}

impl StatelessOp for Direct {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<f32>()?;
            let mut output = ArrayD::<f32>::uninitialized(&*self.output_shape);
            for n in 0..input.shape()[0] {
                let input = input.slice_axis(Axis(0), (n..=n).into());
                let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                self.conv.conv(
                    self.packed_filters.as_slice::<f32>()?.as_ptr(),
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    self.conv.n() as isize,
                    1,
                );
            }
            Ok(tvec!(output.into()))
        }
    }
}

impl InferenceRulesOp for Direct {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.input_shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.output_shape))?;
        Ok(())
    }
}
