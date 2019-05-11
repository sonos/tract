use crate::internal::*;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use tract_linalg::Conv;

#[derive(CustomDebug, Clone, new)]
pub struct Direct {
    conv: Box<Conv<f32>>,
    input_shape: DataShape,
    output_shape: DataShape,
    #[debug(skip)]
    packed_filters: Tensor,
}

impl Op for Direct {
    fn name(&self) -> Cow<str> {
        "ConvDirect".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.conv)))
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            batch * self.conv.n() * self.conv.co() * self.conv.k()
        )))
    }

    fn rounding_errors(&self) -> bool {
        true
    }
}

impl StatelessOp for Direct {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<f32>()?;
            let mut output = ArrayD::<f32>::uninitialized(&*self.output_shape.shape);
            for n in 0..self.input_shape.n() {
                let input = input.slice_axis(Axis(0), (n..=n).into());
                let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                self.conv.conv(
                    self.packed_filters.as_slice::<f32>()?.as_ptr(),
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    self.output_shape.c_stride() as isize,
                    self.output_shape.w_stride() as isize,
                );
            }
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl InferenceRulesOp for Direct {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!()
    }
}
