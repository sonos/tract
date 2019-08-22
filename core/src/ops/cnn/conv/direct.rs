use crate::internal::*;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use tract_linalg::Tile;

#[derive(CustomDebug, Clone, new)]
pub struct Direct {
    tile: Box<dyn Tile<f32>>,
    data_offsets: Vec<isize>,
    kernel_offsets: Vec<isize>,
    input_shape: DataShape,
    output_shape: DataShape,
    packed_filters: Tensor,
}

impl Direct {
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape.shape
    }
}

impl Op for Direct {
    fn name(&self) -> Cow<str> {
        "ConvDirect".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec!(format!("{:?}", self.tile)))
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            batch * self.tile.n() * self.tile.m() * self.tile.k()
        )))
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    to_typed!();
}

impl StatelessOp for Direct {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<f32>()?;
            let mut output = ArrayD::<f32>::uninitialized(&*self.output_shape.shape);
            let filters = self.packed_filters.as_ptr::<f32>()?;
            for n in 0..*self.input_shape.n() {
                let input = input.slice_axis(Axis(0), (n..=n).into());
                let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                self.tile.run(
                    &self.tile.a_from_packed(filters),
                    &self.tile.b_from_data_and_offsets(
                        input.as_ptr(),
                        &self.kernel_offsets,
                        &self.data_offsets,
                    ),
                    &mut self.tile.c_from_data_and_strides(
                        output.as_mut_ptr(),
                        *self.output_shape.c_stride() as isize,
                        *self.output_shape.w_stride() as isize,
                    ),
                );
            }
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl TypedOp for Direct {
    typed_op_as_op!();

    fn output_facts(
        &self,
        inputs: TVec<&NormalizedTensorInfo>,
    ) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(NormalizedTensorInfo::dt_shape(
            inputs[0].datum_type,
            &*self.output_shape.shape
        )?))
    }
}
