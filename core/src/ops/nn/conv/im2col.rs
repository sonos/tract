use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::Patch;

#[derive(Debug, Clone, new, PartialEq)]
pub(super) struct Im2Col<D: Datum> {
    pub patch: Patch,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub _phantom: PhantomData<D>,
}

impl<D: Datum> Im2Col<D> {
    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, D>) -> TractResult<Array2<D>> {
        let input_shape = &self.patch.input_shape;
        let mut mega_matrix = unsafe {
            Array2::<D>::uninitialized((self.k, self.n * input_shape.n_dim() * self.group))
        };
        let visitor = self.patch.wrap(input);
        let ci_per_group = input_shape.c_dim() / self.group;
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mm_offset = self.n * (g + (i * self.group));
                let mut coords = vec![0; input_shape.rank()];
                coords[input_shape.n_axis()] = i;
                for (mut spatial, mut col) in ndarray::indices(&*self.patch.output_spatial_shape)
                    .into_iter()
                    .zip(
                        mega_matrix
                            .slice_axis_mut(Axis(1), (mm_offset..(mm_offset + self.n)).into())
                            .axis_iter_mut(Axis(1)),
                    )
                {
                    let mut col = col.iter_mut();
                    coords[input_shape.h_axis()..][..input_shape.hw_rank()]
                        .copy_from_slice(spatial.slice());
                    for ci in 0..ci_per_group {
                        coords[input_shape.c_axis()] = ci + g * ci_per_group;
                        for v in visitor.at(&*coords) {
                            *col.next().expect("geometry error in conv") =
                                v.unwrap_or(D::default());
                        }
                    }
                }
            }
        }
        trace!("im2col: {:?}", mega_matrix);
        Ok(mega_matrix)
    }
}

impl<D: Datum> Op for Im2Col<D> {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    impl_op_same_as!();
}

impl<D: Datum> StatelessOp for Im2Col<D> {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let output = self.im2col(&inputs[0].to_array_view::<D>()?)?;
        Ok(tvec!(output.into()))
    }
}

impl<D: Datum> InferenceRulesOp for Im2Col<D> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].datum_type, D::datum_type())?;
        s.equals(
            &inputs[0].shape,
            ShapeFact::from(&*self.patch.input_shape.shape),
        )?;
        s.equals(
            &outputs[0].shape,
            ShapeFact::from(&[self.k, self.n * self.patch.input_shape.n_dim() * self.group]),
        )?;
        Ok(())
    }
}
