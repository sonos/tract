use tract_linalg::MatMul;

use std::sync::Arc;

use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::Patch;

use num::Zero;
use std::ops::Mul;

#[derive(Debug, Clone, new)]
pub(super) struct Im2Col<T: Datum + Mul + Zero> {
    pub patch: Patch,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub packed_b_len: usize,
    pub mm: Arc<MatMul<T>>,
}

impl<T: Datum + Mul + Zero> PartialEq for Im2Col<T> {
    fn eq(&self, other: &Im2Col<T>) -> bool {
        self.patch == other.patch && self.m == other.m && self.n == other.n && self.k == other.k && self.group == other.group && self.packed_b_len == other.packed_b_len
    }
}

impl<T: Datum + Mul + Zero> Im2Col<T> {
    pub(super) fn im2col<'i>(
        &'i self,
        input: &'i ArrayViewD<'i, T>,
    ) -> TractResult<Tensor> {
        let input_shape = &self.patch.input_shape;
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((self.k, self.n)) };

        let mut packed = unsafe { Tensor::uninitialized_aligned::<T>(&[self.mm.packed_b_len() * self.group * input_shape.n_dim()], self.mm.packed_b_alignment())? };
        let visitor = self.patch.wrap(input);
        let ci_per_group = input_shape.c_dim() / self.group;
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mut coords = vec![0; input_shape.rank()];
                coords[input_shape.n_axis()] = i;
                for (mut spatial, mut col) in ndarray::indices(&*self.patch.output_spatial_shape)
                    .into_iter()
                    .zip(mega_matrix.axis_iter_mut(Axis(1)))
                {
                    let mut col = col.iter_mut();
                    coords[input_shape.h_axis()..][..input_shape.hw_rank()]
                        .copy_from_slice(spatial.slice());
                    for ci in 0..ci_per_group {
                        coords[input_shape.c_axis()] = ci + g * ci_per_group;
                        for v in visitor.at(&*coords) {
                            *col.next().expect("geometry error in conv") =
                                v.unwrap_or(T::default());
                        }
                    }
                }
                unsafe {
                    self.mm.pack_b(
                        packed
                            .as_slice_mut::<T>()?
                            .as_mut_ptr()
                            .offset(((i * self.group + g) * self.packed_b_len) as isize),
                        mega_matrix.as_ptr(),
                        mega_matrix.strides()[0],
                        mega_matrix.strides()[1],
                    );
                }
            }
        }
        Ok(packed)
    }
}

impl<T: Datum + Mul + Zero> Op for Im2Col<T> {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    impl_op_same_as!();
}

impl<T: Datum+Mul+Zero> StatelessOp for Im2Col<T> {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let tensor = self.im2col(&inputs[0].to_array_view()?)?;
        Ok(tvec!(tensor.into()))
    }
}

impl<T: Datum + Mul + Zero> InferenceRulesOp for Im2Col<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(
            &inputs[0].shape,
            ShapeFact::from(&*self.patch.input_shape.shape),
        )?;
        s.equals(
            &outputs[0].shape,
            ShapeFact::from(&[self.packed_b_len * self.group]),
        )?;
        Ok(())
    }
}
