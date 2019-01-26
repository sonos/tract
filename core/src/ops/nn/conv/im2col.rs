use tract_linalg::MatMul;

use std::sync::Arc;

use crate::ops::prelude::*;
use ndarray::prelude::*;

use crate::ops::nn::Patch;
use crate::ops::nn::PatchVisitor;

use num_traits::Zero;
use std::ops::Mul;

#[derive(Debug, Clone, new)]
pub(super) struct Im2Col<T: Datum + Mul + Zero> {
    pub patch: Patch,
    pub group: usize,
    pub ci_per_group: usize,
    pub packed_b_len: usize,
    pub mm: Arc<MatMul<T>>,
}

impl<T: Datum + Mul + Zero> PartialEq for Im2Col<T> {
    fn eq(&self, other: &Im2Col<T>) -> bool {
        self.patch == other.patch
            && self.mm.m() == other.mm.m()
            && self.mm.n() == other.mm.n()
            && self.mm.k() == other.mm.k()
            && self.group == other.group
            && self.packed_b_len == other.packed_b_len
    }
}

impl<T: Datum + Mul + Zero> Im2Col<T> {
    fn im2col_gen<'i, 'p>(
        &'i self,
        visitor: &'p PatchVisitor<'i, 'p, T>,
        mega_matrix: &mut ArrayViewMut2<T>,
        packed: &mut Tensor,
        n: usize,
        g: usize,
    ) -> TractResult<()> {
        let mut coords = tvec![0; self.patch.input_shape.rank()];
        coords[self.patch.input_shape.n_axis()] = n;
        unsafe {
            for (col, spatial) in ndarray::indices(&*self.patch.output_spatial_shape)
                .into_iter()
                .enumerate()
            {
                let mut mmptr = mega_matrix.as_mut_ptr().offset(col as isize);
                coords
                    .get_unchecked_mut(self.patch.input_shape.hw_axes())
                    .copy_from_slice(spatial.slice());
                for ci in 0..self.ci_per_group {
                    *coords.get_unchecked_mut(self.patch.input_shape.c_axis()) = ci + g * self.ci_per_group;
                    for v in visitor.at(&*coords) {
                        *mmptr = v.unwrap_or(T::default());
                        mmptr = mmptr.offset(self.mm.n() as isize);
                    }
                }
            }
            self.mm.pack_b(
                packed
                    .as_slice_mut::<T>()?
                    .as_mut_ptr()
                    .offset(((n * self.group + g) * self.packed_b_len) as isize),
                mega_matrix.as_ptr(),
                1,
                self.mm.k() as isize,
            );
        }
        Ok(())
    }

    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, T>) -> TractResult<Tensor> {
        let input_shape = &self.patch.input_shape;
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((self.mm.n(), self.mm.k())) };
        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[self.mm.packed_b_len() * self.group * input_shape.n_dim()],
                self.mm.packed_b_alignment(),
            )?
        };
        let mut visitor = self.patch.wrap(input);
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                self.im2col_gen(&mut visitor, &mut mega_matrix.view_mut(), &mut packed, i, g)?;
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

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!(
            "Pack: {:?}\nMatMul: {:?}",
            self.patch, self.mm
        )))
    }
}

impl<T: Datum + Mul + Zero> StatelessOp for Im2Col<T> {
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
