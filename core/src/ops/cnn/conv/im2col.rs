use tract_linalg::PackB;

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

use num_traits::Zero;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub(super) struct Im2Col<T: Copy + Datum + Mul + Zero> {
    pub patch: Patch,
    pub input_shape: DataShape,
    pub output_shape: DataShape,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub ci_per_group: usize,
    pub b_pack: PackB<T>,
}

impl<T: Copy + Datum + Mul + Zero> PartialEq for Im2Col<T> {
    fn eq(&self, other: &Im2Col<T>) -> bool {
        self.patch == other.patch
            && self.m == other.m
            && self.n == other.n
            && self.k == other.k
            && self.group == other.group
            && self.b_pack == other.b_pack
    }
}

impl<T: Copy + Datum + Mul + Zero> Im2Col<T> {
    pub fn new(
        patch: Patch,
        input_shape: DataShape,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        b_pack: PackB<T>,
    ) -> Im2Col<T> {
        let output_shape = input_shape.fmt.shape(tvec!(input_shape.n_dim(), group, b_pack.len()));
        Im2Col { patch, input_shape, output_shape, m, k, n, group, ci_per_group, b_pack }
    }

    pub(super) fn output_shape(&self) -> &[usize] {
        &self.output_shape.shape
    }

    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, T>) -> TractResult<Tensor> {
        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(&*self.output_shape.shape, self.b_pack.alignment())?
        };
        for i in 0..self.input_shape.n_dim() {
            unsafe {
                let iptr = input.as_ptr().offset((self.input_shape.n_stride() * i) as isize);
                for g in 0..self.group {
                    let mut packed = packed.to_array_view_mut::<T>()?;
                    packed.slice_axis_inplace(Axis(0), (i..=i).into());
                    packed.slice_axis_inplace(Axis(1), (g..=g).into());
                    let mut writer =
                        self.b_pack.write_packed_by_rows(packed.as_slice_mut().unwrap());
                    for ci in 0..self.ci_per_group {
                        let iptr = iptr.offset(
                            (self.input_shape.c_stride() * (ci + g * self.ci_per_group)) as isize,
                        );
                        for kgeo in 0..self.patch.standard_layout_data_field.len() {
                            self.patch.visit_output_in_order(|v| {
                                if let Some(of) = v.nth_offset_if_valid(kgeo) {
                                    writer.write(*iptr.offset(of))
                                } else {
                                    writer.write(T::zero())
                                }
                            });
                        }
                    }
                }
            }
        }
        Ok(packed)
    }
}

impl<T: Copy + Datum + Mul + Zero> Op for Im2Col<T> {
    fn name(&self) -> Cow<str> {
        "Conv::Im2col".into()
    }

    impl_op_same_as!();

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("Pack: {:?}\nMatMul: {:?}", self.patch, self.b_pack)))
    }
}

impl<T: Copy + Datum + Mul + Zero> StatelessOp for Im2Col<T> {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let tensor = self.im2col(&inputs[0].to_array_view()?)?;
        Ok(tvec!(tensor.into()))
    }
}

impl<T: Copy + Datum + Mul + Zero> InferenceRulesOp for Im2Col<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.input_shape.shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.output_shape.shape))?;
        Ok(())
    }
}

