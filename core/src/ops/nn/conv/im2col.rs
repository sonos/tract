use tract_linalg::MatMul;

use std::sync::Arc;

use crate::ops::prelude::*;
use ndarray::prelude::*;

use crate::ops::nn::Patch;

use num_traits::Zero;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub(super) struct Im2Col<T: Datum + Mul + Zero> {
    pub patch: Patch,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub ci_per_group: usize,
    pub packed_b_len: usize,
    pub mm: Arc<MatMul<T>>,
    patcher: Patcher,
}

impl<T: Datum + Mul + Zero> PartialEq for Im2Col<T> {
    fn eq(&self, other: &Im2Col<T>) -> bool {
        self.patch == other.patch
            && self.m == other.m
            && self.n == other.n
            && self.k == other.k
            && self.group == other.group
            && self.packed_b_len == other.packed_b_len
    }
}

impl<T: Datum + Mul + Zero> Im2Col<T> {
    pub fn new(
        patch: Patch,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        packed_b_len: usize,
        mm: Arc<MatMul<T>>,
    ) -> Im2Col<T> {
        let patcher = if !patch.padded && patch.input_shape.hw_rank() == 2 {
            Patcher::Valid2d
        } else {
            Patcher::Generic
        };
        Im2Col {
            patch,
            m,
            k,
            n,
            group,
            ci_per_group,
            packed_b_len,
            mm,
            patcher,
        }
    }

    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, T>) -> TractResult<Tensor> {
        let input_shape = &self.patch.input_shape;
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((self.k, self.n)) };

        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[self.mm.packed_b_len() * self.group * input_shape.n_dim()],
                self.mm.packed_b_alignment(),
            )?
        };
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                self.patcher
                    .patch(self, input, &mut mega_matrix.view_mut(), i, g);
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

#[derive(Copy, Clone, Debug)]
enum Patcher {
    Generic,
    Valid2d,
}

impl Patcher {
    fn patch<'i, T: Datum + Mul + Zero>(
        &self,
        im2col: &'i Im2Col<T>,
        input: &'i ArrayViewD<'i, T>,
        mega_matrix: &mut ArrayViewMut2<T>,
        i: usize,
        g: usize,
    ) {
        match self {
            Patcher::Valid2d => Self::valid_2d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                mega_matrix,
                i,
                g,
            ),
            Patcher::Generic => Self::generic(im2col, input, mega_matrix, i, g),
        }
    }

    fn generic<'i, T: Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayViewD<'i, T>,
        mega_matrix: &mut ArrayViewMut2<T>,
        i: usize,
        g: usize,
    ) {
        let visitor = im2col.patch.wrap(input);
        let mut coords = vec![0; im2col.patch.input_shape.rank()];
        coords[im2col.patch.input_shape.n_axis()] = i;
        for (spatial, mut col) in ndarray::indices(&*im2col.patch.output_spatial_shape)
            .into_iter()
            .zip(mega_matrix.axis_iter_mut(Axis(1)))
        {
            let mut col = col.iter_mut();
            coords[im2col.patch.input_shape.hw_axes()].copy_from_slice(spatial.slice());
            for ci in 0..im2col.ci_per_group {
                coords[im2col.patch.input_shape.c_axis()] = ci + g * im2col.ci_per_group;
                for v in visitor.at(&*coords) {
                    *col.next().expect("geometry error in conv") = v.unwrap_or(T::default());
                }
            }
        }
    }

    fn valid_2d<'i, T: Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayView4<'i, T>,
        mega_matrix: &mut ArrayViewMut2<T>,
        i: usize,
        g: usize,
    ) {
        unsafe {
            let y_stride = input.strides()[im2col.patch.input_shape.hw_axes()][0] * im2col.patch.kernel_strides[0] as isize;
            let x_stride = input.strides()[im2col.patch.input_shape.hw_axes()][1] * im2col.patch.kernel_strides[1] as isize;
            let c_stride = input.strides()[im2col.patch.input_shape.c_axis()] as isize;
            let mut optr = mega_matrix.as_mut_ptr();
            let iptr = input.slice_axis(Axis(0), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for y in 0..im2col.patch.output_spatial_shape[0] {
                        let iptr = iptr.offset(y as isize * y_stride);
                        for x in 0..im2col.patch.output_spatial_shape[1] {
                            *optr = *iptr.offset(x as isize * x_stride);
                            optr = optr.offset(1);
                        }
                    }
                }
            }
        }
    }
}
