use tract_linalg::PackB;

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::nn::Patch;

use num_traits::Zero;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub(super) struct Im2Col<T: Copy + Datum + Mul + Zero> {
    pub patch: Patch,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub ci_per_group: usize,
    pub b_pack: PackB<T>,
    patcher: Patcher,
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
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        b_pack: PackB<T>,
    ) -> Im2Col<T> {
        let patcher = if !patch.padded && patch.input_shape.hw_rank() == 2 {
            Patcher::Valid2d
        } else if patch.input_shape.hw_rank() == 2 {
            Patcher::Padded2d
        } else if !patch.padded && patch.input_shape.hw_rank() == 1 {
            Patcher::Valid1d
        } else {
            Patcher::Generic
        };
        Im2Col { patch, m, k, n, group, ci_per_group, b_pack, patcher }
    }

    pub(super) fn output_shape(&self) -> TractResult<TVec<usize>> {
        let input_shape = &self.patch.input_shape;
        Ok(tvec!(input_shape.n_dim(), self.group, self.b_pack.len()))
    }

    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, T>) -> TractResult<Tensor> {
        let input_shape = &self.patch.input_shape;

        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[input_shape.n_dim(), self.group, self.b_pack.len()],
                self.b_pack.alignment(),
            )?
        };
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mut packed = packed.to_array_view_mut::<T>()?;
                packed.slice_axis_inplace(Axis(0), (i..=i).into());
                packed.slice_axis_inplace(Axis(1), (g..=g).into());
                self.patcher.patch(self, input, packed.as_slice_mut().unwrap(), i, g);
            }
        }
        Ok(packed)
    }
}

impl<T: Copy + Datum + Mul + Zero> Op for Im2Col<T> {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
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
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.patch.input_shape.shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&[self.b_pack.len() * self.group]))?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
enum Patcher {
    Generic,
    Valid1d,
    Valid2d,
    Padded2d,
}

impl Patcher {
    fn patch<'i, 'p, T: Copy + Datum + Mul + Zero>(
        &self,
        im2col: &'i Im2Col<T>,
        input: &'i ArrayViewD<'i, T>,
        pack: &'p mut [T],
        i: usize,
        g: usize,
    ) {
        match self {
            Patcher::Valid1d => Self::valid_1d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                i,
                g,
            ),
            Patcher::Valid2d => Self::valid_2d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                i,
                g,
            ),
            Patcher::Padded2d => Self::padded_2d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                i,
                g,
            ),
            _ => Self::generic(im2col, input, pack, i, g),
        }
    }

    #[inline(never)]
    fn generic<'i, 'p, T: Copy + Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayViewD<'i, T>,
        pack: &'p mut [T],
        i: usize,
        g: usize,
    ) {
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((im2col.k, im2col.n)) };
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
        im2col.b_pack.pack(
            pack.as_mut_ptr(),
            mega_matrix.as_ptr(),
            mega_matrix.strides()[0],
            mega_matrix.strides()[1],
        );
    }

    #[inline(never)]
    fn valid_1d<'i, 'p, T: Copy + Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayView3<'i, T>,
        pack: &'p mut [T],
        i: usize,
        g: usize,
    ) {
        unsafe {
            let x_stride = input.strides()[im2col.patch.input_shape.h_axis()]
                * im2col.patch.spec.strides[0] as isize;
            let c_stride = input.strides()[im2col.patch.input_shape.c_axis()] as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr =
                input.slice_axis(Axis(im2col.patch.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for x in 0..*im2col.patch.output_spatial_shape.get_unchecked(0) {
                        writer.write(*iptr.offset(x as isize * x_stride));
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn padded_2d<'i, 'p, T: Copy + Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayView4<'i, T>,
        pack: &'p mut [T],
        i: usize,
        g: usize,
    ) {
        unsafe {
            let y_stride = im2col.patch.spec.strides[0] as isize;
            let x_stride = im2col.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * input.strides()[im2col.patch.input_shape.hw_axes()][0];
            let x_stride_ptr = x_stride * input.strides()[im2col.patch.input_shape.hw_axes()][1];
            let c_stride_ptr = input.strides()[im2col.patch.input_shape.c_axis()] as isize;
            let input_heigth = im2col.patch.input_shape.hw_dims()[0] as isize;
            let input_width = im2col.patch.input_shape.hw_dims()[1] as isize;
            let kernel_len = im2col.patch.standard_layout_data_field.len();
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr =
                input.slice_axis(Axis(im2col.patch.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for kitem in 0..kernel_len {
                    let dy = *im2col.patch.data_field.as_ptr().offset(kitem as isize * 2);
                    let dx = *im2col.patch.data_field.as_ptr().offset(1 + kitem as isize * 2);
                    let iptr =
                        iptr.offset(*im2col.patch.standard_layout_data_field.get_unchecked(kitem));
                    for yo in 0..*im2col.patch.output_spatial_shape.get_unchecked(0) {
                        let y = yo as isize * y_stride + dy;
                        let iptr = iptr.offset(yo as isize * y_stride_ptr);
                        if y >= 0 && y < input_heigth {
                            for xo in 0..*im2col.patch.output_spatial_shape.get_unchecked(1) {
                                let x = xo as isize * x_stride + dx;
                                if x >= 0 && x < input_width {
                                    writer.write(*iptr.offset(xo as isize * x_stride_ptr));
                                } else {
                                    writer.write(T::default());
                                }
                            }
                        } else {
                            for _x in 0..*im2col.patch.output_spatial_shape.get_unchecked(1) {
                                writer.write(T::default());
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn valid_2d<'i, 'p, T: Copy + Datum + Mul + Zero>(
        im2col: &'i Im2Col<T>,
        input: &'i ArrayView4<'i, T>,
        pack: &'p mut [T],
        i: usize,
        g: usize,
    ) {
        unsafe {
            let y_stride = input.strides()[im2col.patch.input_shape.hw_axes()][0]
                * im2col.patch.spec.strides[0] as isize;
            let x_stride = input.strides()[im2col.patch.input_shape.hw_axes()][1]
                * im2col.patch.spec.strides[1] as isize;
            let c_stride = input.strides()[im2col.patch.input_shape.c_axis()] as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr =
                input.slice_axis(Axis(im2col.patch.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for y in 0..*im2col.patch.output_spatial_shape.get_unchecked(0) {
                        let iptr = iptr.offset(y as isize * y_stride);
                        for x in 0..*im2col.patch.output_spatial_shape.get_unchecked(1) {
                            writer.write(*iptr.offset(x as isize * x_stride));
                        }
                    }
                }
            }
        }
    }
}
