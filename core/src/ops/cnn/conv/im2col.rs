use tract_linalg::frame::PackB;

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

use num_traits::Zero;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct Im2Col<T: Copy + Datum + Mul + Zero> {
    pub patch: Patch,
    pub input_shape: DataShape,
    pub output_shape: DataShape,
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
        input_shape: DataShape,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        b_pack: PackB<T>,
    ) -> Im2Col<T> {
        let patcher = if !patch.padded && patch.rank() == 2 {
            Patcher::Valid2d
        } else if patch.rank() == 2 {
            Patcher::Padded2d
        } else if !patch.padded && patch.rank() == 1 {
            Patcher::Valid1d
        } else {
            Patcher::Generic
        };
        let output_shape = input_shape.fmt.shape(tvec!(*input_shape.n_dim(), group, b_pack.len()));
        Im2Col { patch, input_shape, output_shape, m, k, n, group, ci_per_group, b_pack, patcher }
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape.shape
    }

    pub(super) fn im2col<'i>(&'i self, input: &'i ArrayViewD<'i, T>) -> TractResult<Tensor> {
        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(&*self.output_shape.shape, self.b_pack.alignment())?
        };
        for i in 0..*self.input_shape.n_dim() {
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
        "Conv::Im2col".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "MatMul: (m,k,n):{:?} groups:{} {:?}",
            (self.m, self.k, self.n),
            self.group,
            self.b_pack
        )])
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl<T: Copy + Datum + Mul + Zero> StatelessOp for Im2Col<T> {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let tensor = self.im2col(&inputs[0].to_array_view()?)?;
        Ok(tvec!(tensor.into()))
    }
}

impl<T: Copy + Datum + Mul + Zero> TypedOp for Im2Col<T> {
    typed_op_as_op!();

    fn output_facts(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(T::datum_type(), &*self.output_shape.shape)?))
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
        let ptr = input.as_ptr();
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((im2col.k, im2col.n)) };
        let shape = &im2col.input_shape;
        unsafe {
            let ptr = ptr.offset((shape.n_stride() * i) as isize);
            let ptr = ptr.offset((shape.c_stride() * (g * im2col.ci_per_group)) as isize);
            for (spatial, mut col) in ndarray::indices(&*im2col.patch.output_shape)
                .into_iter()
                .zip(mega_matrix.axis_iter_mut(Axis(1)))
            {
                let mut col = col.iter_mut();
                for ci in 0..im2col.ci_per_group {
                    let ptr = ptr.offset((shape.c_stride() * ci) as isize);
                    for v in im2col.patch.at(spatial.slice()) {
                        *col.next().expect("geometry error in conv") =
                            v.map(|o| *ptr.offset(o)).unwrap_or(T::default());
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
            let x_stride = input.strides()[im2col.input_shape.h_axis()]
                * im2col.patch.spec.strides[0] as isize;
            let c_stride = input.strides()[im2col.input_shape.c_axis()] as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.slice_axis(Axis(im2col.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for x in 0..*im2col.patch.output_shape.get_unchecked(0) {
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
            let y_stride_ptr = y_stride * input.strides()[im2col.input_shape.hw_axes()][0];
            let x_stride_ptr = x_stride * input.strides()[im2col.input_shape.hw_axes()][1];
            let c_stride_ptr = input.strides()[im2col.input_shape.c_axis()] as isize;
            let input_heigth = im2col.input_shape.hw_dims()[0] as isize;
            let input_width = im2col.input_shape.hw_dims()[1] as isize;
            let kernel_len = im2col.patch.standard_layout_data_field.len();
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.slice_axis(Axis(im2col.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for kitem in 0..kernel_len {
                    let dy = *im2col.patch.data_field.as_ptr().offset(kitem as isize * 2);
                    let dx = *im2col.patch.data_field.as_ptr().offset(1 + kitem as isize * 2);
                    let iptr =
                        iptr.offset(*im2col.patch.standard_layout_data_field.get_unchecked(kitem));
                    for yo in 0..*im2col.patch.output_shape.get_unchecked(0) {
                        let y = yo as isize * y_stride + dy;
                        let iptr = iptr.offset(yo as isize * y_stride_ptr);
                        if y >= 0 && y < input_heigth {
                            for xo in 0..*im2col.patch.output_shape.get_unchecked(1) {
                                let x = xo as isize * x_stride + dx;
                                if x >= 0 && x < input_width {
                                    writer.write(*iptr.offset(xo as isize * x_stride_ptr));
                                } else {
                                    writer.write(T::default());
                                }
                            }
                        } else {
                            for _x in 0..*im2col.patch.output_shape.get_unchecked(1) {
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
            let y_stride = input.strides()[im2col.input_shape.hw_axes()][0]
                * im2col.patch.spec.strides[0] as isize;
            let x_stride = input.strides()[im2col.input_shape.hw_axes()][1]
                * im2col.patch.spec.strides[1] as isize;
            let c_stride = input.strides()[im2col.input_shape.c_axis()] as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.slice_axis(Axis(im2col.input_shape.n_axis()), (i..=i).into()).as_ptr();
            for ci in (im2col.ci_per_group * g)..(im2col.ci_per_group * (g + 1)) {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for y in 0..*im2col.patch.output_shape.get_unchecked(0) {
                        let iptr = iptr.offset(y as isize * y_stride);
                        for x in 0..*im2col.patch.output_shape.get_unchecked(1) {
                            writer.write(*iptr.offset(x as isize * x_stride));
                        }
                    }
                }
            }
        }
    }
}
