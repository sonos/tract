use tract_linalg::frame::PackB;

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::Patch;
use crate::ops::nn::DataFormat;

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct Im2Col {
    pub patch: Patch,
    pub data_format: DataFormat,
    pub data_format_with_n: DataFormat,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub ci_per_group: usize,
    pub b_pack: PackB,
    patcher: Patcher,
    pad_value: Tensor,
}

impl DynHash for Im2Col {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        dyn_hash(self, state)
    }
}

impl PartialEq for Im2Col {
    fn eq(&self, other: &Im2Col) -> bool {
        self.patch == other.patch
            && self.m == other.m
            && self.n == other.n
            && self.k == other.k
            && self.group == other.group
            && self.b_pack == other.b_pack
            && self.pad_value == other.pad_value
    }
}

impl Im2Col {
    pub fn new(
        patch: Patch,
        data_format: DataFormat,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        b_pack: PackB,
        pad_value: Tensor,
    ) -> TractResult<Im2Col> {
        let patcher = if !patch.padded && patch.rank() == 2 {
            Patcher::Valid2d
        } else if patch.rank() == 2 {
            Patcher::Padded2d
        } else if !patch.padded && patch.rank() == 1 {
            Patcher::Valid1d
        } else {
            Patcher::Generic
        };
        let data_format_with_n = match data_format {
            DataFormat::HWC => DataFormat::NHWC,
            DataFormat::CHW => DataFormat::NCHW,
            any => any,
        };
        Ok(Im2Col {
            patch,
            data_format,
            data_format_with_n,
            m,
            k,
            n,
            group,
            ci_per_group,
            b_pack,
            patcher,
            pad_value,
        })
    }

    fn output_shape<D: DimLike>(&self, input_shape: &[D]) -> TractResult<TVec<D>> {
        let mut output_shape: TVec<D> = tvec!();
        if self.data_format.has_n() {
            output_shape.push(self.data_format.shape(input_shape)?.n().unwrap().clone());
        }
        output_shape.push(self.group.into());
        output_shape.push(self.b_pack.len().into());
        Ok(output_shape)
    }

    pub(super) unsafe fn im2col<'i, T: Copy + Datum>(
        &'i self,
        input: &'i Tensor,
        packed: &mut Tensor,
    ) {
        if input.len() == 0 {
            return;
        }
        let pad_value = *self.pad_value.to_scalar_unchecked();
        let mut input = input.to_array_view_unchecked::<T>();
        let mut packed = packed.to_array_view_mut_unchecked::<T>();
        if !self.data_format.has_n() {
            input.insert_axis_inplace(Axis(0));
            packed.insert_axis_inplace(Axis(0));
        }
        for n in 0..*self.data_format_with_n.shape(&input.shape()).unwrap().n().unwrap() {
            let mut input = input.view();
            let mut packed = packed.view_mut();
            input.slice_axis_inplace(Axis(0), (n..=n).into());
            packed.index_axis_inplace(Axis(0), n);
            for g in 0..self.group {
                let mut packed = packed.index_axis_mut(Axis(0), g);
                self.patcher.patch(self, &input, packed.as_slice_mut().unwrap(), g, pad_value);
            }
        }
    }
}

impl Op for Im2Col {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "MatMul: (m,k,n):{:?} groups:{} {:?}",
            (self.m, self.k, self.n),
            self.group,
            self.b_pack
        )])
    }

    op_core_lir!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Im2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let mut tensor = Tensor::uninitialized_aligned_dt(
                inputs[0].datum_type(),
                &*self.output_shape(&*inputs[0].shape())?,
                self.b_pack.alignment(),
            )?;
            dispatch_copy_by_size!(Self::im2col(inputs[0].datum_type())(
                self,
                &inputs[0],
                &mut tensor
            ));
            Ok(tvec!(tensor.into()))
        }
    }
}

impl TypedOp for Im2Col {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            &*self.output_shape(&*inputs[0].shape)?
        )?))
    }
}

#[derive(Copy, Clone, Debug, Hash)]
enum Patcher {
    Generic,
    Valid1d,
    Valid2d,
    Padded2d,
}

impl Patcher {
    fn patch<'i, 'p, T: Copy + Datum>(
        &self,
        im2col: &'i Im2Col,
        input: &'i ArrayViewD<'i, T>,
        pack: &'p mut [T],
        g: usize,
        pad_value: T,
    ) {
        match self {
            Patcher::Valid1d => Self::valid_1d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                g,
            ),
            Patcher::Valid2d => Self::valid_2d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                g,
            ),
            Patcher::Padded2d => Self::padded_2d(
                im2col,
                input.view().into_dimensionality().as_ref().unwrap(),
                pack,
                g,
                pad_value,
            ),
            _ => Self::generic(im2col, input, pack, g, pad_value),
        }
    }

    #[inline(never)]
    fn generic<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i ArrayViewD<'i, T>,
        pack: &'p mut [T],
        g: usize,
        pad_value: T,
    ) {
        let mut mega_matrix = unsafe { Array2::<T>::uninitialized((im2col.k, im2col.n)) };
        let input_shape: TVec<usize> = input.shape().into();
        let shape = im2col.data_format_with_n.shape(&input_shape).unwrap();
        unsafe {
            let ptr = input.as_ptr().offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
            for (spatial, mut col) in ndarray::indices(&*im2col.patch.output_shape)
                .into_iter()
                .zip(mega_matrix.axis_iter_mut(Axis(1)))
            {
                let mut col = col.iter_mut();
                for ci in 0..im2col.ci_per_group {
                    let ptr = ptr.offset((shape.c_stride() * ci) as isize);
                    for v in im2col.patch.at(spatial.slice()) {
                        *col.next().expect("geometry error in conv") =
                            v.map(|o| *ptr.offset(o)).unwrap_or(pad_value);
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
    fn valid_1d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i ArrayView3<'i, T>,
        pack: &'p mut [T],
        g: usize,
    ) {
        unsafe {
            let input_shape: TVec<usize> = input.shape().into();
            let shape = im2col.data_format_with_n.shape(&input_shape).unwrap();
            let x_stride = *shape.h_stride() as isize * im2col.patch.spec.strides[0] as isize;
            let c_stride = *shape.c_stride() as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.as_ptr().offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..im2col.ci_per_group {
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
    fn padded_2d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i ArrayView4<'i, T>,
        pack: &'p mut [T],
        g: usize,
        pad_value: T,
    ) {
        unsafe {
            let input_shape: TVec<usize> = input.shape().into();
            let shape = im2col.data_format_with_n.shape(&input_shape).unwrap();
            let y_stride = im2col.patch.spec.strides[0] as isize;
            let x_stride = im2col.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let input_heigth = shape.hw_dims()[0] as isize;
            let input_width = shape.hw_dims()[1] as isize;
            let kernel_len = im2col.patch.standard_layout_data_field.len();
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.as_ptr().offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..im2col.ci_per_group {
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
                                    writer.write(pad_value);
                                }
                            }
                        } else {
                            for _x in 0..*im2col.patch.output_shape.get_unchecked(1) {
                                writer.write(pad_value);
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn valid_2d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i ArrayView4<'i, T>,
        pack: &'p mut [T],
        g: usize,
    ) {
        unsafe {
            let input_shape: TVec<usize> = input.shape().into();
            let shape = im2col.data_format_with_n.shape(&input_shape).unwrap();
            let y_stride = im2col.patch.spec.strides[0] as isize;
            let x_stride = im2col.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let mut writer = im2col.b_pack.write_packed_by_rows(pack);
            let iptr = input.as_ptr().offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..im2col.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for y in 0..*im2col.patch.output_shape.get_unchecked(0) {
                        let iptr = iptr.offset(y as isize * y_stride_ptr);
                        for x in 0..*im2col.patch.output_shape.get_unchecked(1) {
                            writer.write(*iptr.offset(x as isize * x_stride_ptr));
                        }
                    }
                }
            }
        }
    }
}
