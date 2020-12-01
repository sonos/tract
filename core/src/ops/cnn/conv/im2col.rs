use tract_linalg::frame::Packer;

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::Patch;
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct Im2Col {
    pub patch: Patch,
    pub data_format: DataFormat,
    pub data_format_with_n: DataFormat,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub ci_per_group: usize,
    pub b_pack: Packer,
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
        k: usize,
        n: usize,
        group: usize,
        ci_per_group: usize,
        b_pack: Packer,
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
        if self.group != 1 {
            output_shape.push(self.group.into());
        }
        output_shape.push(self.b_pack.len(self.n).into());
        Ok(output_shape)
    }
}

impl Op for Im2Col {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("k:{} n:{} groups:{} {:?}", self.k, self.n, self.group, self.b_pack)])
    }

    op_core_lir!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Im2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let mut input = args_1!(inputs).into_tensor();
            let output_shape = self.output_shape(input.shape())?;
            let mut output = Tensor::uninitialized_aligned_dt(
                input.datum_type(),
                &*output_shape,
                self.b_pack.alignment(),
            )?;
            if !self.data_format.has_n() {
                input.insert_axis(0)?;
                output.insert_axis(0)?;
            }
            if self.group == 1 {
                output.insert_axis(1)?;
            }
            let input_shape = self.data_format_with_n.shape(input.shape().into())?;
            // in the loop, we have normalized the input so that N is
            // always here, and output so that N and G are there.
            if !output_shape.iter().any(|d| *d == 0) {
                for i in 0..*input_shape.n().unwrap_or(&1) {
                    let input = input.view_at_prefix(&[i])?;
                    for g in 0..self.group {
                        let full_prefix = [i, g];
                        let actual_prefix = &full_prefix[..=(self.group > 1) as usize];
                        let mut packed = output.view_at_prefix_mut(actual_prefix)?;
                        dispatch_copy_by_size!(Patcher::patch(input.datum_type())(
                            &self.patcher,
                            self,
                            &input,
                            &input_shape,
                            &mut packed,
                            g
                        ))?
                    }
                }
            }
            if self.group == 1 {
                output.remove_axis(1)?;
            }
            if !self.data_format.has_n() {
                output.remove_axis(0)?;
            }
            Ok(tvec!(output.into()))
        }
    }
}

impl TypedOp for Im2Col {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, self.output_shape(&*inputs[0].shape)?)))
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
        input: &'i TensorView,
        input_shape: &DataShape,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        match self {
            Patcher::Valid1d => Self::valid_1d::<T>(im2col, input, input_shape, pack, g),
            Patcher::Valid2d => Self::valid_2d::<T>(im2col, input, input_shape, pack, g),
            Patcher::Padded2d => Self::padded_2d::<T>(im2col, input, input_shape, pack, g),
            _ => Self::generic::<T>(im2col, input, input_shape, pack, g),
        }
    }

    #[inline(never)]
    fn generic<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i TensorView,
        shape: &DataShape,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *im2col.pad_value.to_scalar_unchecked::<T>();
            let mut mega_matrix = Tensor::uninitialized::<T>(&[im2col.k, im2col.n])?;
            let mut mega_matrix_view = mega_matrix.to_array_view_mut_unchecked::<T>();
            let ptr = input.as_ptr_unchecked::<T>();
            let ptr = ptr.offset((shape.c_stride() * (g * im2col.ci_per_group)) as isize);
            for (spatial, mut col) in ndarray::indices(&*im2col.patch.output_shape)
                .into_iter()
                .zip(mega_matrix_view.axis_iter_mut(Axis(1)))
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
            im2col.b_pack.pack(pack, mega_matrix.view(), 0, 1);
            Ok(())
        }
    }

    #[inline(never)]
    fn valid_1d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i TensorView,
        shape: &DataShape,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let x_stride = *shape.h_stride() as isize * im2col.patch.spec.strides[0] as isize;
            let c_stride = *shape.c_stride() as isize;
            let pack = pack.as_slice_mut_unchecked::<T>();
            let mut writer = im2col.b_pack.write_with_k_outer(pack, im2col.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..im2col.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &im2col.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for x in 0..*im2col.patch.output_shape.get_unchecked(0) {
                        writer.write(*iptr.offset(x as isize * x_stride));
                    }
                }
            }
            Ok(())
        }
    }

    #[inline(never)]
    fn padded_2d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i TensorView,
        shape: &DataShape,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *im2col.pad_value.to_scalar_unchecked::<T>();
            let pack = pack.as_slice_mut_unchecked::<T>();
            let y_stride = im2col.patch.spec.strides[0] as isize;
            let x_stride = im2col.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let input_heigth = shape.hw_dims()[0] as isize;
            let input_width = shape.hw_dims()[1] as isize;
            let kernel_len = im2col.patch.standard_layout_data_field.len();
            let mut writer = im2col.b_pack.write_with_k_outer(pack, im2col.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
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
        Ok(())
    }

    #[inline(never)]
    fn valid_2d<'i, 'p, T: Copy + Datum>(
        im2col: &'i Im2Col,
        input: &'i TensorView,
        shape: &DataShape,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let pack = pack.as_slice_mut_unchecked::<T>();
            let y_stride = im2col.patch.spec.strides[0] as isize;
            let x_stride = im2col.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let mut writer = im2col.b_pack.write_with_k_outer(pack, im2col.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset((g * im2col.ci_per_group * shape.c_stride()) as isize);
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
            Ok(())
        }
    }
}
