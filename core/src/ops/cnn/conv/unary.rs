use ndarray::*;

use num_traits::AsPrimitive;

use crate::internal::*;
use crate::model::*;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use super::mat_mat::MatMat;
use super::vec_mat::VecMat;
use super::Conv;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::DataFormat;

use std::iter::Sum;

#[derive(Debug, Clone)]
pub struct ConvUnary {
    pub data_format: DataFormat,
    pub kernel_fmt: KernelFormat,
    pub padding: PaddingSpec,
    pub dilations: TVec<usize>,
    pub strides: TVec<usize>,
    pub kernel: Tensor,

    pub full_input_shape: TVec<TDim>,
    pub full_output_shape: TVec<TDim>,
    pub group: usize,
}

impl ConvUnary {
    pub fn new(
        conv: &Conv,
        full_input_shape: &[TDim],
        full_output_shape: &[TDim],
        kernel: Tensor,
        group: usize,
    ) -> TractResult<ConvUnary> {
        for td in full_input_shape {
            if let Ok(d) = td.to_integer() {
                if d < 0 {
                    bail!("Negative input shape dim detected");
                }
            }
        }
        for td in full_output_shape {
            if let Ok(d) = td.to_integer() {
                if d < 0 {
                    bail!("Negative output shape dim detected");
                }
            }
        }
        let spatial_rank = full_input_shape.len() - 2;
        let dilations =
            conv.dilations.as_ref().map(|a| TVec::from(&**a)).unwrap_or(tvec!(1; spatial_rank));
        let strides =
            conv.strides.as_ref().map(|a| TVec::from(&**a)).unwrap_or(tvec!(1; spatial_rank));

        let unary = ConvUnary {
            data_format: conv.data_format,
            kernel_fmt: conv.kernel_fmt,
            padding: conv.padding.clone(),
            dilations,
            strides,
            kernel,
            full_input_shape: full_input_shape.into(),
            full_output_shape: full_output_shape.into(),
            group,
        };
        Ok(unary)
    }

    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..(input_full_shape.len() - 2)];
        let output_inner_stride = match self.data_format {
            DataFormat::NHWC => self.output_channels(),
            DataFormat::NCHW => 1,
        };
        let spec = PatchSpec::for_full_shape(self.data_format.clone(), input_full_shape)
            .with_kernel_shape(kernel_spatial_shape.into())
            .with_padding(self.padding.clone())
            .with_dilations(self.dilations.clone())
            .with_strides(self.strides.clone())
            .with_output_inner_stride(output_inner_stride);
        spec.into_patch()
    }

    fn input_channels(&self) -> usize {
        match self.kernel_fmt {
            KernelFormat::OIHW => self.kernel.shape()[1],
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.shape().len() - 2],
        }
    }

    fn output_channels(&self) -> usize {
        self.data_format.shape(&self.full_output_shape).c_dim().to_integer().unwrap() as usize
    }

    pub fn to_direct(&self, input_full_shape: &[usize]) -> TractResult<super::Direct> {
        assert!(
            (0..input_full_shape.len() - 2).all(|ax| self.padding.valid_dim(ax)) && self.group == 1
        );

        let patch = self.patch(input_full_shape);
        assert!(!patch.padded);

        let input_shape = self.data_format.shape(input_full_shape.into());
        let output_shape = self.data_format.from_n_c_hw(
            *input_shape.n(),
            self.output_channels(),
            &*patch.output_shape,
        );
        let channel_stride = input_shape.c_stride();
        let rpatch = &patch;
        let data_offsets: Vec<isize> = patch.centers_offsets();
        let kernel_offsets: Vec<isize> = (0..self.input_channels())
            .flat_map(|ici| {
                rpatch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| x + (ici * channel_stride) as isize)
            })
            .collect();
        let conv = f32::mmm(self.output_channels(), kernel_offsets.len(), data_offsets.len());

        let kernel = self.kernel_as_group_o_ihw()?;
        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<f32>(&[conv.a_pack().len()], conv.a_pack().alignment())?
        };
        conv.a_pack().pack(
            packed.as_slice_mut()?.as_mut_ptr(),
            kernel.as_slice().unwrap().as_ptr(),
            kernel.strides()[1],
            kernel.strides()[2],
        );

        Ok(super::Direct::new(
            conv,
            data_offsets,
            kernel_offsets,
            input_shape,
            output_shape,
            packed,
            vec![],
        ))
    }

    fn kernel_as_group_o_ihw<T: Datum>(&self) -> TractResult<Array3<T>> {
        let kernel = self.kernel.to_array_view::<T>()?;
        let final_shape = (
            self.group,
            self.output_channels() / self.group,
            kernel.len() / self.output_channels(),
        );
        trace!("kernel shape (group, output, rest) = {:?}", final_shape);
        let hw_rank = kernel.ndim() - 2;
        match self.kernel_fmt {
            KernelFormat::HWIO => {
                let mut shape = kernel.shape().to_vec();
                shape.insert(hw_rank + 1, self.group);
                shape[hw_rank] /= self.group;
                let kernel = kernel.into_shape(shape)?;
                let mut permutation: Vec<usize> = vec![hw_rank + 1, hw_rank + 2, hw_rank];
                permutation.extend(0..hw_rank);
                let permuted = kernel.permuted_axes(permutation);
                Ok(Array3::<T>::from_shape_vec(final_shape, permuted.iter().cloned().collect())?)
            }
            KernelFormat::OIHW => Ok(kernel.into_shape(final_shape)?.to_owned()),
        }
    }

    pub fn to_im2col_pair<T>(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<(Im2Col<T>, TVec<usize>, Box<dyn TypedOp>)>
    where
        T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        trace!("to_im2col_pair: {:?}", self);
        let patch = self.patch(input_full_shape);
        let input_shape = self.data_format.shape(input_full_shape.into());
        let output_shape = self.data_format.from_n_c_hw(
            *input_shape.n(),
            self.output_channels(),
            &*patch.output_shape,
        );
        let kernel = self.kernel.to_array_view::<T>()?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = kernel.len() / self.output_channels();
        let n = patch.output_shape.iter().cloned().product::<usize>();

        let kernel = self.kernel_as_group_o_ihw()?;
        let mut packed_kernels: Vec<Tensor> = vec![];

        let (op2, b_pack): (Box<dyn TypedOp>, _) = if m > 1 {
            let mm = T::mmm(m, k, n);
            let b_pack = mm.b_pack();

            trace!("Gemm iters={} m={} k={} n={}", input_shape.n_dim() * self.group, m, k, n);

            for subkernel in kernel.outer_iter() {
                let mut packed = unsafe {
                    Tensor::uninitialized_aligned::<T>(
                        &[mm.a_pack().len()],
                        mm.a_pack().alignment(),
                    )?
                };
                mm.a_pack().pack(
                    packed.as_slice_mut()?.as_mut_ptr(),
                    subkernel.as_ptr(),
                    subkernel.strides()[0],
                    subkernel.strides()[1],
                );
                packed_kernels.push(packed);
            }
            let conv_gemm = MatMat::new(
                patch.clone(),
                output_shape,
                m,
                k,
                n,
                self.kernel_fmt,
                packed_kernels,
                self.group,
                mm.clone(),
                vec![],
            );
            (Box::new(conv_gemm), b_pack)
        } else {
            let mm = T::packed_vec_mat_mul(k, n);
            let b_pack = mm.b_pack();

            trace!("Gemm iters={} m={} k={} n={}", input_shape.n_dim() * self.group, m, k, n);

            for subkernel in kernel.outer_iter() {
                let mut packed = unsafe {
                    Tensor::uninitialized_aligned::<T>(
                        &[mm.packed_a_len()],
                        mm.packed_a_alignment(),
                    )?
                };
                mm.pack_a(
                    packed.as_slice_mut()?.as_mut_ptr(),
                    subkernel.as_ptr(),
                    subkernel.strides()[1],
                );
                packed_kernels.push(packed);
            }
            let conv_gemm = VecMat::new(
                patch.clone(),
                output_shape,
                k,
                n,
                self.kernel_fmt,
                packed_kernels,
                self.group,
                mm,
            );
            (Box::new(conv_gemm), b_pack)
        };
        let c_dim = input_shape.c_dim().clone();

        let im2col = Im2Col::new(
            patch.clone(),
            input_shape,
            m,
            k,
            n,
            self.group,
            c_dim / self.group,
            b_pack,
        );
        let intermediary_shape = im2col.output_shape().into();
        Ok((im2col, intermediary_shape, op2))
    }

    pub fn to_boxed_im2col_pair<T>(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<(Box<dyn TypedOp>, TVec<usize>, Box<dyn TypedOp>)>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        let (op1, shape, op2) = self.to_im2col_pair::<T>(input_full_shape)?;
        Ok((Box::new(op1), shape, op2))
    }

    fn eval_t<T>(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        let input = args_1!(inputs);
        let (im2col, _shape, conv_gemm) = self.to_im2col_pair::<T>(input.shape())?;
        let mega = im2col.im2col(&input.to_array_view()?)?;
        trace!("im2col: {:?}", mega);
        conv_gemm.as_stateless().unwrap().eval(tvec!(mega.into()))
    }

    pub fn to_depth_wise<T>(&self, shape: &[usize]) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq + Sum,
        f32: AsPrimitive<T>,
    {
        let patch = self.patch(shape);
        let input_shape = self.data_format.shape(shape.into());
        let output_shape = self
            .full_output_shape
            .iter()
            .map(|a| a.to_integer().map(|a| a as usize))
            .collect::<TractResult<TVec<usize>>>()?;
        let output_shape = self.data_format.shape(output_shape);
        let op = DepthWise::<T>::new(
            patch,
            input_shape,
            output_shape,
            self.kernel_as_group_o_ihw()?.into_dyn(),
        );
        Ok(Box::new(op))
    }
}

impl Op for ConvUnary {
    fn name(&self) -> Cow<str> {
        "ConvUnary".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let shape = self.data_format.shape(inputs[0].shape.iter().collect::<TVec<TDim>>());
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let output_dims = self.padding.compute(
            shape.hw_dims(),
            kernel_spatial_shape,
            &*self.dilations,
            &*self.strides,
        );
        let n_output_points: TDim = output_dims.iter().map(|d| d.output.clone()).product::<TDim>();
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.into_iter().product::<usize>().to_dim();
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            shape.n().clone() * shape.c() * n_output_channels * n_output_points * kernel_surface
                / self.group
        )))
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("Data format: {:?}", self.data_format),
            format!(
                "Kernel shape, {:?}: {:?} (strides:{:?} dilations:{:?} groups:{}))",
                self.kernel_fmt,
                self.kernel.shape(),
                self.strides,
                self.dilations,
                self.group
            ),
        ])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let spatial_rank = self.full_input_shape.len() - 2;
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        if kernel_spatial_shape.iter().product::<usize>() == 1
            && self.dilations.iter().all(|&x| x == 1)
            && self.strides.iter().all(|&x| x == 1)
            && self.group == 1
        {
            if self.kernel_fmt == KernelFormat::HWIO && self.data_format == DataFormat::NHWC {
                use crate::ops::math::mat_mul::MatMulUnary;
                let kernel_shape = &self.kernel.shape()[spatial_rank..];
                let kernel = unsafe { self.kernel.clone().into_shape(&kernel_shape)? };
                let op = MatMulUnary::new(kernel.into_arc_tensor(), true, true, true);
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
            }
        } else if let Some(axis) = (0..self.strides.len()).find(|&ax| {
            self.padding.valid_dim(ax)
                && self.strides[ax] > 1
                && self.dilations[ax] % self.strides[ax] == 0
        }) {
            let downsample_factor = self.strides[axis];
            let mut new_op = self.clone();
            if new_op.dilations[axis] > 1 {
                new_op.dilations[axis] /= downsample_factor;
            }
            new_op.strides[axis] /= downsample_factor;
            let mut patch = TypedModelPatch::default();
            patch.tap_model(model, node.inputs[0])?;
            let shape = self.data_format.shape(input_fact.shape.iter().collect::<TVec<TDim>>());
            let downample_op =
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor, 0);
            let downsampled_fact = downample_op.transform_fact(input_fact)?;
            patch.chain(
                format!("Downsample-{}", node.name),
                downample_op,
                tvec!(downsampled_fact),
            )?;
            let id = patch.chain(&*node.name, new_op, tvec!(node.outputs[0].fact.clone()))?;
            patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(id, 0))?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for ConvUnary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl TypedOp for ConvUnary {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*self.full_output_shape)?))
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = self.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>());
        let mut axes = vec![AxisInfo::simple(0).disposable(false)];
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let h_axis = shape.h_axis();
        for (ix, &dim) in kernel_spatial_shape.iter().enumerate() {
            if dim == 1 && self.strides[ix] == 1 {
                axes.push(AxisInfo::simple(ix + h_axis))
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn dispose_dummy_axis(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        axis: usize,
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        let shape = self.data_format.shape(&self.full_input_shape);
        if axis < shape.h_axis() {
            bail!("Only spatial axis can be disposed of.");
        }
        let geo_axis = axis - shape.h_axis();
        if geo_axis >= shape.hw_rank() {
            bail!("Only spatial axis can be disposed of.");
        }
        if self.dilations[geo_axis] != 1
            || self.strides[geo_axis] != 1
            || !self.padding.valid_dim(geo_axis)
        {
            bail!("Can not dispose of axis with dilation, stride or padding.");
        }
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        if kernel_spatial_shape[geo_axis] != 1 {
            bail!("Can not dispose of axis with actual convolution.");
        }
        fn copy_rm_nth<D: DimLike>(input: &[D], nth: usize) -> TVec<D> {
            input.iter().enumerate().filter(|&(ax, _)| ax != nth).map(|(_, d)| d.clone()).collect()
        }
        let kernel_shape: TVec<usize> =
            copy_rm_nth(self.kernel.shape().clone(), geo_axis + self.kernel_fmt.h_axis());
        let kernel = unsafe { self.kernel.clone().into_shape(&kernel_shape)? };
        let new_op = ConvUnary {
            data_format: self.data_format,
            kernel_fmt: self.kernel_fmt,
            padding: self.padding.rm_axis(geo_axis),
            dilations: copy_rm_nth(&self.dilations, geo_axis),
            strides: copy_rm_nth(&self.strides, geo_axis),
            kernel,
            full_input_shape: copy_rm_nth(&self.full_input_shape, axis),
            full_output_shape: copy_rm_nth(&self.full_output_shape, axis),
            group: self.group,
        };
        Ok(Some(Box::new(new_op)))
    }


    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let mut input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?;
        let shape = self.data_format.shape(&fact.shape);
        if fact.axis == shape.n_axis() {
            let mut op = self.clone();
            let mut fact = fact.clone();
            op.full_output_shape[fact.axis] = fact.pulse().to_dim();
            fact.shape =
                op.full_output_shape.iter().map(|d| d.to_integer().unwrap() as usize).collect();
            let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
            Ok(tvec!(OutletId::new(id, 0)))
        } else if fact.axis == shape.c_axis() {
            bail!("Can not pulsify convolution along the input channel axis");
        } else {
            let spatial_rank = self.full_input_shape.len() - 2;
            let geo_axis = fact.axis - shape.h_axis();
            let stride = self.strides[geo_axis];
            if fact.pulse() % stride != 0 {
                bail!("Convolution pulsification can only be achieved when the pulse length is a multiple of the stride. Got pulse={}, stride={}", fact.pulse(), stride)
            }
            let kernel_spatial_shape =
                &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
            let kernel_overreach = (kernel_spatial_shape[geo_axis] - 1) * self.dilations[geo_axis];

            let overlap = (kernel_overreach + 1).saturating_sub(stride);

            let mut augmented_fact = fact.clone();
            augmented_fact.shape[augmented_fact.axis] += overlap;
            augmented_fact.delay += overlap;
            augmented_fact.delay = augmented_fact.delay.div_ceil(stride) * stride;

            let mut conv_op = self.clone();
            conv_op.full_input_shape[fact.axis] = augmented_fact.pulse().to_dim();
            conv_op.full_output_shape[fact.axis] =
                ((augmented_fact.pulse() - overlap) / self.strides[geo_axis]).to_dim();
            let mut conv_fact = fact.clone();
            conv_fact.shape = conv_op
                .full_output_shape
                .iter()
                .map(|d| d.to_integer().unwrap() as usize)
                .collect();
            conv_fact.delay = augmented_fact.delay / stride;
            conv_fact.dim = (conv_fact.dim - kernel_overreach.to_dim()).div_ceil(stride.to_dim());

            if augmented_fact != *fact {
                let extra_delay = augmented_fact.delay - fact.delay - overlap;
                let delay = crate::pulse::delay::Delay::new(
                    fact.clone(),
                    extra_delay,
                    augmented_fact.pulse() - fact.pulse(),
                );
                let node = target.chain_after(
                    input,
                    format!("{}/Delay", node.name),
                    delay,
                    tvec!(augmented_fact),
                )?;
                input = OutletId::new(node, 0);
            }
            let id = target.chain_after(input, &*node.name, conv_op, tvec!(conv_fact))?;

            Ok(tvec!(OutletId::new(id, 0)))
        }
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let spatial_rank = self.full_input_shape.len() - 2;
        if let Some(shape) = input_fact.shape.as_finite() {
            let dt = input_fact.datum_type;
            if (0..spatial_rank).all(|ax| self.padding.valid_dim(ax))
                && dt == f32::datum_type()
                && self.group == 1
            {
                let op = self.to_direct(&*shape)?;
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
            } else if self.group != 1 && self.group == self.output_channels() {
                return Ok(Some(TypedModelPatch::single_unary_op(
                    model,
                    node,
                    dispatch_floatlike!(Self::to_depth_wise(dt)(self, &shape))?,
                )?));
            } else {
                let (op1, shape, op2) =
                    dispatch_floatlike!(Self::to_boxed_im2col_pair(dt)(self, &shape))?;
                let mut patch = TypedModelPatch::default();
                let _ = patch.tap_model(&model, node.inputs[0])?;
                patch.chain(
                    format!("{}-im2col", node.name),
                    op1,
                    tvec!(TypedTensorInfo::dt_shape(dt, &*shape)?),
                )?;
                let mm = patch.chain(&*node.name, op2, tvec!(node.outputs[0].fact.clone()))?;
                patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(mm, 0))?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

}
