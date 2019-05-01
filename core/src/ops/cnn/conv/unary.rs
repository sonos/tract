use ndarray::*;

use crate::internal::*;
use crate::model::*;
use insideout::InsideOut;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use super::mat_mat::MatMat;
use super::vec_mat::VecMat;
use super::Conv;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::DataFormat;

use num_traits::Zero;

use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul};

#[derive(Debug, Clone)]
pub struct ConvUnary {
    pub data_format: DataFormat,
    pub kernel_fmt: KernelFormat,
    pub padding: PaddingSpec,
    pub dilations: TVec<usize>,
    pub strides: TVec<usize>,
    pub kernel: Tensor,

    pub bias: Option<Tensor>,
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
        bias: Option<Tensor>,
        group: usize,
    ) -> TractResult<ConvUnary> {
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
            bias,
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
        self.data_format.shape(&self.full_input_shape).c_dim().to_integer().unwrap() as usize
    }

    fn output_channels(&self) -> usize {
        self.data_format.shape(&self.full_output_shape).c_dim().to_integer().unwrap() as usize
    }

    pub fn to_direct<T: Datum + Copy + Add + AddAssign + Mul + Zero + FloatLike + ndarray::LinalgScalar>(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<Box<Op>> {
        let input_shape = self.data_format.shape(input_full_shape.into());
        let patch = self.patch(input_full_shape);
        let output_shape = self.data_format.from_n_c_hw(
            input_shape.n(),
            self.output_channels(),
            &*patch.output_shape,
        );
        Ok(Box::new(super::Direct::<T>::new(
            input_shape,
            patch,
            self.kernel_as_group_o_i_hw()?,
            self.bias_reshaped(&output_shape.shape)?,
        )?))
    }

    fn kernel_as_group_o_i_hw<T: Datum>(&self) -> TractResult<Array4<T>> {
        let kernel = self.kernel.to_array_view::<T>()?;
        let hw = kernel.shape()[self.kernel_fmt.h_axis()..][..kernel.shape().len() - 2]
            .iter()
            .cloned()
            .product::<usize>();
        let o = self.output_channels() / self.group;
        let i = self.input_channels() / self.group;
        let final_shape = (self.group, o, i, hw);
        trace!(
            "Shuffling kernel: from {:?} {:?} to g_o_i_hw: {:?}",
            self.kernel_fmt,
            kernel.shape(),
            final_shape
        );
        match self.kernel_fmt {
            KernelFormat::HWIO => {
                // H W I O -> HW G I O
                let hw_g_i_o = kernel.into_shape((hw, self.group, i, o))?;
                // HW G I O -> G O I HW
                let g_o_i_hw = hw_g_i_o.permuted_axes([1, 3, 2, 0]);
                let result =
                    Array4::<T>::from_shape_vec(final_shape, g_o_i_hw.iter().cloned().collect())?;
                Ok(result)
            }
            KernelFormat::OIHW => Ok(kernel.into_shape(final_shape)?.to_owned()),
        }
    }

    fn bias_reshaped<T>(&self, output_shape: &[usize]) -> TractResult<Option<ArrayD<T>>>
    where
        T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T>,
    {
        Ok(self
            .bias
            .as_ref()
            .map(|bias| -> TractResult<_> {
                let mut bias_shape: Vec<usize> =
                    ::std::iter::repeat(1).take(output_shape.len()).collect();
                bias_shape[self.data_format.shape(output_shape).c_axis()] = self.output_channels();
                Ok(bias.to_array_view::<T>()?.into_shape(&*bias_shape)?.to_owned())
            })
            .inside_out()?)
    }

    fn to_im2col_pair<T>(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<(Im2Col<T>, TVec<usize>, Box<Op>)>
    where
        T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + FloatLike,
    {
        let patch = self.patch(input_full_shape);
        let input_shape = self.data_format.shape(input_full_shape.into());
        let output_shape = self.data_format.from_n_c_hw(
            input_shape.n(),
            self.output_channels(),
            &*patch.output_shape,
        );
        let kernel = self.kernel.to_array_view::<T>()?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = kernel.len() / self.output_channels();
        let n = patch.output_shape.iter().cloned().product::<usize>();

        let bias = self.bias_reshaped(&*output_shape.shape)?;

        let kernel = self.kernel_as_group_o_i_hw()?;
        let mut packed_kernels: Vec<Tensor> = vec![];

        let (op2, b_pack): (Box<Op>, _) = if m > 1 {
            let mm = T::packed_mat_mul(m, k, n);
            let b_pack = mm.b_pack();

            trace!("Gemm iters={} {:?}", input_shape.n_dim() * self.group, mm);

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
                    subkernel.strides()[0], // o
                    1,
                );
                packed_kernels.push(packed);
            }
            trace!("packed_kernels: {:?}", packed_kernels);
            let conv_gemm = MatMat::new(
                patch.clone(),
                output_shape,
                m,
                k,
                n,
                self.kernel_fmt,
                packed_kernels,
                bias,
                self.group,
                mm.clone(),
            );
            (Box::new(conv_gemm), b_pack)
        } else {
            let mm = T::packed_vec_mat_mul(k, n);
            let b_pack = mm.b_pack();

            trace!("vmm iters={} {:?}", input_shape.n_dim() * self.group, mm);

            for subkernel in kernel.outer_iter() {
                let mut packed = unsafe {
                    Tensor::uninitialized_aligned::<T>(
                        &[mm.packed_a_len()],
                        mm.packed_a_alignment(),
                    )?
                };
                mm.pack_a(packed.as_slice_mut()?.as_mut_ptr(), subkernel.as_ptr(), 1);
                packed_kernels.push(packed);
            }
            let conv_gemm = VecMat::new(
                patch.clone(),
                output_shape,
                k,
                n,
                self.kernel_fmt,
                packed_kernels,
                bias,
                self.group,
                mm,
            );
            (Box::new(conv_gemm), b_pack)
        };
        let c_dim = input_shape.c_dim();

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
    ) -> TractResult<(Box<Op>, TVec<usize>, Box<Op>)>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
    {
        let (op1, shape, op2) = self.to_im2col_pair::<T>(input_full_shape)?;
        Ok((Box::new(op1), shape, op2))
    }

    fn eval_t<T>(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
    {
        debug!("Running by im2col (unoptimized): {:?}", self);
        let input = args_1!(inputs);
        let (im2col, _shape, conv_gemm) = self.to_im2col_pair::<T>(input.shape())?;
        let mega = im2col.im2col(&input.to_array_view()?)?;
        trace!("im2col: {:?}", mega.to_array_view::<T>());
        trace!("conv_gemm: {:?}", conv_gemm);
        conv_gemm.as_stateless().unwrap().eval(tvec!(mega.into()))
    }

    pub fn rm_dummy_axis(&self, axis: usize) -> TractResult<Option<ConvUnary>> {
        let shape = self.data_format.shape(&self.full_input_shape);
        if axis < shape.h_axis() {
            return Ok(None);
        }
        let geo_axis = axis - shape.h_axis();
        if geo_axis >= shape.hw_rank() {
            return Ok(None);
        }
        if self.dilations[geo_axis] != 1
            || self.strides[geo_axis] != 1
            || !self.padding.valid_dim(geo_axis)
        {
            return Ok(None);
        }
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        if kernel_spatial_shape[geo_axis] != 1 {
            return Ok(None);
        }
        fn copy_rm_nth<D: DimLike>(input: &[D], nth: usize) -> TVec<D> {
            input.iter().enumerate().filter(|&(ax, _)| ax != nth).map(|(_, &d)| d).collect()
        }
        let kernel_shape: TVec<usize> =
            copy_rm_nth(self.kernel.shape().clone(), geo_axis + self.kernel_fmt.h_axis());
        let kernel = self.kernel.clone().into_shape(&kernel_shape)?;
        let new_op = ConvUnary {
            data_format: self.data_format,
            kernel_fmt: self.kernel_fmt,
            padding: self.padding.rm_axis(geo_axis),
            dilations: copy_rm_nth(&self.dilations, geo_axis),
            strides: copy_rm_nth(&self.strides, geo_axis),
            kernel,
            bias: self.bias.clone(),
            full_input_shape: copy_rm_nth(&self.full_input_shape, axis),
            full_output_shape: copy_rm_nth(&self.full_output_shape, axis),
            group: self.group,
        };
        Ok(Some(new_op))
    }

    pub fn to_depth_wise<T>(&self, shape: &[usize]) -> TractResult<Box<Op>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq + Sum,
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
            self.kernel_as_group_o_i_hw()?.into_dyn(),
            self.bias_reshaped(&*shape)?,
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
        let n_output_points: TDim = output_dims.iter().map(|d| d.output).product::<TDim>();
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.into_iter().product::<usize>().to_dim();
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            shape.n() * shape.c() * n_output_channels * n_output_points * kernel_surface
                / self.group
        )))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::{AddDims, RmDims};
        if let (Some(add_node), Some(rm_node)) =
            (model.single_prec(node.id)?, model.single_succ(node.id)?)
        {
            if let (Some(add_op), Some(rm_op)) =
                (add_node.op_as::<AddDims>(), rm_node.op_as::<RmDims>())
            {
                if add_op.axes.len() == 1 && rm_op.axes == add_op.axes {
                    let axis = add_op.axes[0];
                    if let Some(op) = self.rm_dummy_axis(axis)? {
                        let mut patch = TypedModelPatch::default();
                        patch.tap_model(&model, model.single_prec(node.id)?.unwrap().inputs[0])?;
                        let out = patch.model.chain(
                            &*node.name,
                            op,
                            tvec!(rm_node.outputs[0].fact.clone()),
                        )?;
                        patch.shunt_outside(OutletId::new(rm_node.id, 0), OutletId::new(out, 0))?;
                        return Ok(Some(patch));
                    }
                }
            }
        }
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        let spatial_rank = self.full_input_shape.len() - 2;
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        if kernel_spatial_shape.iter().product::<usize>() == 1
            && self.dilations.iter().all(|&x| x == 1)
            && self.strides.iter().all(|&x| x == 1)
            && self.group == 1
            && self.bias.is_none()
        {
            debug!("Translating to simple matmul: {:?}", self);
            if self.kernel_fmt == KernelFormat::HWIO && self.data_format == DataFormat::NHWC {
                use crate::ops::math::mat_mul::MatMulUnaryA;
                let kernel_shape = &self.kernel.shape()[spatial_rank..];
                let kernel = self.kernel.clone().into_shape(&kernel_shape)?;
                return Ok(Some(TypedModelPatch::single_unary_op(
                    model,
                    node,
                    MatMulUnaryA::new(kernel),
                )?));
            }
        } else {
            if let Some(shape) = inputs[0].shape.as_finite() {
                let dt = inputs[0].datum_type;
                debug!("Translating to direct: {:?}", self);
                let op = dispatch_floatlike!(Self::to_direct(dt)(self, &*shape))?;
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
                /*
                if (0..spatial_rank).all(|ax| self.padding.valid_dim(ax)) {
                } else if self.group != 1 && self.group == self.output_channels() {
                    return Ok(Some(TypedModelPatch::single_unary_op(
                        model,
                        node,
                        dispatch_floatlike!(Self::to_depth_wise(dt)(self, &shape))?,
                    )?));
                } else {
                    debug!("Translating to im2col: {:?}", self);
                    let (op1, shape, op2) =
                        dispatch_floatlike!(Self::to_boxed_im2col_pair(dt)(self, &shape))?;
                    let mut patch = TypedModelPatch::default();
                    let _ = patch.tap_model(&model, node.inputs[0])?;
                    patch.chain(
                        format!("{}-im2col", node.name),
                        op1,
                        tvec!(TypedTensorInfo {
                            shape: ShapeInfo::from(&*shape),
                            datum_type: dt,
                            konst: None,
                        }),
                    )?;
                    let mm = patch.chain(&*node.name, op2, tvec!(node.outputs[0].fact.clone()))?;
                    patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(mm, 0))?;
                    return Ok(Some(patch));
                }
                */
            }
        }
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        let shape = self.data_format.shape(&fact.shape);
        if fact.axis == shape.n_axis() {
            let mut op = self.clone();
            op.full_output_shape[fact.axis] = fact.pulse().to_dim();
            fact.shape =
                op.full_output_shape
                    .iter()
                    .enumerate()
                    .map(|(ax, &d)| {
                        if ax == fact.axis {
                            fact.pulse()
                        } else {
                            d.to_integer().unwrap() as usize
                        }
                    })
                    .collect();
            let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
            Ok(tvec!(OutletId::new(id, 0)))
        } else if fact.axis == shape.c_axis() {
            bail!("Can not pulsify convolution alongs the input channel axis");
        } else {
            let spatial_rank = self.full_input_shape.len() - 2;
            let geo_axis = fact.axis - shape.h_axis();
            let kernel_spatial_shape =
                &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
            let kernel_len = (kernel_spatial_shape[geo_axis] - 1)
                * self.strides[geo_axis] // TODO do we really need * strides here ?
                * self.dilations[geo_axis];
            let mut augmented_fact = fact.clone();
            augmented_fact.shape[augmented_fact.axis] += kernel_len;
            augmented_fact.delay += kernel_len;

            let mut conv_op = self.clone();
            conv_op.full_input_shape[fact.axis] = augmented_fact.pulse().to_dim();
            conv_op.full_output_shape[fact.axis] =
                (augmented_fact.pulse() - kernel_len / self.strides[geo_axis]).to_dim();
            let mut conv_fact = fact.clone();
            conv_fact.shape = self
                .full_output_shape
                .iter()
                .enumerate()
                .map(|(ax, &d)| {
                    if ax == fact.axis {
                        fact.pulse() / self.strides[geo_axis]
                    } else {
                        d.to_integer().unwrap() as usize
                    }
                })
                .collect();
            conv_fact.delay += kernel_len;
            conv_fact.dim -= kernel_len.to_dim();

            let delay = crate::pulse::delay::Delay::new(fact, 0, kernel_len);
            target.chain_after(
                input,
                format!("{}/Delay", node.name),
                delay,
                tvec!(augmented_fact),
            )?;
            let id = target.chain(&*node.name, conv_op, tvec!(conv_fact))?;

            Ok(tvec!(OutletId::new(id, 0)))
        }
    }
}

impl StatelessOp for ConvUnary {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for ConvUnary {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, self.full_input_shape.clone())?;
        s.equals(&outputs[0].shape, self.full_output_shape.clone())?;
        Ok(())
    }
}
