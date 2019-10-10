use ndarray::*;

use num_traits::AsPrimitive;

use crate::internal::*;
use crate::model::*;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use super::mat_mat::MatMat;
use super::Conv;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::PoolSpec;
use crate::ops::math::mat_mat_mul::MatMatMulUnaryFinite;
use crate::ops::nn::DataFormat;

use std::iter::Sum;

#[derive(Debug, Clone)]
pub struct ConvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub kernel: Tensor,

    pub group: usize,
}

impl ConvUnary {
    pub fn new(conv: &Conv, kernel: Tensor, group: usize) -> TractResult<ConvUnary> {
        let spatial_rank = kernel.rank() - 2;
        let kshape = kernel.shape();

        let output_channels = match conv.kernel_fmt {
            KernelFormat::OIHW => kshape[0],
            KernelFormat::HWIO => kshape[kshape.len() - 1] * group,
        };

        let unary = ConvUnary {
            pool_spec: PoolSpec {
                data_format: conv.data_format,
                padding: conv.padding.clone(),
                strides: conv.strides.clone(),
                dilations: conv.dilations.clone(),
                kernel_shape: kshape[conv.kernel_fmt.h_axis()..][..spatial_rank].into(),
                output_channel_override: Some(output_channels),
            },
            kernel_fmt: conv.kernel_fmt,
            kernel,
            group,
        };
        Ok(unary)
    }

    fn input_channels(&self) -> usize {
        match self.kernel_fmt {
            KernelFormat::OIHW => self.kernel.shape()[1],
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.shape().len() - 2],
        }
    }

    fn output_channels(&self) -> usize {
        let kshape = self.kernel.shape();
        match self.kernel_fmt {
            KernelFormat::OIHW => kshape[0],
            KernelFormat::HWIO => kshape[kshape.len() - 1] * self.group,
        }
    }

    pub fn to_direct(&self, input_full_shape: &[usize]) -> TractResult<super::Direct> {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_full_shape);

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
        let mut conv = f32::mmm(self.output_channels(), kernel_offsets.len(), data_offsets.len());
        unsafe {
            conv.b_from_data_and_offsets(&kernel_offsets, &data_offsets);
            conv.c_from_data_and_strides(
                *output_shape.c_stride() as isize,
                *output_shape.w_stride() as isize,
            );
        }

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

        Ok(super::Direct::new(conv, input_shape, output_shape, packed, vec![]))
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

    pub fn wire_as_im2col_pair<T>(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
    ) -> TractResult<OutletId>
    where
        T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        trace!("to_im2col_pair: {:?}", self);
        let (input_shape, geo, output_shape) =
            self.pool_spec.compute_geo(&*model.outlet_fact(wire)?.shape.as_finite().unwrap());

        trace!("input: {:?}", input_shape);

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.kernel.len() / self.output_channels();
        let n = geo.output_shape.iter().cloned().product::<usize>();

        let mut mmm = T::mmm(m, k, n);
        let (rsc, csc) = match output_shape.fmt {
            DataFormat::NHWC => (1, (m * self.group) as isize),
            DataFormat::NCHW => (n as isize, 1),
        };
        unsafe {
            mmm.c_from_data_and_strides(rsc, csc);
        }
        let b_pack = mmm.b_pack();

        trace!("Gemm iters={} m={} k={} n={}", input_shape.n_dim() * self.group, m, k, n);

        let kernel = self.kernel_as_group_o_ihw()?;
        let packed_as = Array1::from(
            kernel
                .outer_iter()
                .map(|subkernel| {
                    let mut packed = unsafe {
                        Tensor::uninitialized_aligned::<T>(
                            &[mmm.a_pack().len()],
                            mmm.a_pack().alignment(),
                        )?
                    };
                    mmm.a_pack().pack(
                        packed.as_slice_mut()?.as_mut_ptr(),
                        subkernel.as_ptr(),
                        subkernel.strides()[0],
                        subkernel.strides()[1],
                    );
                    Ok(packed)
                })
                .collect::<TractResult<Vec<_>>>()?,
        )
        .into_dyn();
        let packed_as = packed_as.insert_axis(Axis(0)); // n axis broadcast
                                                        /*
                                                        let op2 = MatMat::new(
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
                                                        */

        trace!("{:?}", packed_as);

        let c_dim = input_shape.c_dim().clone();

        wire = model.wire_node(
            format!("{}-im2col", name),
            Im2Col::new(geo.clone(), input_shape, m, k, n, self.group, c_dim / self.group, b_pack),
            &[wire],
        )?[0];

        wire = model.wire_node(
            format!("{}-matmatmul", name),
            MatMatMulUnaryFinite {
                c_shape: output_shape.shape.clone(),
                c_prefix: tvec!(*output_shape.n(), self.group),
                c_prefix_strides: tvec![
                    *output_shape.n_stride() as isize,
                    (output_shape.c() / self.group * output_shape.c_stride()) as isize
                ],
                packed_as,
                mmm,
                non_linear: vec![],
            },
            &[wire],
        )?[0];

        trace!("{:#?}", model);

        Ok(wire)
    }

    /*
    pub fn to_boxed_im2col_pair<T>(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<(Box<dyn TypedOp>, Box<dyn TypedOp>)>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        let (op1, op2) = self.to_im2col_pair::<T>(input_full_shape)?;
        Ok((Box::new(op1), op2))
    }
    */

    fn eval_t<T>(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + FloatLike,
        f32: AsPrimitive<T>,
    {
        let mut model = TypedModel::default();
        let wire =
            model.add_source("source", TypedFact::dt_shape(T::datum_type(), inputs[0].shape())?)?;
        let wire = self.wire_as_im2col_pair::<T>(&mut model, "im2col-adhoc", wire)?;
        model.set_output_outlets(&[wire])?;
        let plan = SimplePlan::new(model)?;
        plan.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
        /*
        let (im2col, conv_gemm) = self.to_im2col_pair::<T>(input.shape())?;
        let mega = im2col.im2col(&input.to_array_view()?)?;
        trace!("im2col: {:?}", mega);
        conv_gemm.as_stateless().unwrap().eval(tvec!(mega.into()))
        */
    }

    pub fn to_depth_wise<T>(&self, input_full_shape: &[usize]) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq + Sum,
        f32: AsPrimitive<T>,
    {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_full_shape);
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

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.pool_spec.info();
        info.push(format!(
            "Kernel shape, {:?}: {:?} (groups:{}))",
            self.kernel_fmt,
            self.kernel.shape(),
            self.group
        ));
        Ok(info)
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for ConvUnary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = self.pool_spec.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>());
        let mut axes = vec![AxisInfo::simple(0).disposable(false)];
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let h_axis = shape.h_axis();
        for (ix, &dim) in kernel_spatial_shape.iter().enumerate() {
            if dim == 1 && self.pool_spec.stride(ix) == 1 {
                axes.push(AxisInfo::simple(ix + h_axis))
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let full_input_shape = input_fact.shape.to_tvec();
        let spatial_rank = full_input_shape.len() - 2;
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        if kernel_spatial_shape.iter().product::<usize>() == 1
            && (0..spatial_rank)
                .all(|i| self.pool_spec.stride(i) == 1 && self.pool_spec.dilation(i) == 1)
            && self.group == 1
        {
            if self.kernel_fmt == KernelFormat::HWIO
                && self.pool_spec.data_format == DataFormat::NHWC
            {
                use crate::ops::math::mat_mul::MatMulUnary;
                let kernel_shape = &self.kernel.shape()[spatial_rank..];
                let kernel = unsafe { self.kernel.clone().into_shape(&kernel_shape)? };
                let op = MatMulUnary::new(kernel.into_arc_tensor(), true, true, true);
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
            }
        } else if let Some(axis) = (0..spatial_rank).find(|&ax| {
            self.pool_spec.padding.valid_dim(ax)
                && self.pool_spec.stride(ax) > 1
                && self.pool_spec.dilation(ax) % self.pool_spec.stride(ax) == 0
        }) {
            let downsample_factor = self.pool_spec.stride(axis);
            let mut new_op = self.clone();
            if new_op.pool_spec.dilation(axis) > 1 {
                new_op.pool_spec.dilations.as_mut().unwrap()[axis] /= downsample_factor;
            }
            new_op.pool_spec.strides.as_mut().unwrap()[axis] /= downsample_factor;
            let mut patch = TypedModelPatch::default();
            let tap = patch.tap_model(model, node.inputs[0])?;
            let shape =
                self.pool_spec.data_format.shape(input_fact.shape.iter().collect::<TVec<TDim>>());
            let down = patch.wire_node(
                format!("Downsample-{}", node.name),
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor, 0),
                &[tap],
            )?;
            let id = patch.wire_node(&*node.name, new_op, &down)?[0];
            patch.shunt_outside(OutletId::new(node.id, 0), id)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec());
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let output_dims = self.pool_spec.padding.compute(
            shape.hw_dims(),
            kernel_spatial_shape,
            &*self.pool_spec.dilations.clone().unwrap_or(tvec!(1; kernel_spatial_shape.len())),
            &*self.pool_spec.strides.clone().unwrap_or(tvec!(1; kernel_spatial_shape.len())),
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

    fn dispose_dummy_axis(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        axis: usize,
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let shape = self.pool_spec.data_format.shape(full_input_shape);
        if axis < shape.h_axis() {
            bail!("Only spatial axis can be disposed of.");
        }
        let geo_axis = axis - shape.h_axis();
        if geo_axis >= shape.hw_rank() {
            bail!("Only spatial axis can be disposed of.");
        }
        if self.pool_spec.dilation(geo_axis) != 1
            || self.pool_spec.stride(geo_axis) != 1
            || !self.pool_spec.padding.valid_dim(geo_axis)
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
            pool_spec: PoolSpec {
                data_format: self.pool_spec.data_format,
                padding: self.pool_spec.padding.rm_axis(geo_axis),
                dilations: self.pool_spec.dilations.as_ref().map(|d| copy_rm_nth(&d, geo_axis)),
                kernel_shape: copy_rm_nth(&self.pool_spec.kernel_shape, geo_axis),
                strides: self.pool_spec.strides.as_ref().map(|s| copy_rm_nth(&s, geo_axis)),
                output_channel_override: self.pool_spec.output_channel_override,
            },
            kernel_fmt: self.kernel_fmt,
            kernel,
            group: self.group,
        };
        Ok(Some(Box::new(new_op)))
    }

    fn pulsify(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        self.pool_spec.pulsify(source, node, self, target, mapping)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let spatial_rank = full_input_shape.len() - 2;
        if let Some(shape) = input_fact.shape.as_finite() {
            let dt = input_fact.datum_type;
            if (0..spatial_rank).all(|ax| self.pool_spec.padding.valid_dim(ax))
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
                let mut patch = TypedModelPatch::default();
                let wire = patch.tap_model(model, node.inputs[0])?;
                let wire = dispatch_floatlike!(Self::wire_as_im2col_pair(dt)(
                    self,
                    &mut patch,
                    &*node.name,
                    wire
                ))?;
                patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    typed_op_as_op!();
}

impl PulsedOp for ConvUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        self.pool_spec.pulsed_output_facts(inputs)
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
