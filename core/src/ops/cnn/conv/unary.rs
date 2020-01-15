use std::fmt;
use std::ops::{Add, Mul};

use ndarray::*;

use num_traits::Zero;

use crate::internal::*;
use crate::model::*;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use super::Conv;
use crate::ops::array::TypedReshape;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::PoolSpec;
use crate::ops::matmul;
use crate::ops::matmul::mmm_wrapper::MMMWrapper;
use crate::ops::nn::DataFormat;
use crate::ops::quant::QParams;

use tract_linalg::frame::mmm::FusedSpec;
use tract_linalg::frame::PackA;

use std::iter::Sum;

#[derive(Debug, Clone)]
pub struct ConvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub kernel: Arc<Tensor>,

    pub group: usize,

    pub bias: Option<Arc<Tensor>>,
    pub q_params: Option<QParams>,
}

impl ConvUnary {
    pub fn new(
        conv: &Conv,
        kernel: Arc<Tensor>,
        group: usize,
        bias: Option<Arc<Tensor>>,
        q_params: Option<QParams>,
    ) -> TractResult<ConvUnary> {
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
            bias,
            q_params,
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

    fn kernel_as_packed_as<T: Datum + Copy + Zero>(
        &self,
        packer: &PackA<T>,
    ) -> TractResult<ArrayD<Arc<Tensor>>> {
        let kernel = self.kernel_as_group_o_ihw()?;
        let packed_as = Array1::from(
            kernel
                .outer_iter()
                .map(|subkernel| {
                    let mut packed = unsafe {
                        Tensor::uninitialized_aligned::<T>(&[packer.len()], packer.alignment())?
                    };
                    packer.pack(
                        packed.as_slice_mut()?.as_mut_ptr(),
                        subkernel.as_ptr(),
                        subkernel.strides()[0],
                        subkernel.strides()[1],
                    );
                    Ok(packed.into_arc_tensor())
                })
                .collect::<TractResult<Vec<_>>>()?,
        )
        .into_dyn();
        Ok(packed_as.insert_axis(Axis(0)))
    }

    fn bias_as_non_linear<T>(&self) -> TractResult<Option<ArrayD<Vec<FusedSpec<T>>>>>
    where
        T: Datum + Copy,
    {
        use crate::itertools::Itertools;
        if let Some(bias) = &self.bias {
            let bias = bias.cast_to::<T>()?;
            let bias = bias.as_slice::<T>()?;
            Ok(Some(
                Array2::from_shape_vec(
                    (1, self.group),
                    bias.iter()
                        .chunks(self.output_channels() / self.group)
                        .into_iter()
                        .map(|c| vec![FusedSpec::PerRowAdd(c.into_iter().cloned().collect())])
                        .collect(),
                )?
                .into_dyn(),
            ))
        } else {
            Ok(None)
        }
    }

    pub unsafe fn wire_as_im2col_pair(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: OutletId,
        direct: bool,
    ) -> TractResult<OutletId> {
        let a = self.kernel.datum_type();
        let b = model.outlet_fact(wire)?.datum_type;
        if (a, b) == (f32::datum_type(), f32::datum_type()) {
            return self.wire_as_im2col_pair_t(model, name, wire, direct, &|m, k, n| {
                MMMWrapper::Plain((tract_linalg::ops().smmm)(m, k, n))
            });
        } else if (a, b) == (u8::datum_type(), u8::datum_type()) {
            return self.wire_as_im2col_pair_t(model, name, wire, direct, &|m, k, n| {
                MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_i32)(m, k, n))
            });
        } else if (a, b) == (i8::datum_type(), i8::datum_type()) {
            if let Some(q) = &self.q_params {
                if q.c_datum_type == i8::datum_type() {
                    return self.wire_as_im2col_pair_t(model, name, wire, direct, &|m, k, n| {
                        MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i8)(m, k, n))
                    });
                }
            } else {
                return self.wire_as_im2col_pair_t(model, name, wire, direct, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n))
                });
            }
        }
        bail!("Unsupported combination for Conv (filters: {:?}, data:{:?})", a, b);
    }

    unsafe fn wire_as_im2col_pair_t<TA, TB, TC, TI>(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
        direct: bool,
        mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
    ) -> TractResult<OutletId>
    where
        TA: Datum + Copy + Zero,
        TB: Datum + Copy + Zero,
        TC: Datum + Copy,
        TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
    {
        trace!("to_im2col_pair: {:?}", self);
        let (input_shape, geo, output_shape) =
            self.pool_spec.compute_geo(&*model.outlet_fact(wire)?.shape.as_finite().unwrap());

        trace!("input: {:?}", input_shape);

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.kernel.len() / self.output_channels();
        let n = geo.output_shape.iter().cloned().product::<usize>();

        let mut mmm = mmm(m, k, n);
        let (rsc, csc) = match output_shape.fmt {
            DataFormat::NHWC | DataFormat::HWC => (1, self.output_channels() as isize),
            DataFormat::NCHW | DataFormat::CHW => (n as isize, 1),
        };
        mmm.as_mmm_mut().c_from_data_and_strides(rsc, csc);

        if let Some(q) = self.q_params.as_ref() {
            mmm.set_quant_params(q)?;
        }

        trace!(
            "Gemm iters={} m={} k={} n={}",
            input_shape.n_dim().unwrap_or(&1) * self.group,
            m,
            k,
            n
        );
        trace!("{:?}", mmm);

        if direct {
            let channel_stride = input_shape.c_stride();
            let data_offsets: Vec<isize> = geo.centers_offsets();
            let kernel_offsets: Vec<isize> = (0..self.input_channels())
                .flat_map(|ici| {
                    geo.standard_layout_data_field
                        .iter()
                        .map(move |x| x + (ici * channel_stride) as isize)
                })
                .collect();
            mmm.as_mmm_mut().b_from_data_and_offsets(&kernel_offsets, &data_offsets);
        } else {
            let c_dim = *input_shape.c_dim();
            wire = model.wire_node(
                format!("{}-im2col", name),
                Im2Col::new(
                    geo.clone(),
                    input_shape,
                    m,
                    k,
                    n,
                    self.group,
                    c_dim / self.group,
                    mmm.as_mmm().b_pack(),
                    self.q_params
                        .as_ref()
                        .and_then(|q| q.zero_point_b.as_ref())
                        .map(|t| t.to_scalar::<TB>().map(|x| *x))
                        .transpose()?
                        .unwrap_or(TB::default()),
                ),
                &[wire],
            )?[0];
        }

        let c_prefix_dim_and_stride = if *output_shape.n().unwrap_or(&1) != 1 || self.group != 1 {
            let mut dims = tvec!(self.group as usize);
            let mut strides =
                tvec!((output_shape.c() / self.group * output_shape.c_stride()) as isize);
            if output_shape.n().is_some() {
                dims.insert(0, *output_shape.n().unwrap());
                strides.insert(0, *output_shape.n_stride().unwrap() as isize);
            }
            Some((dims, strides))
        } else {
            None
        };

        wire = model.wire_node(
            format!("{}-matmatmul", name),
            matmul::phy::MatMatMulUnaryFinite {
                c_trans: false,
                c_fact: TypedFact::dt_shape(TC::datum_type(), &*output_shape.shape)?,
                c_prefix_dim_and_stride,
                packed_as: self.kernel_as_packed_as(&mmm.as_mmm().a_pack())?,
                fused_ops: self.bias_as_non_linear()?,
                mmm,
            },
            &[wire],
        )?[0];

        Ok(wire)
    }

    pub fn to_depth_wise<T>(&self, input_full_shape: &[usize]) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + PartialEq + Sum,
    {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_full_shape);
        let bias =
            if let Some(b) = self.bias.as_ref() { Some(b.as_slice::<T>()?.to_vec()) } else { None };
        let op = DepthWise::<T>::new(
            patch,
            input_shape,
            output_shape,
            self.kernel_as_group_o_ihw()?.into_dyn(),
            bias,
        );
        Ok(Box::new(op))
    }

    fn declutter_stride_slice_to_downsample(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let spatial_rank = self.kernel.rank() - 2;
        if let Some(axis) = (0..spatial_rank).find(|&ax| {
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

    fn declutter_as_matmul(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::matmul::MatMulUnary;
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape);
        if input_shape.hw_rank() == 1
            && self.group == 1
            && self.pool_spec.stride(0) == 1
            && self.pool_spec.dilation(0) == 1
            && self.kernel.len() == self.input_channels() * self.output_channels()
        {
            let ci = self.input_channels();
            let co = self.output_channels();
            let ker = self.kernel.clone().into_tensor();
            let (a_shape, a_trans) = if self.kernel_fmt == KernelFormat::HWIO {
                ([ci, co], true)
            } else {
                ([co, ci], false)
            };
            let a = unsafe { ker.into_shape(&a_shape)? }.into_arc_tensor();
            let trans_data = self.pool_spec.data_format == DataFormat::HWC || self.pool_spec.data_format == DataFormat::NHWC;
            let op = MatMulUnary {
                a,
                a_trans,
                b_trans: trans_data,
                c_trans: trans_data,
                q_params: self.q_params.clone(),
            };
            let mut patch = TypedModelPatch::default();
            let wire = patch.tap_model(model, node.inputs[0])?;
            let mut wire = patch.wire_node(&*node.name, op, &[wire])?[0];
            if let Some(b) = &self.bias {
                let bias_shape = if trans_data { tvec!(1, co) } else { tvec!(co, 1) };
                let b = unsafe { b.clone().into_tensor().into_shape(&bias_shape)? };
                wire = patch.wire_node(
                    format!("{}-bias", node.name),
                    crate::ops::math::add::unary(b.into_arc_tensor()),
                    &[wire],
                )?[0];
            }
            patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
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
        if let Some(b) = &self.bias {
            info.push(format!("Bias: {:?}", b))
        }
        if let Some(qp) = &self.q_params {
            info.push(format!("Quant: {:?}", qp))
        }
        Ok(info)
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for ConvUnary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut model = TypedModel::default();
        let dt = inputs[0].datum_type();
        let wire = model.add_source("source", TypedFact::dt_shape(dt, inputs[0].shape())?)?;
        let wire = unsafe { self.wire_as_im2col_pair(&mut model, "im2col-adhoc", wire, false)? };
        model.set_output_outlets(&[wire])?;
        let plan = SimplePlan::new(model)?;
        plan.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = self.pool_spec.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>());
        let mut axes = vec![];
        if let Some(n_axis) = shape.n_axis() {
            axes.push(AxisInfo::simple(n_axis).disposable(true));
        }
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
        for d in &[Self::declutter_stride_slice_to_downsample, Self::declutter_as_matmul] {
            if let Some(p) = d(&self, model, node)? {
                return Ok(Some(p));
            }
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
        let one = 1.to_dim();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            shape.n().unwrap_or(&one).clone()
                * shape.c()
                * n_output_channels
                * n_output_points
                * kernel_surface
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
        if Some(axis) == shape.n_axis() {
            return Ok(Some(Box::new(ConvUnary {
                pool_spec: self.pool_spec.dispose_n_axis(),
                ..self.clone()
            })));
        }
        if axis == shape.c_axis() {
            bail!("Channel axis can not be disposed of");
        }
        let geo_axis = axis - shape.h_axis();
        if geo_axis >= shape.hw_rank() {
            bail!("Only spatial axis can be disposed of.");
        }
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        if self.pool_spec.dilation(geo_axis) != 1
            || self.pool_spec.stride(geo_axis) != 1
            || (!self.pool_spec.padding.valid_dim(geo_axis) && kernel_spatial_shape[geo_axis] != 1)
        {
            bail!("Can not dispose of axis with dilation, stride or padding.");
        }
        if kernel_spatial_shape[geo_axis] != 1 {
            bail!("Can not dispose of axis with actual convolution.");
        }
        fn copy_rm_nth<D: DimLike>(input: &[D], nth: usize) -> TVec<D> {
            input.iter().enumerate().filter(|&(ax, _)| ax != nth).map(|(_, d)| d.clone()).collect()
        }
        let kernel_shape: TVec<usize> =
            copy_rm_nth(self.kernel.shape().clone(), geo_axis + self.kernel_fmt.h_axis());
        let kernel = unsafe { self.kernel.as_ref().clone().into_shape(&kernel_shape)? };
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
            kernel: kernel.into_arc_tensor(),
            group: self.group,
            bias: self.bias.clone(),
            q_params: self.q_params.clone(),
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
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape);
        let spatial_rank = input_shape.hw_rank();
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        if let Some(shape) = input_fact.shape.as_finite() {
            unsafe {
                let dt = input_fact.datum_type;
                if kernel_spatial_shape.iter().product::<usize>() == 1
                    && (0..spatial_rank)
                        .all(|i| self.pool_spec.stride(i) == 1 && self.pool_spec.dilation(i) == 1)
                    && self.group == 1
                {
                    use crate::ops::matmul::MatMulUnary;
                    let mut patch = TypedModelPatch::default();
                    let mut wire = patch.tap_model(model, node.inputs[0])?;
                    let input_c_is_last = input_shape.c_axis() == input_shape.rank() - 1;
                    let mut reshaped_input = tvec!(
                        input_shape.n().cloned().unwrap_or(1.to_dim()),
                        input_shape.hw_dims().iter().cloned().product::<TDim>(),
                        input_shape.c().clone(),
                    );
                    if !input_c_is_last {
                        reshaped_input.swap(1, 2);
                    }
                    wire =
                        patch.wire_node(&*node.name, TypedReshape::new(reshaped_input), &[wire])?
                            [0];
                    let kernel_shape = match self.kernel_fmt {
                        KernelFormat::HWIO => &self.kernel.shape()[spatial_rank..],
                        KernelFormat::OIHW => &self.kernel.shape()[..2],
                    };
                    let kernel = self.kernel.as_ref().clone().into_shape(&kernel_shape)?;
                    wire = patch.wire_node(
                        &*node.name,
                        MatMulUnary::new(
                            kernel.into_arc_tensor(),
                            self.kernel_fmt == KernelFormat::HWIO,
                            input_c_is_last,
                            input_c_is_last,
                            self.q_params.clone(),
                        ),
                        &[wire],
                    )?[0];
                    if let Some(ref bias) = self.bias {
                        let bias: Arc<Tensor> = if input_c_is_last {
                            bias.clone()
                        } else {
                            bias.clone()
                                .into_tensor()
                                .into_shape(&[bias.len(), 1])?
                                .into_arc_tensor()
                        };
                        wire = patch.wire_node(
                            format!("{}-bias", node.name),
                            crate::ops::math::add::unary(bias),
                            &[wire],
                        )?[0];
                    }
                    wire = patch.wire_node(
                        &*node.name,
                        TypedReshape::new(node.outputs[0].fact.shape.to_tvec()),
                        &[wire],
                    )?[0];
                    patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                } else if (0..spatial_rank).all(|ax| self.pool_spec.padding.valid_dim(ax))
                    && self.group == 1
                {
                    let mut patch = TypedModelPatch::default();
                    let wire = patch.tap_model(model, node.inputs[0])?;
                    let wire = self.wire_as_im2col_pair(&mut patch, &*node.name, wire, true)?;
                    patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                } else if self.group != 1 && self.group == self.output_channels() {
                    return Ok(Some(TypedModelPatch::single_unary_op(
                        model,
                        node,
                        dispatch_floatlike!(Self::to_depth_wise(dt)(self, &shape))?,
                    )?));
                } else {
                    let mut patch = TypedModelPatch::default();
                    let wire = patch.tap_model(model, node.inputs[0])?;
                    let wire = self.wire_as_im2col_pair(&mut patch, &*node.name, wire, false)?;
                    patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                }
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
