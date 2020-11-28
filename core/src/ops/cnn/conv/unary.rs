use ndarray::*;

use crate::internal::*;
use crate::model::*;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::PoolSpec;
use crate::ops::matmul;
use crate::ops::nn::{DataFormat, DataShape};
use crate::ops::quant::QParams;

use tract_linalg::frame::Packer;
use tract_linalg::mmm::FusedSpec;

use std::iter::Sum;

#[derive(Debug, Clone, new, Hash)]
pub struct ConvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub kernel: Arc<Tensor>,

    pub group: usize,

    pub bias: Option<Arc<Tensor>>,
    pub q_params: Option<QParams>,
}

impl_dyn_hash!(ConvUnary);

impl ConvUnary {
    fn input_channels(&self) -> usize {
        match self.kernel_fmt {
            KernelFormat::OIHW => self.kernel.shape()[1] * self.group,
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

    pub fn kernel_as_group_o_ihw(&self) -> TractResult<Arc<Tensor>> {
        let final_shape = [
            self.group,
            self.output_channels() / self.group,
            self.kernel.len() / self.output_channels(),
        ];
        trace!("kernel shape (group, output, rest) = {:?}", final_shape);
        let hw_rank = self.kernel.rank() - 2;
        match self.kernel_fmt {
            KernelFormat::HWIO => {
                let mut shape = self.kernel.shape().to_vec();
                shape.insert(hw_rank + 1, self.group); // HWIGO
                shape[self.pool_spec.rank()] = self.input_channels() / self.group;
                let mut kernel = self.kernel.as_ref().clone();
                kernel.set_shape(&shape)?;
                let mut permutation: Vec<usize> = vec![hw_rank + 1, hw_rank + 2, hw_rank];
                permutation.extend(0..hw_rank);
                let mut kernel = kernel.permute_axes(&permutation)?;
                kernel.set_shape(&final_shape)?;
                Ok(kernel.into_arc_tensor())
            }
            KernelFormat::OIHW => {
                Ok(self.kernel.clone().into_tensor().into_shape(&final_shape)?.into_arc_tensor())
            }
        }
    }

    fn kernel_as_packed_as(&self, packer: &Packer, m: usize) -> TractResult<ArrayD<Arc<Tensor>>> {
        let kernel = self.kernel_as_group_o_ihw()?;
        unsafe {
            let mut packed_as = Array1::from(
                (0..self.group)
                    .map(|g| {
                        let mut packed = Tensor::uninitialized_aligned_dt(
                            kernel.datum_type(),
                            &[packer.len(m)],
                            packer.alignment(),
                        )?;
                        packer.pack(
                            &mut TensorView::at_prefix(&mut packed, &[])?,
                            &kernel.view_at_prefix(&[g])?,
                            1,
                            0,
                        );
                        Ok(packed.into_arc_tensor())
                    })
                    .collect::<TractResult<Vec<_>>>()?,
            )
            .into_dyn();
            if self.group == 1 {
                packed_as.index_axis_inplace(Axis(0), 0);
            }
            if self.pool_spec.data_format.has_n() {
                packed_as.insert_axis_inplace(Axis(0));
            }
            Ok(packed_as)
        }
    }

    fn bias_as_non_linear<T>(&self) -> TractResult<Option<ArrayD<Vec<FusedSpec>>>>
    where
        T: Datum + Copy,
    {
        use crate::itertools::Itertools;
        if let Some(bias) = &self.bias {
            let bias = bias.cast_to::<T>()?;
            let bias = bias.as_slice::<T>()?;
            let mut bias = Array1::from(
                bias.iter()
                    .chunks(self.output_channels() / self.group)
                    .into_iter()
                    .map(|c| vec![FusedSpec::PerRowAdd(tensor1(&*c.cloned().collect::<Vec<_>>()))])
                    .collect::<Vec<_>>(),
            )
            .into_dyn();
            if self.group == 1 {
                bias.index_axis_inplace(Axis(0), 0);
            }
            if self.pool_spec.data_format.has_n() {
                bias.insert_axis_inplace(Axis(0));
            }
            Ok(Some(bias))
        } else {
            Ok(None)
        }
    }

    pub unsafe fn wire_as_im2col_pair(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
        b_dt: DatumType,
        direct: bool,
    ) -> TractResult<OutletId> {
        trace!("to_im2col_pair: {:?}", self);
        let (input_shape, geo, output_shape) =
            self.pool_spec.compute_geo(&*model.outlet_fact(wire)?.shape.as_concrete().unwrap())?;

        trace!("input: {:?}", input_shape);
        let a_dt = self.kernel.datum_type();

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.kernel.len() / self.output_channels();
        let n = geo.output_shape.iter().cloned().product::<usize>();

        let c_dt = self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(a_dt);
        let mut mmm = tract_linalg::ops()
            .mmm(a_dt, b_dt, c_dt, m, k, n)
            .with_context(|| format!("No multiplier for {:?}x{:?} to {:?}", a_dt, b_dt, c_dt,))?;

        if let Some(q) = self.q_params.as_ref() {
            q.inject_into_mmm(&mut *mmm)?;
        }

        trace!(
            "Gemm iters={} m={} k={} n={}",
            input_shape.n_dim().unwrap_or(&1) * self.group,
            m,
            k,
            n
        );
        trace!("{:?}", mmm);

        let b_storage = if direct {
            let channel_stride = input_shape.c_stride();
            let data_offsets: Vec<isize> = geo.centers_offsets();
            let kernel_offsets: Vec<isize> = (0..self.input_channels())
                .flat_map(|ici| {
                    geo.standard_layout_data_field
                        .iter()
                        .map(move |x| x + (ici * channel_stride) as isize)
                })
                .collect();
            mmm.b_from_data_and_offsets(&kernel_offsets, &data_offsets)
        } else {
            let c_dim = *input_shape.c_dim();
            wire = model.wire_node(
                format!("{}.im2col", name),
                Im2Col::new(
                    geo.clone(),
                    self.pool_spec.data_format.clone(),
                    m,
                    k,
                    n,
                    self.group,
                    c_dim / self.group,
                    mmm.b_pack(),
                    self.q_params
                        .as_ref()
                        .and_then(|q| q.zero_point_b.clone())
                        .map(|a| a.into_tensor())
                        .unwrap_or(Tensor::zero_dt(b_dt, &[]).unwrap()),
                )?,
                &[wire],
            )?[0];
            mmm.b_packed()
        };

        let kernels = self.kernel_as_packed_as(&mmm.a_pack(), m)?;
        let mut dims = tvec!();
        let mut strides = tvec!();
        if self.pool_spec.data_format.has_n() {
            dims.push(*output_shape.n().unwrap());
            strides.push(*output_shape.n_stride().unwrap());
        }
        if self.group > 1 {
            dims.push(self.group);
            strides.push(output_shape.c() / self.group * output_shape.c_stride());
        }
        dims.push(m);
        dims.push(n);
        strides.push(*output_shape.c_stride());
        strides.push(*output_shape.w_stride());
        debug_assert_eq!(dims.iter().product::<usize>(), output_shape.shape.iter().product());
        let fused_ops = dispatch_copy!(Self::bias_as_non_linear(mmm.internal_type())(self))?;

        wire = model.wire_node(
            format!("{}.matmatmul", name),
            matmul::lir_unary::LirMatMulUnary {
                b_storage,
                c_fact: TypedFact::dt_shape(
                    self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(a_dt),
                    output_shape.shape,
                ),
                c_shape_override: Some((Dims::from(&*dims), Dims::from(&*strides))),
                packed_as: kernels,
                fused_ops,
                mmm,
                k,
                m,
            },
            &[wire],
        )?[0];

        Ok(wire)
    }

    pub fn to_depth_wise<T>(&self, input_full_shape: &[usize]) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + PartialEq + Sum,
    {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_full_shape)?;
        let op = DepthWise::new(
            patch,
            input_shape,
            output_shape,
            self.kernel_as_group_o_ihw().context("in kernel_as_group_o_ihw")?,
            self.bias.clone(),
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
            let shape = self
                .pool_spec
                .data_format
                .shape(input_fact.shape.iter().collect::<TVec<TDim>>())?;
            let down = patch.wire_node(
                format!("{}.downsample", node.name),
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor as isize, 0),
                &[tap],
            )?;
            let id = patch.wire_node(&*node.name, new_op, &down)?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), id)?;
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
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let full_input_shape = input_fact.shape.to_tvec();
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
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
            let a = ker
                .into_shape(&a_shape)?
                .broadcast_into_rank(full_input_shape.len())?
                .into_arc_tensor();
            let trans_data = self.pool_spec.data_format == DataFormat::HWC
                || self.pool_spec.data_format == DataFormat::NHWC;
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, node.inputs[0])?;
            let output_type =
                self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(input_fact.datum_type);
            let must_split_quant =
                self.bias.is_some() && self.bias.as_ref().unwrap().datum_type() != output_type;
            let q_params = if must_split_quant {
                Some(QParams {
                    c_datum_type: self.bias.as_ref().unwrap().datum_type(),
                    zero_point_c: None,
                    scale_factor: None,
                    ..self.q_params.clone().unwrap()
                })
            } else {
                self.q_params.clone()
            };
            let op = MatMulUnary::new(a, a_trans, trans_data, trans_data);
            wire = patch.wire_node(&*node.name, op, &[wire])?[0];
            if let Some(b) = &self.bias {
                let mut bias_shape = tvec!(1; input_shape.rank());
                bias_shape[input_shape.c_axis()] = co;
                let b = b.clone().into_tensor().into_shape(&bias_shape)?;
                wire = patch.wire_node(
                    format!("{}.bias", node.name),
                    crate::ops::math::add::unary(b.into_arc_tensor()),
                    &[wire],
                )?[0];
            }
            if must_split_quant {
                use crate::ops::quant::*;
                let qp = self.q_params.as_ref().unwrap();
                let scale = qp.scale_factor.unwrap_or(1.0);
                let op = match output_type {
                    DatumType::I8 => quantize_linear_i8(
                        scale,
                        qp.zero_point_c
                            .as_ref()
                            .map(|zp| zp.to_scalar().map(|&x: &i8| x.clone()))
                            .transpose()?
                            .unwrap_or(0),
                    ),
                    DatumType::U8 => quantize_linear_u8(
                        scale,
                        qp.zero_point_c
                            .as_ref()
                            .map(|zp| zp.to_scalar().map(|&x: &u8| x.clone()))
                            .transpose()?
                            .unwrap_or(0),
                    ),
                    _ => unimplemented!("Unexpected quant type"),
                };
                wire = patch.wire_node(format!("{}.quant", node.name), op, &[wire])?[0];
            }
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }
}

impl Op for ConvUnary {
    fn name(&self) -> Cow<str> {
        "Conv".into()
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

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for ConvUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut model = TypedModel::default();
        let dt = inputs[0].datum_type();
        let wire = model.add_source("source", TypedFact::dt_shape(dt, inputs[0].shape()))?;
        let wire = unsafe {
            self.wire_as_im2col_pair(
                &mut model,
                "im2col-adhoc",
                wire,
                inputs[0].datum_type(),
                false,
            )?
        };
        model.set_output_outlets(&[wire])?;
        let plan = SimplePlan::new(model)?;
        plan.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.pool_spec.data_format.shape(&**inputs[0].shape)?.c()
            != &self.input_channels().to_dim()
        {
            bail!(
                "Inconsistent convolution: input is {:?}, kernel expects {} input channels, {:?}",
                inputs[0],
                self.input_channels(),
                self
            );
        }
        if self.pool_spec.output_channel_override != Some(self.output_channels()) {
            bail!(
                "Inconsistent convolution: output channels from pool spec is {:?}, kernel expects {} output channels, {:?}",
                self.pool_spec.output_channel_override,
                self.output_channels(),
                self
                );
        }
        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);

        if let Some(bias) = &self.bias {
            if bias.len() != self.output_channels() {
                bail!("Bias should have one value per output channel, got:{:?}", bias);
            }
        }
        if let Some(q_params) = &self.q_params {
            fact.datum_type = q_params.c_datum_type;
        }
        Ok(tvec!(fact))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = self.pool_spec.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>())?;
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
        let shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let output_dims = self.pool_spec.padding.compute(
            shape.hw_dims(),
            kernel_spatial_shape,
            &*self.pool_spec.dilations.clone().unwrap_or(tvec!(1; kernel_spatial_shape.len())),
            &*self.pool_spec.strides.clone().unwrap_or(tvec!(1; kernel_spatial_shape.len())),
        );
        let n_output_points: TDim = output_dims.iter().map(|d| d.output.clone()).maybe_product()?;
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.into_iter().product::<usize>().to_dim();
        let one = 1.to_dim();
        Ok(tvec!(
            (
                Cost::Params(inputs[0].datum_type),
                (self.kernel.len() + self.bias.as_ref().map(|b| b.len()).unwrap_or(0)).to_dim()
            ),
            (
                Cost::FMA(inputs[0].datum_type),
                shape
                    .n()
                    .unwrap_or(&one)
                    .maybe_mul(shape.c())?
                    .maybe_mul(&n_output_channels)?
                    .maybe_mul(&n_output_points)?
                    .maybe_mul(&kernel_surface)?
                    / self.group
            )
        ))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let shape = self.pool_spec.data_format.shape(full_input_shape.clone())?;
        // remove n
        if let Some(n) = shape.n_axis() {
            assert_eq!(n, 0);
            if change == &AxisOp::Rm(n) {
                let op = ConvUnary { pool_spec: self.pool_spec.dispose_n_axis(), ..self.clone() };
                return Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(op)),
                    change,
                )));
            }
            if change.transform_axis(n).map(|axis| axis > 0).unwrap_or(true) {
                return Ok(None);
            }
        }
        // format swap: chw <-> hwc
        let (new_format, axis_move) = match self.pool_spec.data_format {
            DataFormat::NCHW => {
                (DataFormat::NHWC, AxisOp::Move(shape.c_axis(), full_input_shape.len() - 1))
            }
            DataFormat::CHW => {
                (DataFormat::HWC, AxisOp::Move(shape.c_axis(), full_input_shape.len() - 1))
            }
            DataFormat::NHWC => (DataFormat::NCHW, AxisOp::Move(shape.c_axis(), 1)),
            DataFormat::HWC => (DataFormat::CHW, AxisOp::Move(shape.c_axis(), 0)),
        };
        if *change == axis_move {
            let mut new_op = self.clone();
            new_op.pool_spec.data_format = new_format;
            return Ok(Some(AxisChangeConsequence {
                substitute_op: Some(Box::new(new_op)),
                wire_changes: tvec!(
                    (InOut::In(0), change.clone()),
                    (InOut::Out(0), change.clone())
                ),
            }));
        }
        // geo axis manips
        use AxisOp::*;
        let h_axis = shape.h_axis();
        let hw_axes = shape.hw_axes();
        let kh_axis = if self.kernel_fmt == KernelFormat::OIHW { 2 } else { 0 };
        let (geo_adjusted, kernel_adjusted) = match change {
            Rm(a)
                if hw_axes.contains(a)
                    && self.pool_spec.dilation(a - h_axis) == 1
                    && self.pool_spec.stride(a - h_axis) == 1
                    && self.kernel.shape()[a - h_axis] == 1 =>
            {
                (Rm(a - h_axis), Rm(a - h_axis + kh_axis))
            }
            Add(a) if hw_axes.contains(a) => (Add(a - h_axis), Add(a - h_axis + kh_axis)),
            Move(f, t) if hw_axes.contains(f) && hw_axes.contains(t) => {
                (Move(f - h_axis, t - h_axis), Move(f - h_axis + kh_axis, t - h_axis + kh_axis))
            }
            _ => return Ok(None),
        };
        let mut kernel = self.kernel.clone().into_tensor();
        kernel_adjusted.change_tensor(&mut kernel)?;
        let mut dilations = self.pool_spec.dilations().into_owned().into();
        geo_adjusted.change_shape_array(&mut dilations)?;
        let mut kernel_shape = self.pool_spec.kernel_shape.clone();
        geo_adjusted.change_shape_array(&mut kernel_shape)?;
        let mut strides = self.pool_spec.strides().into_owned().into();
        geo_adjusted.change_shape_array(&mut strides)?;
        let new_op = ConvUnary {
            pool_spec: PoolSpec {
                data_format: self.pool_spec.data_format,
                padding: self.pool_spec.padding.clone(), // fixme (explicit padding)
                dilations: Some(dilations),
                kernel_shape,
                strides: Some(strides),
                output_channel_override: self.pool_spec.output_channel_override,
            },
            kernel_fmt: self.kernel_fmt,
            kernel: kernel.into_arc_tensor(),
            group: self.group,
            bias: self.bias.clone(),
            q_params: self.q_params.clone(),
        };
        return Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(new_op)),
            wire_changes: tvec!((InOut::In(0), change.clone()), (InOut::Out(0), change.clone())),
        }));
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
        let spatial_rank = input_shape.hw_rank();
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        if let Some(shape) = input_fact.shape.as_concrete() {
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
                    let geo_dim: TDim = input_shape.hw_dims().iter().maybe_product()?;
                    wire = patch.wire_node(
                        format!("{}.reshape", &*node.name),
                        AxisOp::Reshape(
                            input_shape.h_axis(),
                            input_shape.hw_dims().into(),
                            tvec!(geo_dim.clone()),
                        ),
                        &[wire],
                    )?[0];
                    let kernel_shape = match self.kernel_fmt {
                        KernelFormat::HWIO => &self.kernel.shape()[spatial_rank..],
                        KernelFormat::OIHW => &self.kernel.shape()[..2],
                    };
                    let operating_rank = input_fact.rank() + 1 - kernel_spatial_shape.len();
                    let kernel = self
                        .kernel
                        .as_ref()
                        .clone()
                        .into_shape(&kernel_shape)?
                        .broadcast_into_rank(operating_rank)?;
                    wire = patch.wire_node(
                        &*node.name,
                        MatMulUnary::new(
                            kernel.into_arc_tensor(),
                            self.kernel_fmt == KernelFormat::HWIO,
                            input_c_is_last,
                            input_c_is_last,
                        ),
                        &[wire],
                    )?[0];
                    if let Some(ref bias) = self.bias {
                        let bias_shape =
                            if input_c_is_last { [1, bias.len()] } else { [bias.len(), 1] };
                        let bias = bias
                            .clone()
                            .into_tensor()
                            .into_shape(&bias_shape)?
                            .broadcast_into_rank(operating_rank)?
                            .into_arc_tensor();
                        wire = patch.wire_node(
                            format!("{}.bias", node.name),
                            crate::ops::math::add::unary(bias),
                            &[wire],
                        )?[0];
                    }
                    wire = patch.wire_node(
                        &*node.name,
                        AxisOp::Reshape(
                            input_shape.h_axis(),
                            tvec!(geo_dim),
                            input_shape.hw_dims().into(),
                        ),
                        &[wire],
                    )?[0];
                    patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                } else if should_use_direct(
                    &self.pool_spec.data_format.shape(shape.into())?,
                    &self.pool_spec,
                    self.group,
                ) {
                    let mut patch = TypedModelPatch::default();
                    let wire = patch.tap_model(model, node.inputs[0])?;
                    let wire = self
                        .wire_as_im2col_pair(
                            &mut patch,
                            &*node.name,
                            wire,
                            input_fact.datum_type,
                            true,
                        )
                        .context("in wire_as_im2col_pair, direct")?;
                    patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                } else if self.group != 1
                    && self.group == self.output_channels()
                    && self.group == self.input_channels()
                {
                    let op = dispatch_floatlike!(Self::to_depth_wise(dt)(self, &shape))
                        .context("in to_depth_wise")?;
                    return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
                } else {
                    let mut patch = TypedModelPatch::default();
                    let wire = patch.tap_model(model, node.inputs[0])?;
                    let wire = self
                        .wire_as_im2col_pair(
                            &mut patch,
                            &*node.name,
                            wire,
                            input_fact.datum_type,
                            false,
                        )
                        .context("in wire_as_im2col_pair, indirect")?;
                    patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }

    as_op!();
}

fn should_use_direct(input_shape: &DataShape, pool_spec: &PoolSpec, group: usize) -> bool {
    let spatial_rank = input_shape.hw_rank();
    if group != 1 || !(0..spatial_rank).all(|ax| pool_spec.padding.valid_dim(ax)) {
        return false;
    }
    let direct =
        // no real rationale here, pure heuristic to force "right" pick in
        // both hey_snips v3 and v4. just hope this will generalize ok
        (0..spatial_rank).any(|d| pool_spec.dilation(d) > 1 && pool_spec.kernel_shape[d] > 2) ||
        // that one kind of make sense, better use direct that generate a huge
        // im2col matrix (when both kernel and input are big)
        pool_spec.kernel_shape.iter().product::<usize>() * input_shape.shape.iter().product::<usize>() > 1048576;
    direct
}

#[allow(non_snake_case)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::cnn::PaddingSpec;
    use DataFormat::{HWC, NHWC};

    #[test]
    fn conv_vs_direct_arm_ml_kws_cnn_m_0() {
        let input = NHWC.from_n_c_hw(1, 1, &[49, 10]).unwrap();
        let pool = {
            PoolSpec::new(
                HWC,
                tvec!(10, 4),
                PaddingSpec::Valid,
                Some(tvec!(1, 1)),
                Some(tvec!(1, 1)),
                Some(64),
            )
        };
        assert!(!should_use_direct(&input, &pool, 1));
    }

    #[test]
    fn conv_vs_direct_arm_ml_kws_cnn_m_1() {
        let input = NHWC.from_n_c_hw(1, 64, &[40, 7]).unwrap();
        let pool = {
            PoolSpec::new(
                HWC,
                tvec!(10, 4),
                PaddingSpec::Valid,
                Some(tvec!(2, 1)),
                Some(tvec!(1, 1)),
                Some(48),
            )
        };
        assert!(should_use_direct(&input, &pool, 1));
    }

    #[test]
    fn conv_vs_direct_hey_snips_v31() {
        use crate::ops::cnn::PaddingSpec;
        use DataFormat::HWC;

        fn dil_use_direct(size: usize, d: usize) -> bool {
            let input = HWC.from_n_c_hw(1, 128, &[size]).unwrap();
            let pool = {
                PoolSpec::new(
                    HWC,
                    tvec!(2),
                    PaddingSpec::Valid,
                    Some(tvec!(d)),
                    Some(tvec!(1)),
                    Some(64),
                )
            };
            should_use_direct(&input, &pool, 1)
        }
        assert!(!dil_use_direct(36, 1));
        assert!(!dil_use_direct(33, 2));
        assert!(!dil_use_direct(27, 4));
        assert!(!dil_use_direct(18, 8));
    }

    #[test]
    fn conv_vs_direct_hey_snips_v4() {
        fn dil_use_direct(d: usize) -> bool {
            let input = HWC.from_n_c_hw(1, 16, &[8 + 2 * d]).unwrap();
            let pool = {
                PoolSpec::new(
                    HWC,
                    tvec!(3),
                    PaddingSpec::Valid,
                    Some(tvec!(d)),
                    Some(tvec!(1)),
                    Some(64),
                )
            };
            should_use_direct(&input, &pool, 1)
        }
        assert!(!dil_use_direct(1));
        assert!(dil_use_direct(2));
        assert!(dil_use_direct(4));
        assert!(dil_use_direct(8));
    }

    #[test]
    fn conv_vs_direct_am_lda_2M() {
        let pool = {
            PoolSpec::new(
                HWC,
                tvec!(5),
                PaddingSpec::Valid,
                Some(tvec!(1)),
                Some(tvec!(1)),
                Some(200),
            )
        };
        let input = HWC.from_n_c_hw(1, 40, &[28]).unwrap();
        assert!(!should_use_direct(&input, &pool, 1));
    }

    #[test]
    fn conv_vs_direct_hey_am_tdnn_2M() {
        fn use_direct(size: usize, stride: usize) -> bool {
            let input = HWC.from_n_c_hw(1, 256, &[size]).unwrap();
            let pool = {
                PoolSpec::new(
                    HWC,
                    tvec!(3),
                    PaddingSpec::Valid,
                    Some(tvec!(1)),
                    Some(tvec!(stride)),
                    Some(64),
                )
            };
            should_use_direct(&input, &pool, 1)
        }
        assert!(!use_direct(26, 1)); // tdnn2
        assert!(!use_direct(24, 3)); // tdnn3
        assert!(!use_direct(10, 1)); // tdnn4,5
    }
}
