use ndarray::*;

use crate::internal::*;
use crate::model::*;
use crate::ops;
use crate::ops::matmul::mir_quant::wire_offset_u8_as_i8;
use crate::ops::matmul::mir_quant::QParamKind;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};
use crate::ops::matmul::lir_unary::{
    ConcreteMatMulGeometry, LirMatMulUnary, MatMulGeometry, ProtoFusedSpec, SymbolicMatMulGeometry,
};
use crate::ops::matmul::MatMulQParams;
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape};

use tract_linalg::frame::Packer;
use tract_linalg::mmm::MatMatMul;

use std::iter::Sum;

#[derive(Debug, Clone, new, Hash)]
pub struct ConvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub kernel: Arc<Tensor>,

    pub group: usize,

    pub bias: Option<Arc<Tensor>>,

    pub q_params: Option<(DatumType, MatMulQParams)>,
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
        self.kernel_fmt.kernel_as_group_o_ihw(
            &self.kernel,
            self.group,
            self.input_channels(),
            self.output_channels(),
        )
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

    fn kernel_offset_u8_as_i8(
        &self,
        inputs: &mut [OutletId],
        model: &mut TypedModel,
    ) -> TractResult<Option<Self>> {
        if let DatumType::U8 = self.kernel.datum_type().unquantized() {
            let new_op = Self {
                kernel: self.kernel.offset_u8_as_i8(),
                q_params: self
                    .q_params
                    .as_ref()
                    .map(|(dt, qp)| -> TractResult<_> {
                        let a0 = match &qp.a0 {
                            QParamKind::Attr(_) | QParamKind::FromQType => {
                                qp.a0.offset_u8_as_i8(&model, &[])?
                            }
                            QParamKind::FromInput(i) => {
                                match model.outlet_fact(inputs[*i])?.datum_type.unquantized() {
                                    DatumType::U8 => {
                                        inputs[*i] = model.wire_node(
                                            format!(
                                                "{}.offset_{}_as_i8",
                                                model.node(inputs[*i].node).name,
                                                "a0"
                                            ),
                                            ops::quant::offset_u8_as_i8(),
                                            &[inputs[*i]],
                                        )?[0];
                                    }
                                    DatumType::I32 => {
                                        inputs[*i] = model.wire_node(
                                            format!(
                                                "{}.offset_{}_as_i8",
                                                model.node(inputs[*i].node).name,
                                                "a0"
                                            ),
                                            ops::math::add::unary(rctensor0(-128i32)),
                                            &[inputs[*i]],
                                        )?[0];
                                    }
                                    _ => (),
                                }
                                QParamKind::FromInput(*i)
                            }
                        };
                        Ok((*dt, MatMulQParams { a0, ..qp.clone() }))
                    })
                    .transpose()?,
                ..self.clone()
            };
            return Ok(Some(new_op));
        } else {
            Ok(None)
        }
    }

    fn bias_as_non_linear<T>(&self) -> TractResult<ArrayD<Vec<ProtoFusedSpec>>>
    where
        T: Datum + Copy,
    {
        let mut ops = Array1::from_elem(self.group, vec![]);

        if let Some(bias) = &self.bias {
            let bias = bias.cast_to::<T>()?;
            let bias = bias.as_slice::<T>()?;
            ops.iter_mut().zip(bias.chunks(self.output_channels() / self.group)).for_each(
                |(ops, bias)| {
                    ops.push(ProtoFusedSpec::BinPerRow(
                        rctensor1(bias).into(),
                        tract_linalg::mmm::BinOp::Add,
                    ));
                },
            )
        }
        let mut ops = ops.into_dyn();

        if self.group == 1 {
            ops.index_axis_inplace(Axis(0), 0);
        }
        if self.pool_spec.data_format.has_n() {
            ops.insert_axis_inplace(Axis(0));
        }
        Ok(ops)
    }

    pub unsafe fn wire_as_quant_im2col(
        &self,
        model: &mut TypedModel,
        name: &str,
        b_dt: DatumType,
        wires: &[OutletId],
    ) -> TractResult<OutletId> {
        use crate::ops::matmul::mir_quant as qmm;

        let c_dt = self.q_params.as_ref().unwrap().0;

        let params = self.q_params.as_ref().unwrap().1.as_outlet_ids(
            model,
            name,
            &wires,
            self.kernel.datum_type(),
            b_dt,
            c_dt,
        )?;

        let a0 = params[0];
        let a_scale = params[1];
        let mut b0 = params[2];
        let b_scale = params[3];
        let c0 = params[4];
        let c_scale = params[5];

        let b = wire_offset_u8_as_i8(model, name, wires[0], "b", &mut b0, "b0")?;
        let b_fact = model.outlet_fact(b)?.clone();
        let (_, m, k, n, mmm) = self.compute_geo(&b_fact)?;
        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;

        let abc_scale = qmm::combine_scales(model, name, a_scale, b_scale, c_scale)?;

        let im2col = model.wire_node(
            format!("{}.im2col", name),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[b, b0],
        )?[0];

        let a = self.kernel_as_group_o_ihw()?.into_tensor();
        let a = a.cast_to_dt(i32::datum_type())?;
        let a = a.to_array_view::<i32>()?;
        let mut sum_a = a.sum_axis(Axis(a.ndim() - 1));
        if self.group == 1 {
            sum_a.index_axis_inplace(Axis(0), 0);
        }

        if self.pool_spec.data_format.has_n() {
            sum_a.insert_axis_inplace(Axis(0));
        }
        let sum_a = model.add_const(format!("{}.sum_a", name), sum_a)?;

        let mut sum_b = model.wire_node(
            format!("{}.sum_b", name),
            super::QSumB { n: n.clone(), r: mmm.b_pack(k).panel_width(), k },
            &[im2col],
        )?[0];

        if self.group > 1 && self.pool_spec.data_format.c_is_last() {
            let has_n = self.pool_spec.data_format.has_n() as usize;
            sum_b = model.wire_node(
                format!("{}.transpose_sum_b", name),
                AxisOp::Move(has_n, 1 + has_n),
                &[sum_b],
            )?[0];
        }

        let b_dt = model.outlet_fact(b)?.datum_type;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let mut geometry = MatMulGeometry::from(SymbolicMatMulGeometry {
            b_datum_type: b_dt,
            m: m.to_dim(),
            k: k.to_dim(),
            n: n.clone(),
            mmm: mmm.clone(),
        });
        if n.to_usize().is_ok() {
            geometry = geometry.optimize_if(Some(&SymbolValues::default()))?;
        }
        let wire = self.wire_lir_matmatmul(
            model,
            name,
            im2col,
            mmm,
            i32::datum_type(),
            mmm_output_shape.clone().into(),
            m,
            k,
            geometry.into(),
            c_axis,
            h_axis,
        )?;
        let has_n = self.pool_spec.data_format.has_n() as usize;
        let has_group = (self.group > 1) as usize;
        let (m_axis, n_axis) = if self.pool_spec.data_format.c_is_last() {
            (1 + has_group + has_n, has_n)
        } else {
            (has_group + has_n, 1 + has_n + has_group)
        };
        let wire = qmm::compensate_zero_points(
            model,
            name,
            wire,
            k.to_dim(),
            a0,
            b0,
            sum_a,
            sum_b,
            m_axis,
            n_axis,
        )?;

        let mut wire = qmm::requant(model, name, wire, c_dt, abc_scale, c0)?;
        if self.group > 1 {
            wire = model.wire_node(
                format!("{}.reshape_group", name),
                AxisOp::Reshape(
                    c_axis - 1,
                    mmm_output_shape[c_axis - 1..][..2].iter().map(|d| d.to_dim()).collect(),
                    tvec!((m * self.group).to_dim()),
                ),
                &[wire],
            )?[0];
        }
        let wire = Self::wire_geo_reshape(model, name, wire, &output_shape)?;
        Ok(wire)
    }

    pub unsafe fn wire_as_im2col_pair(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
    ) -> TractResult<OutletId> {
        let b_fact = model.outlet_fact(wire)?.clone();
        let b_dt = b_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_fact.datum_type);

        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;
        let (_, m, k, n, mmm) = self.compute_geo(model.outlet_fact(wire)?)?;
        let padding = model.add_const(format!("{}.b0", name), Tensor::zero_dt(b_dt, &[])?)?;

        wire = model.wire_node(
            format!("{}.im2col", name),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[wire, padding],
        )?[0];

        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let mut geometry = MatMulGeometry::from(SymbolicMatMulGeometry {
            b_datum_type: b_dt,
            m: m.to_dim(),
            k: k.to_dim(),
            n: n.clone(),
            mmm: mmm.clone(),
        });
        if n.to_usize().is_ok() {
            geometry = geometry.optimize_if(Some(&SymbolValues::default()))?;
        }
        let mut wire = self.wire_lir_matmatmul(
            model,
            name,
            wire,
            mmm,
            c_dt,
            mmm_output_shape.clone().into(),
            m.to_usize().unwrap(),
            k.to_usize().unwrap(),
            geometry,
            c_axis,
            h_axis,
        )?;

        if self.group > 1 {
            wire = model.wire_node(
                format!("{}.reshape_group", name),
                AxisOp::Reshape(
                    c_axis - 1,
                    mmm_output_shape[c_axis - 1..][..2].iter().map(|d| d.to_dim()).collect(),
                    tvec!((m * self.group).to_dim()),
                ),
                &[wire],
            )?[0];
        }
        let wire = Self::wire_geo_reshape(model, name, wire, &output_shape)?;
        Ok(wire)
    }

    fn mmm_output_shape<D: DimLike>(
        &self,
        output_shape: &BaseDataShape<D, TVec<D>>,
    ) -> TractResult<(TVec<D>, usize, usize)> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        let shape: BaseDataShape<D, TVec<D>> = output_shape.fmt.from_n_c_hw(
            output_shape.n().cloned().unwrap_or(1.into()),
            output_shape.c().clone(),
            tvec!(geo_collapsed_out),
        )?;
        let mut mmm_output_shape: TVec<D> = shape.shape.clone();
        let mut c_axis = shape.c_axis();
        let mut h_axis = shape.h_axis();
        if self.group > 1 {
            mmm_output_shape[shape.c_axis()] =
                mmm_output_shape[shape.c_axis()].clone() / self.group;
            mmm_output_shape.insert(shape.c_axis(), self.group.into());
            if self.group > 1 {
                if h_axis > c_axis {
                    h_axis += 1;
                }
                c_axis += 1;
            }
        }
        Ok((mmm_output_shape, c_axis, h_axis))
    }

    fn wire_geo_reshape<D: DimLike>(
        model: &mut TypedModel,
        name: &str,
        wire: OutletId,
        output_shape: &BaseDataShape<D, TVec<D>>,
    ) -> TractResult<OutletId> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        let wire = model.wire_node(
            name,
            AxisOp::Reshape(
                output_shape.h_axis(),
                tvec!(geo_collapsed_out.to_dim()),
                output_shape.hw_dims().iter().map(|d| d.to_dim()).collect(),
            ),
            &[wire],
        )?;
        Ok(wire[0])
    }

    pub unsafe fn wire_as_direct(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: OutletId,
        input_shape: &[usize],
    ) -> TractResult<OutletId> {
        let b_fact = model.outlet_fact(wire)?.clone();
        let c_dt = crate::ops::matmul::output_type(b_fact.datum_type);
        let (geo, m, k, n, mmm) = self.compute_geo(&b_fact)?;
        let geo = geo.to_concrete(input_shape)?;
        let input_shape: DataShape = self.pool_spec.data_format.shape(input_shape.into())?;
        let c_stride = input_shape.c_stride();
        let data_offsets: Vec<isize> = geo.patch.centers_offsets();
        let kernel_offsets: Vec<isize> = (0..self.input_channels())
            .flat_map(|ici| {
                geo.patch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| x + (ici * c_stride) as isize)
            })
            .collect();
        let b_storage = mmm.b_from_data_and_offsets(
            b_fact.datum_type.size_of(),
            &kernel_offsets,
            &data_offsets,
        );
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&geo.output_shape)?;

        let geometry = MatMulGeometry::Concrete(ConcreteMatMulGeometry {
            m,
            k,
            n: n.to_usize().unwrap(),
            b_storage,
        });
        let wire = self.wire_lir_matmatmul(
            model,
            name,
            wire,
            mmm,
            c_dt,
            mmm_output_shape.into(),
            m.to_usize().unwrap(),
            k,
            geometry,
            c_axis,
            h_axis,
        )?;

        let wire = Self::wire_geo_reshape(model, name, wire, &geo.output_shape)?;
        Ok(wire)
    }

    fn compute_geo(
        &self,
        input_fact: &TypedFact,
    ) -> TractResult<(PoolGeometry, usize, usize, TDim, Box<dyn MatMatMul>)> {
        let a_dt = self.kernel.datum_type();
        let b_dt = input_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_dt);

        let geo = self.pool_spec.compute_geo(&input_fact.shape)?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.kernel.len() / self.output_channels();
        let n: TDim =
            self.pool_spec.output_shape(&input_fact.shape)?.hw_dims().iter().cloned().product();

        let mmm = tract_linalg::ops()
            .mmm(a_dt, b_dt, c_dt, Some(m), Some(k), n.to_usize().ok())
            .with_context(|| format!("No multiplier for {:?}x{:?} to {:?}", a_dt, b_dt, c_dt,))?;

        Ok((geo, m, k, n, mmm))
    }

    fn wire_lir_matmatmul(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: OutletId,
        mmm: Box<dyn MatMatMul>,
        c_datum_type: DatumType,
        mmm_output_shape: ShapeFact,
        m: usize,
        k: usize,
        geometry: MatMulGeometry,
        c_m_axis: usize,
        c_n_axis: usize,
    ) -> TractResult<OutletId> {
        let kernels = self.kernel_as_packed_as(&mmm.a_pack(k), m)?;
        let shape = kernels.shape();
        let mut fused_ops = dispatch_copy!(Self::bias_as_non_linear(mmm.internal_type())(self))?;
        for fo in &mut fused_ops {
            fo.push(ProtoFusedSpec::Store);
        }
        let mut iter = kernels.iter().cloned().zip(fused_ops.iter().cloned());
        let micro_ops = ArrayD::from_shape_fn(shape, |_| iter.next().unwrap());

        let wire = model.wire_node(
            format!("{}.matmatmul", name),
            LirMatMulUnary {
                c_fact: TypedFact::dt_shape(c_datum_type, mmm_output_shape.clone()),
                micro_ops,
                c_m_axis,
                c_n_axis,
                c_final_shape: mmm_output_shape.into(),
                reshape_post: vec![],
                geometry: geometry.into(),
                mmm,
            },
            &[wire],
        )?[0];
        Ok(wire)
    }

    pub fn to_depth_wise<T>(&self, input: &TypedFact) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + PartialEq + Sum,
    {
        let input_shape = input.shape.as_concrete().unwrap();
        let ConcretePoolGeometry { input_shape, patch, output_shape } =
            self.pool_spec.compute_geo(&input.shape)?.to_concrete(input_shape)?.into_owned();
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
        use crate::ops::matmul::*;
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
            let mut patch = TypedModelPatch::new("declutter_as_matmul");
            let a = patch.add_const(format!("{}.filters", &node.name), a)?;
            let mut inputs = node
                .inputs
                .iter()
                .map(|i| patch.tap_model(model, *i))
                .collect::<TractResult<TVec<_>>>()?;
            inputs.insert(0, a);
            // in Q case, the bias has to be injected inside the QMatMul (as it
            // must be added before requantization)
            let wire = if let Some(q_params) = &self.q_params {
                let mut params = q_params.1.clone();
                params.insert_input(0); // kernel as input
                params.insert_input(2); // bias as input
                let mut bias = self.bias.clone().unwrap_or(rctensor0(0i32));
                if bias.len() > 1 {
                    let shape = if trans_data { [1, bias.len()] } else { [bias.len(), 1] };
                    bias = bias.into_tensor().into_shape(&shape)?.into_arc_tensor();
                }
                let bias = patch.add_const(format!("{}.bias", &node.name), bias)?;
                inputs.insert(2, bias);
                let op = QMatMul {
                    a_trans,
                    b_trans: trans_data,
                    c_trans: trans_data,
                    output_type: q_params.0,
                    params: q_params.1.clone(),
                };
                patch.wire_node(&*node.name, op, &inputs)?[0]
            } else {
                let op = MatMul { a_trans, b_trans: trans_data, c_trans: trans_data };
                let mut wire = patch.wire_node(format!("{}.matmul", node.name), op, &inputs)?[0];
                if let Some(b) = self.bias.as_ref().filter(|_| self.q_params.is_none()) {
                    let mut bias_shape = tvec!(1; input_shape.rank());
                    bias_shape[input_shape.c_axis()] = co;
                    let b = b.clone().into_tensor().into_shape(&bias_shape)?;
                    wire = patch.wire_node(
                        format!("{}.bias", node.name),
                        crate::ops::math::add::unary(b.into_arc_tensor()),
                        &[wire],
                    )?[0];
                }
                wire
            };
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
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
        Ok(info)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
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

        let mut wires: TVec<OutletId> = inputs
            .iter()
            .enumerate()
            .map(|(ix, v)| {
                model.add_source(
                    format!("source.{}", ix),
                    TypedFact::dt_shape(v.datum_type(), v.shape()),
                )
            })
            .collect::<TractResult<_>>()?;
        let new_op = self.kernel_offset_u8_as_i8(&mut wires, &mut model)?;
        let wire = unsafe {
            if self.q_params.is_some() {
                let op_ref = if let Some(op) = new_op.as_ref() { op } else { self };
                op_ref.wire_as_quant_im2col(
                    &mut model,
                    "im2col-adhoc",
                    inputs[0].datum_type(),
                    &*wires,
                )?
            } else {
                self.wire_as_im2col_pair(&mut model, "im2col-adhoc", wires[0])?
            }
        };
        model.set_output_outlets(&[wire])?;
        let plan = SimplePlan::new(model)?;
        plan.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let q_inputs = self.q_params.as_ref().map(|(_, qp)| qp.input_count()).unwrap_or(0);
        if inputs.len() != 1 + q_inputs {
            bail!("Wrong number of inputs: expected {} got {}", 1 + q_inputs, inputs.len());
        }
        if self.pool_spec.data_format.shape(&*inputs[0].shape)?.c()
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
        if let Some(bias) = &self.bias {
            if bias.len() != self.output_channels() {
                bail!("Bias should have one value per output channel, got:{:?}", bias);
            }
        }

        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);
        if let Some((dt, _qp)) = self.q_params.as_ref() {
            fact.datum_type = *dt;
        }
        Ok(tvec!(fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let fact = &inputs[0];
        let shape = self.pool_spec.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>())?;
        let mut axes = vec![];
        if let Some(n_axis) = shape.n_axis() {
            let mut info = AxisInfo::simple(n_axis).disposable(true);
            info.inputs.extend(std::iter::repeat(None).take(inputs.len() - 1));
            axes.push(info);
        }
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let h_axis = shape.h_axis();
        for (ix, &dim) in kernel_spatial_shape.iter().enumerate() {
            if dim == 1 && self.pool_spec.stride(ix) == 1 {
                let mut info = AxisInfo::simple(ix + h_axis).disposable(true);
                info.inputs.extend(std::iter::repeat(None).take(inputs.len() - 1));
                axes.push(info)
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some((_, qp)) = self.q_params.as_ref() {
            if let Some((inputs, qp)) = qp.inline_static(model, node)? {
                let mut op = self.clone();
                op.q_params.as_mut().unwrap().1 = qp;
                let patch = TypedModelPatch::replace_single_op(model, node, &inputs, op)?
                    .with_context("inlining quantized conv params");
                return Ok(Some(patch));
            }
        }
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
        let n_output_points: TDim =
            output_dims.iter().map(|d| d.convoluted.clone()).product::<TDim>();
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.into_iter().product::<usize>().to_dim();
        let one = 1.to_dim();
        Ok(tvec!(
            (
                Cost::Params(inputs[0].datum_type.unquantized()),
                (self.kernel.len() + self.bias.as_ref().map(|b| b.len()).unwrap_or(0)).to_dim()
            ),
            (
                Cost::FMA(inputs[0].datum_type),
                shape.n().cloned().unwrap_or(one)
                    * shape.c()
                    * n_output_channels
                    * n_output_points
                    * kernel_surface
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
        kernel_adjusted.change_tensor(&mut kernel, false)?;
        let mut dilations = self.pool_spec.dilations().into_owned().into();
        geo_adjusted.change_shape_array(&mut dilations, false)?;
        let mut kernel_shape = self.pool_spec.kernel_shape.clone();
        geo_adjusted.change_shape_array(&mut kernel_shape, false)?;
        let mut strides = self.pool_spec.strides().into_owned().into();
        geo_adjusted.change_shape_array(&mut strides, false)?;
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
        if let DatumType::U8 = self.kernel.datum_type().unquantized() {
            let mut patch = TypedModelPatch::default();
            let mut inputs = node
                .inputs
                .iter()
                .map(|w| patch.tap_model(model, *w))
                .collect::<TractResult<TVec<_>>>()?;
            let new_op = self.kernel_offset_u8_as_i8(&mut inputs, &mut patch)?.unwrap();
            let wire = patch.wire_node(&node.name, new_op, &inputs)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            return Ok(Some(patch));
        }

        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
        let spatial_rank = input_shape.hw_rank();
        let kernel_spatial_shape = &self.kernel.shape()[self.kernel_fmt.h_axis()..][..spatial_rank];
        unsafe {
            let dt = input_fact.datum_type;
            if self.q_params.is_some() {
                let mut patch = TypedModelPatch::default();
                let inputs = node
                    .inputs
                    .iter()
                    .map(|w| patch.tap_model(model, *w))
                    .collect::<TractResult<TVec<_>>>()?;
                let wire = self.wire_as_quant_im2col(
                    &mut patch,
                    &node.name,
                    model.node_input_facts(node.id)?[0].datum_type,
                    &inputs,
                )?;
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            } else if kernel_spatial_shape.iter().product::<usize>() == 1
                && (0..spatial_rank)
                    .all(|i| self.pool_spec.stride(i) == 1 && self.pool_spec.dilation(i) == 1)
                && self.group == 1
            {
                use crate::ops::matmul::MatMulUnary;
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                let input_c_is_last = input_shape.c_axis() == input_shape.rank() - 1;
                let geo_dim: TDim = input_shape.hw_dims().iter().product();
                wire = patch.wire_node(
                    format!("{}.reshape_input", &*node.name),
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
                    &format!("{}.matmul", &node.name),
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
            } else if input_fact
                .shape
                .as_concrete()
                .map(|s| {
                    should_use_direct(
                        &self.pool_spec.data_format.shape(s.into()).unwrap(),
                        &self.pool_spec,
                        self.group,
                    )
                })
                .unwrap_or(false)
            {
                let mut patch = TypedModelPatch::default();
                let wire = patch.tap_model(model, node.inputs[0])?;
                let wire = self
                    .wire_as_direct(
                        &mut patch,
                        &*node.name,
                        wire,
                        input_fact.shape.as_concrete().unwrap(),
                    )
                    .context("in wire_as_direct")?;
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            } else if self.group != 1
                && self.group == self.output_channels()
                && self.group == self.input_channels()
                && input_fact.shape.as_concrete().is_some()
            {
                let op = dispatch_floatlike!(Self::to_depth_wise(dt)(self, &input_fact))
                    .context("in to_depth_wise")?;
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
            } else {
                let mut patch = TypedModelPatch::default();
                let wire = patch.tap_model(model, node.inputs[0])?;
                let wire = self
                    .wire_as_im2col_pair(&mut patch, &*node.name, wire)
                    .context("in wire_as_im2col_pair")?;
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
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
    use DataFormat::*;

    #[test]
    fn onnx_basic_convinteger() {
        let op = ConvUnary {
            pool_spec: PoolSpec {
                data_format: NCHW,
                kernel_shape: tvec!(2, 2),
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: None,
                output_channel_override: Some(1),
            },
            kernel_fmt: KernelFormat::OIHW,
            kernel: rctensor4(&[[[[1u8, 1], [1, 1]]]]),
            group: 1,
            bias: None,
            q_params: Some((i32::datum_type(), MatMulQParams::all_dynamic(1))),
        };
        let input = tvec!(
            rctensor4(&[[[[1u8, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            rctensor0(0u8),
            rctensor0(1.0f32),
            rctensor0(1u8),
            rctensor0(1.0f32),
            rctensor0(0i32),
            rctensor0(1.0f32),
        );
        let output = op.eval(input).unwrap();
        assert_eq!(&*output[0], &tensor4(&[[[[8i32, 12], [20, 24]]]]));
    }

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
