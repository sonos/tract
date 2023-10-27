use ndarray::*;
use num_integer::Integer;
use tract_data::itertools::izip;
use tract_linalg::mmm::InputStoreSpec;
use tract_num_traits::Zero;

use crate::internal::*;
use crate::model::*;
use crate::ops;
use crate::ops::array::Pad;
use crate::ops::array::PadMode;
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::cast;
use crate::ops::cnn::PaddingSpec::*;
use crate::ops::einsum::EinSum;
use crate::ops::math::Add;
use crate::ops::math::Div;
use crate::ops::math::Mul;
use crate::ops::math::Sub;
use crate::ops::matmul::lir_unary::AddMatMulGeometry;
use crate::ops::matmul::lir_unary::MapOutputAxisToInput;
use crate::ops::matmul::mir_quant::wire_offset_u8_as_i8;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};
use crate::ops::matmul::lir_unary::{LirMatMulUnary, ProtoFusedSpec};
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

    pub q_params: Option<DatumType>,
}

impl ConvUnary {
    fn input_channels(&self) -> usize {
        self.kernel_fmt.input_channels(self.kernel.shape(), self.group)
    }

    fn output_channels(&self) -> usize {
        self.kernel_fmt.output_channels(self.kernel.shape(), self.group)
    }

    pub fn kernel_as_group_o_ihw(&self) -> TractResult<Arc<Tensor>> {
        self.kernel_fmt.kernel_as_group_o_ihw(
            &self.kernel,
            self.group,
            self.input_channels(),
            self.output_channels(),
        )
    }

    // shape is g,packed
    fn kernel_as_packed_as(&self, packer: &Packer, k: usize, m: usize) -> TractResult<Arc<Tensor>> {
        let kernel = self.kernel_as_group_o_ihw()?;
        unsafe {
            let packed = Integer::next_multiple_of(
                &packer.len(k, m),
                &(packer.alignment() / kernel.datum_type().size_of()),
            );
            let packed = Tensor::uninitialized_aligned_dt(
                kernel.datum_type(),
                &[self.group, packed],
                packer.alignment(),
            )?;
            for g in 0..self.group {
                packer.pack(
                    &mut TensorView::at_prefix(&packed, &[g])?,
                    &kernel.view_at_prefix(&[g])?,
                    1,
                    0,
                );
            }
            Ok(packed.into_arc_tensor())
        }
    }

    pub fn kernel_offset_u8_as_i8(
        &self,
        inputs: &mut [OutletId],
        model: &mut TypedModel,
    ) -> TractResult<Option<Self>> {
        ensure!(self.q_params.is_some());
        if let DatumType::U8 = self.kernel.datum_type().unquantized() {
            let new_op = Self { kernel: self.kernel.offset_u8_as_i8(), ..self.clone() };
            let name = format!("{}.a0_as_i8", model.node(inputs[1].node).name);
            match model.outlet_fact(inputs[1])?.datum_type.unquantized() {
                DatumType::U8 => {
                    inputs[1] =
                        model.wire_node(name, ops::quant::offset_u8_as_i8(), &[inputs[1]])?[0];
                }
                DatumType::I32 | DatumType::I8 => {
                    inputs[1] = model.wire_node(
                        format!("{name}.cast"),
                        cast(i32::datum_type()),
                        &[inputs[1]],
                    )?[0];
                    let cst = model.add_const(format!("{name}.cst"), tensor0(-128i32))?;
                    inputs[1] = model.wire_node(name, ops::math::add(), &[inputs[1], cst])?[0];
                }
                _ => (),
            };
            Ok(Some(new_op))
        } else {
            Ok(None)
        }
    }

    // group,bias
    fn bias_as_non_linear<T>(
        &self,
        c_group_axis: usize,
    ) -> TractResult<Option<(ProtoFusedSpec, Tensor)>>
    where
        T: Datum + Copy + Zero,
    {
        use tract_linalg::mmm::BinOp::Add;
        if let Some(bias) = &self.bias {
            if let Some(uni) = bias.as_uniform() {
                if uni == Tensor::zero_scalar::<T>()? {
                    Ok(None)
                } else {
                    Ok(Some((ProtoFusedSpec::BinScalar(2, Add), uni)))
                }
            } else {
                let bias = bias
                    .clone()
                    .into_tensor()
                    .into_shape(&[self.group, bias.len() / self.group])?;
                Ok(Some((
                    ProtoFusedSpec::BinPerRow(
                        2,
                        Add,
                        MapOutputAxisToInput(tvec!((c_group_axis, 0))),
                    ),
                    bias,
                )))
            }
        } else {
            Ok(None)
        }
    }

    pub unsafe fn wire_as_quant_im2col(
        &self,
        model: &mut TypedModel,
        name: &str,
        wires: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        ensure!(self.q_params.is_some());
        use crate::ops::matmul::mir_quant as qmm;

        let c_dt = self.q_params.unwrap();
        let [a0, mut a_scale, mut b0, b_scale, c0, c_scale] = wires[1..] else {
            bail!("Wrong number of inputs")
        };
        let b = wire_offset_u8_as_i8(model, name, wires[0], "b", &mut b0, "b0")?;
        let b_fact = model.outlet_fact(b)?.clone();
        let (_, m, k, n, mmm) = self.compute_geo(&b_fact)?;
        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;

        if !model.outlet_fact(a_scale)?.shape.volume().is_one() {
            // requant is performed before geo_reshape, so we need at most one geo axis to the
            // right
            if !output_shape.fmt.c_is_last() {
                a_scale = model.wire_node(
                    format!("{name}.a_scale_axis_fix"),
                    AxisOp::Add(1),
                    &[a_scale],
                )?[0];
            }
        }

        let abc_scale = qmm::combine_scales(model, name, a_scale, b_scale, c_scale)?;

        let im2col = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[b, b0],
        )?[0];

        let a = self.kernel_as_group_o_ihw()?.into_tensor();
        let a = a.cast_to_dt(i32::datum_type())?;
        let a = a.to_array_view::<i32>()?;
        let sum_a = a.sum_axis(Axis(a.ndim() - 1));
        let mut sum_a_shape: TVec<usize> = sum_a.shape().into();
        // align sum_A from G,C to "C" shape: N,HW,G,C (or N,G,C,HW)
        sum_a_shape.insert(0, 1);
        if self.pool_spec.data_format.c_is_last() {
            sum_a_shape.insert(1, 1);
        } else {
            sum_a_shape.push(1)
        }
        let sum_a = sum_a.into_shape(&*sum_a_shape)?;
        let sum_a = model.add_const(format!("{name}.sum_a"), sum_a)?;

        let mut sum_b = model.wire_node(
            format!("{name}.sum_b"),
            super::QSumB { n, r: mmm.b_pack().panel_width(), k },
            &[im2col],
        )?;
        // sum_b is N,G,HW. make it N,HW,G,C or N,G,C,HW
        sum_b = model.wire_node(format!("{name}.add_c"), AxisOp::Add(2), &sum_b)?;
        if self.pool_spec.data_format.c_is_last() {
            sum_b =
                model.wire_node(format!("{name}.transpose_sum_b"), AxisOp::Move(3, 1), &sum_b)?;
        }

        let b_dt = model.outlet_fact(b)?.datum_type;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let b_storage = unsafe { mmm.b_packed(b_dt.size_of(), k) };
        let wire = self.wire_lir_matmatmul(
            model,
            name,
            &[im2col],
            mmm,
            i32::datum_type(),
            mmm_output_shape.clone().into(),
            m,
            k,
            c_axis,
            h_axis,
            b_storage,
        )?;

        let wire =
            qmm::compensate_zero_points(model, name, wire[0], k.to_dim(), a0, b0, sum_a, sum_b[0])?;

        let wire = self.wire_remove_group(model, name, &[wire], &mmm_output_shape, c_axis)?;
        let wire = self.wire_rm_n_if_needed(model, name, &wire)?;
        let wire = qmm::requant(model, name, wire[0], c_dt, abc_scale, c0)?;
        Self::wire_geo_reshape(model, name, &[wire], &output_shape)
    }

    pub fn wire_remove_group<D: DimLike>(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
        mmm_output_shape: &[D],
        c_axis: usize,
    ) -> TractResult<TVec<OutletId>> {
        let m = &mmm_output_shape[c_axis];
        let op = if self.group == 1 {
            AxisOp::Rm(c_axis - 1)
        } else {
            AxisOp::Reshape(
                c_axis - 1,
                tvec!(self.group.to_dim(), m.to_dim()),
                tvec!(m.to_dim() * self.group),
            )
        };
        model.wire_node(format!("{name}.reshape_group"), op, wire)
    }
    pub unsafe fn wire_as_im2col_pair(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let b_fact = model.outlet_fact(wire[0])?.clone();
        let b_dt = b_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_fact.datum_type);

        let (_, m, k, _, mmm) = self.compute_geo(model.outlet_fact(wire[0])?)?;
        let geo_output_shape = self.pool_spec.output_shape(&b_fact.shape)?;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&geo_output_shape)?;

        let padding = model.add_const(format!("{name}.b0"), Tensor::zero_dt(b_dt, &[])?)?;
        let wire = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[wire[0], padding],
        )?;

        let b_storage = unsafe { mmm.b_packed(b_dt.size_of(), k) };
        let wire = self.wire_lir_matmatmul(
            model,
            name,
            &wire,
            mmm,
            c_dt,
            mmm_output_shape.clone().into(),
            m.to_usize().unwrap(),
            k.to_usize().unwrap(),
            c_axis,
            h_axis,
            b_storage,
        )?;

        let wire = self.wire_remove_group(model, name, &wire, &mmm_output_shape, c_axis)?;
        let wire = self.wire_rm_n_if_needed(model, name, &wire)?;
        Self::wire_geo_reshape(model, name, &wire, &geo_output_shape)
    }

    // always have N and G. G is right before C, c_axis point to C, c_axis-1 points to G
    fn mmm_output_shape<D: DimLike>(
        &self,
        output_shape: &BaseDataShape<D, TVec<D>>,
    ) -> TractResult<(TVec<D>, usize, usize)> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        let shape: BaseDataShape<D, TVec<D>> = output_shape.fmt.with_n().from_n_c_hw(
            output_shape.n().cloned().unwrap_or_else(|| 1.into()),
            output_shape.c().clone(),
            tvec!(geo_collapsed_out),
        )?;
        let mut mmm_output_shape: TVec<D> = shape.shape.clone();
        let mut c_axis = shape.c_axis();
        let mut h_axis = shape.h_axis();
        mmm_output_shape[shape.c_axis()] = mmm_output_shape[c_axis].clone() / self.group;
        mmm_output_shape.insert(c_axis, self.group.into());
        if h_axis > c_axis {
            h_axis += 1;
        }
        c_axis += 1;
        Ok((mmm_output_shape, c_axis, h_axis))
    }

    fn wire_rm_n_if_needed(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if self.pool_spec.data_format.has_n() {
            Ok(wire.into())
        } else {
            model.wire_node(format!("{name}.rm_n"), AxisOp::Rm(0), wire)
        }
    }

    fn wire_geo_reshape<D: DimLike>(
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
        output_shape: &BaseDataShape<D, TVec<D>>,
    ) -> TractResult<TVec<OutletId>> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        model
            .wire_node(
                name,
                AxisOp::Reshape(
                    output_shape.h_axis(),
                    tvec!(geo_collapsed_out.to_dim()),
                    output_shape.hw_dims().iter().map(|d| d.to_dim()).collect(),
                ),
                wire,
            )
            .context("in wire_geo_reshape")
    }

    pub unsafe fn wire_as_lazy_im2col(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        let mut b_fact = model.outlet_fact(wire)?.clone();
        let (geo, m, k, _, mmm) = self.compute_geo(&b_fact)?;
        let input_shape = b_fact.shape.as_concrete().unwrap().to_vec();
        let mut geo = geo.to_concrete(&input_shape)?.into_owned();
        let mut input_shape: DataShape = self.pool_spec.data_format.shape(input_shape.into())?;
        let padding = self.pool_spec.computed_padding(input_shape.hw_dims());
        if padding.iter().any(|axis| axis.pad_before != 0 || axis.pad_after != 0) {
            let mut pads = vec![(0, 0); b_fact.rank()];
            for (ix, ax) in padding.iter().enumerate() {
                pads[input_shape.h_axis() + ix] = (ax.pad_before, ax.pad_after);
            }
            let op = crate::ops::array::Pad {
                mode: crate::ops::array::PadMode::Constant(
                    Tensor::zero_scalar_dt(b_fact.datum_type)?.into_arc_tensor(),
                ),
                pads,
            };
            wire = model.wire_node(format!("{name}.pad"), op, &[wire])?[0];
            let valid_pool_spec = PoolSpec { padding: Valid, ..self.pool_spec.clone() };
            b_fact = model.outlet_fact(wire)?.clone();
            let concrete_shape = b_fact.shape.as_concrete().unwrap();
            input_shape = valid_pool_spec.data_format.shape(concrete_shape.into())?;
            geo = valid_pool_spec
                .compute_geo(&b_fact.shape)?
                .to_concrete(concrete_shape)?
                .into_owned();
        }
        let c_dt = crate::ops::matmul::output_type(b_fact.datum_type);
        let c_stride = input_shape.c_stride();
        let size_of_b = b_fact.datum_type.size_of() as isize;
        let n_bytes_offsets: Vec<isize> =
            geo.patch.centers_offsets().into_iter().map(|x| x * size_of_b).collect();
        let k_bytes_offsets: Vec<isize> = (0..self.input_channels())
            .flat_map(|ici| {
                geo.patch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| (x + (ici * c_stride) as isize) * size_of_b)
            })
            .collect();
        let virtual_input = super::lazy_im2col::LazyIm2colSpec { n_bytes_offsets, k_bytes_offsets };
        let b_storage = mmm.b_virtual_input(Box::new(virtual_input), k);
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&geo.output_shape)?;

        let wire = self.wire_lir_matmatmul(
            model,
            name,
            &[wire],
            mmm,
            c_dt,
            mmm_output_shape.clone().into(),
            m.to_usize().unwrap(),
            k,
            c_axis,
            h_axis,
            b_storage,
        )?;

        let wire = self.wire_remove_group(model, name, &wire, &mmm_output_shape, c_axis)?;
        let wire = self.wire_rm_n_if_needed(model, name, &wire)?;
        Self::wire_geo_reshape(model, name, &wire, &geo.output_shape)
    }

    #[allow(clippy::type_complexity)]
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
            .with_context(|| format!("No multiplier for {a_dt:?}x{b_dt:?} to {c_dt:?}",))?;

        Ok((geo, m, k, n, mmm))
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_lir_matmatmul(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
        mmm: Box<dyn MatMatMul>,
        c_datum_type: DatumType,
        mmm_output_shape: ShapeFact,
        m: usize,
        k: usize,
        c_m_axis: usize,
        c_n_axis: usize,
        b_storage: InputStoreSpec,
    ) -> TractResult<TVec<OutletId>> {
        let kernels = self.kernel_as_packed_as(&mmm.a_pack(), k, m)?;
        let a_storage = unsafe { mmm.a_packed(self.kernel.datum_type().size_of(), k) };
        let (mut c_to_a_axis_mapping, mut c_to_b_axis_mapping) = (tvec!(), tvec!());

        c_to_a_axis_mapping.push((c_m_axis - 1, 0)); // Group
        c_to_b_axis_mapping.push((0, 0)); // Batch
        c_to_b_axis_mapping.push((c_m_axis - 1, 1)); // Group

        let geo = AddMatMulGeometry {
            k: k.to_dim(),
            a_storage: Some(a_storage),
            b_storage: Some(b_storage),
            mmm: mmm.clone(),
            c_to_a_axis_mapping: MapOutputAxisToInput(c_to_a_axis_mapping),
            c_to_b_axis_mapping: MapOutputAxisToInput(c_to_b_axis_mapping),
        };
        let mut wires: TVec<OutletId> = wire.into();
        let kernels = model.add_const(format!("{name}.kernels"), kernels)?;
        wires.push(kernels);
        let mut ops: Vec<ProtoFusedSpec> = vec![ProtoFusedSpec::AddMatMul(geo, 1, 0)];
        if let Some((fused, tensor)) =
            dispatch_numbers!(Self::bias_as_non_linear(mmm.internal_type())(self, c_m_axis - 1))?
        {
            let bias = model.add_const(format!("{name}.bias"), tensor)?;
            wires.push(bias);
            ops.push(fused);
        }
        ops.push(ProtoFusedSpec::Store(unsafe { mmm.c_view(c_m_axis, c_n_axis) }));
        model.wire_node(
            format!("{name}.matmatmul"),
            LirMatMulUnary::new(mmm, c_datum_type.fact(mmm_output_shape), c_m_axis, c_n_axis, ops)?,
            &wires,
        )
    }

    pub fn to_depth_wise<T>(&self, input: &TypedFact) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + PartialEq + Sum,
    {
        let input_shape = input.shape.as_concrete().unwrap();
        let ConcretePoolGeometry { input_shape, patch, output_shape } =
            self.pool_spec.compute_geo(&input.shape)?.to_concrete(input_shape)?.into_owned();
        let bias = if let Some(b) = &self.bias {
            b.clone()
        } else {
            Tensor::zero::<T>(&[*input_shape.c()])?.into_arc_tensor()
        };
        let op = DepthWise::new(
            patch,
            input_shape,
            output_shape,
            self.kernel_as_group_o_ihw().context("in kernel_as_group_o_ihw")?,
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
            self.pool_spec.stride(ax) > 1
                && self.pool_spec.padding.valid_dim(ax, self.pool_spec.stride(ax) == 1)
                && (self.pool_spec.kernel_shape[ax] == 1
                    || self.pool_spec.dilation(ax) % self.pool_spec.stride(ax) == 0)
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
                format!("{}.downsample.{}", node.name, axis),
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor as isize, 0),
                &[tap],
            )?;
            let id = patch.wire_node(&*node.name, new_op, &down)?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), id)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn declutter_as_einsum(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let (input_facts, output_facts) = model.node_facts(node.id)?;
        let full_input_shape = input_facts[0].shape.to_tvec();
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
        if self.group == 1
            && self.pool_spec.strides().iter().all(|s| *s == 1)
            && self.pool_spec.dilations().iter().all(|d| *d == 1)
            && self.kernel.len() == self.input_channels() * self.output_channels()
            && self
                .pool_spec
                .computed_padding(input_shape.hw_dims())
                .iter()
                .all(|pad| pad.pad_after.is_zero() && pad.pad_before.is_zero())
        {
            let name = &node.name;
            let ci = self.input_channels();
            let co = self.output_channels();
            let ker = self.kernel.clone().into_tensor();
            let a_shape = if self.kernel_fmt == KernelFormat::HWIO { [ci, co] } else { [co, ci] };
            let a = ker.into_shape(&a_shape)?.into_arc_tensor();
            let mut patch = TypedModelPatch::new("declutter_as_einsum");
            let a = patch.add_const(format!("{name}.filters"), a)?;
            let mut inputs = patch.taps(model, &node.inputs)?;
            inputs.insert(0, a);
            let mut axes = self.axes_mapping(&input_facts, &output_facts)?.with_extra_input(0)?;
            axes = axes.with_extra_axis('0', InOut::In(0), 0)?.with_extra_axis(
                '1',
                InOut::In(0),
                1,
            )?;
            if self.kernel_fmt == KernelFormat::HWIO {
                axes = axes.linking('I', '0')?.linking('O', '1')?;
            } else {
                axes = axes.linking('I', '1')?.linking('O', '0')?;
            }
            let wire = if self.q_params.is_some() {
                let bias = self.bias.clone().unwrap_or_else(|| rctensor0(0i32));
                anyhow::ensure!(bias.rank() == 0 || bias.rank() == 1);
                axes = axes.with_extra_input(2)?;
                if bias.rank() == 1 {
                    axes = axes.with_extra_axis('$', InOut::In(2), 0)?.linking('O', '$')?;
                }
                let bias = patch.add_const(format!("{name}.bias"), bias)?;
                inputs.insert(2, bias);
                let op = EinSum { axes, operating_dt: i32::datum_type(), q_params: self.q_params };
                patch.wire_node(format!("{}.einsum", node.name), op, &inputs)?[0]
            } else {
                let op = EinSum { axes, operating_dt: input_facts[0].datum_type, q_params: None };
                let mut wire = patch.wire_node(format!("{}.einsum", node.name), op, &inputs)?[0];
                if let Some(b) = self.bias.as_ref().filter(|_| self.q_params.is_none()) {
                    anyhow::ensure!(b.rank() == 0 || b.rank() == 1);
                    let mut bias_shape = tvec!(1; input_shape.rank());
                    bias_shape[input_shape.c_axis()] = co;
                    let b = b.clone().into_tensor().into_shape(&bias_shape)?;
                    let b =
                        patch.add_const(format!("{}.bias.cst", node.name), b.into_arc_tensor())?;
                    wire = patch.wire_node(
                        format!("{}.bias", node.name),
                        crate::ops::math::add(),
                        &[wire, b],
                    )?[0];
                }
                wire
            };
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn declutter_precursor_padding(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if matches!(self.pool_spec.padding, ExplicitOnnxPool(_, _, _) | SameLower | SameUpper) {
            return Ok(None);
        }
        let prec = model.node(node.inputs[0].node);
        let pad = if let Some(pad) = prec.op_as::<Pad>() { pad } else { return Ok(None) };
        let value = if let PadMode::Constant(c) = &pad.mode {
            c
        } else {
            return Ok(None);
        };
        let shape = self.pool_spec.data_format.shape(&model.outlet_fact(node.inputs[0])?.shape)?;
        if !value.is_zero()?
            || (self.pool_spec.data_format.has_n() && pad.pads[0] != (0, 0))
            || pad.pads[shape.c_axis()] != (0, 0)
        {
            return Ok(None);
        }
        let mut before: TVec<usize> = pad.pads[shape.hw_axes()].iter().map(|pair| pair.0).collect();
        let mut after: TVec<usize> = pad.pads[shape.hw_axes()].iter().map(|pair| pair.1).collect();
        if let Explicit(bef, aft) = &self.pool_spec.padding {
            izip!(&mut before, bef).for_each(|(pad, cv)| *pad += cv);
            izip!(&mut after, aft).for_each(|(pad, cv)| *pad += cv);
        }
        let padding = Explicit(before, after);
        let mut new = self.clone();
        new.pool_spec.padding = padding;
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.taps(model, &node.inputs)?;
        wire[0] = patch.tap_model(model, prec.inputs[0])?;
        let wire = patch.wire_node(&node.name, new, &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    }

    fn declutter_channel_arithmetic_succ(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.q_params.is_some() || self.group != 1 {
            return Ok(None);
        }
        let &[succ] = &*node.outputs[0].successors else { return Ok(None) };
        let Some(bin) = model.node(succ.node).op_as::<TypedBinOp>() else { return Ok(None) };
        let other_input = model.node(succ.node).inputs[1 - succ.slot];
        let other_fact = &model.outlet_fact(other_input)?;
        let Some(konst) = &other_fact.konst else { return Ok(None) };
        let axes_mapping = model.node_axes_mapping(succ.node)?;
        let input_shape =
            self.pool_spec.data_format.shape(&model.outlet_fact(node.inputs[0])?.shape)?;
        let conv_c_axis = input_shape.c_axis();
        let &[konst_c_axis] =
            &*axes_mapping.axis((InOut::In(succ.slot), conv_c_axis))?.inputs[1 - succ.slot]
        else {
            return Ok(None);
        };
        let Ok(co) = node.outputs[0].fact.shape[conv_c_axis].to_usize() else { return Ok(None) };
        let operand_for_bias = if konst.shape()[konst_c_axis] == co && konst.len() == co {
            konst.clone().into_tensor().into_shape(&[co])?
        } else if konst.len() == 1 {
            konst.clone().to_scalar_tensor()?.broadcast_scalar_to_shape(&[co])?
        } else {
            return Ok(None);
        };
        let mut bias = if let Some(b) = &self.bias {
            b.clone()
        } else {
            Tensor::zero_dt(other_fact.datum_type, &[co])?.into_arc_tensor()
        };
        let mut kernel = self.kernel.clone();
        let mut operand_shape_for_kernel = tvec!(1; 2 + input_shape.hw_rank());
        let o_axis = if self.kernel_fmt == KernelFormat::OIHW { 0 } else { self.kernel.rank() - 1 };
        operand_shape_for_kernel[o_axis] = co;
        let operand_for_kernel = operand_for_bias.clone().into_shape(&operand_shape_for_kernel)?;
        if bin.0.is::<Sub>() && succ.slot == 0 {
            bias = (bias.into_tensor().into_array::<f32>()?
                - operand_for_bias.to_array_view::<f32>()?)
            .into_arc_tensor()
        } else if bin.0.is::<Div>() && succ.slot == 0 {
            bias = (bias.into_tensor().into_array::<f32>()?
                / operand_for_bias.to_array_view::<f32>()?)
            .into_arc_tensor();
            kernel = (kernel.into_tensor().into_array::<f32>()?
                / operand_for_kernel.to_array_view::<f32>()?)
            .into_arc_tensor();
        } else if bin.0.is::<Add>() {
            bias = (bias.into_tensor().into_array::<f32>()?
                + operand_for_bias.to_array_view::<f32>()?)
            .into_arc_tensor();
        } else if bin.0.is::<Mul>() {
            bias = (bias.into_tensor().into_array::<f32>()?
                * operand_for_bias.to_array_view::<f32>()?)
            .into_arc_tensor();
            kernel = (kernel.into_tensor().into_array::<f32>()?
                * operand_for_kernel.to_array_view::<f32>()?)
            .into_arc_tensor();
        } else {
            return Ok(None);
        };
        let new_op = ConvUnary { bias: Some(bias), kernel, ..self.clone() };
        let mut patch = TypedModelPatch::default();
        let wire = patch.tap_model(model, node.inputs[0])?;
        let wire = patch.wire_node(&node.name, new_op, &[wire])?[0];
        patch.shunt_outside(model, succ.node.into(), wire)?;
        Ok(Some(patch))
    }
}

impl Op for ConvUnary {
    fn name(&self) -> Cow<str> {
        "ConvUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.pool_spec.info();
        info.push(format!(
            "Kernel {:?} (groups:{}), {:?}",
            self.kernel_fmt, self.group, self.kernel
        ));
        if let Some(b) = &self.bias {
            info.push(format!("Bias: {b:?}"))
        }
        Ok(info)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for ConvUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut model = TypedModel::default();
        let mut wire: TVec<OutletId> = inputs
            .iter()
            .enumerate()
            .map(|(ix, v)| model.add_source(format!("source.{ix}"), v.datum_type().fact(v.shape())))
            .collect::<TractResult<_>>()?;
        let wire = unsafe {
            if self.q_params.is_some() {
                let new_op = self.kernel_offset_u8_as_i8(&mut wire, &mut model)?;
                let op_ref = if let Some(op) = new_op.as_ref() { op } else { self };
                op_ref.wire_as_quant_im2col(&mut model, "im2col-adhoc", &wire)?
            } else {
                self.wire_as_im2col_pair(&mut model, "im2col-adhoc", &wire)?
            }
        };
        model.set_output_outlets(&wire)?;
        model.into_runnable()?.run(inputs)
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let q_inputs = if self.q_params.is_some() { 6 } else { 0 };
        if inputs.len() != 1 + q_inputs {
            bail!("Wrong number of inputs: expected {} got {}", 1 + q_inputs, inputs.len());
        }
        ensure!(self.pool_spec.rank() == self.kernel.rank() - 2);
        if self.pool_spec.data_format.shape(&*inputs[0].shape)?.c()
            != &self.input_channels().to_dim()
        {
            bail!(
                    "Inconsistent convolution: input is {:?}, but kernel expects {} input channels.\n{:?}",
                    inputs[0],
                    self.input_channels(),
                    self
                    );
        }
        if self.pool_spec.output_channel_override != Some(self.output_channels()) {
            bail!(
                "Inconsistent convolution: output channels from pool spec is {:?}, kernel expects {} output channels.\n{:?}",
                self.pool_spec.output_channel_override,
                self.output_channels(),
                self
                );
        }
        if let ExplicitOnnxPool(bef, after, _) | Explicit(bef, after) = &self.pool_spec.padding {
            anyhow::ensure!(bef.len() == self.pool_spec.rank());
            anyhow::ensure!(after.len() == self.pool_spec.rank());
        }
        if let Some(bias) = &self.bias {
            ensure!(
                bias.rank() == 0 || (bias.rank() == 1 && bias.len() == self.output_channels()),
                "Bias should be scalar or a vector with one value per output channel, got:{:?}",
                bias
            );
        }
        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);
        if let Some(dt) = self.q_params {
            fact.datum_type = dt;
        } else {
            ensure!(
                inputs[0].datum_type == self.kernel.datum_type(),
                "Convolution input and weights must have the same type. (resp {:?} and {:?})",
                inputs[0].datum_type,
                self.kernel.datum_type(),
            )
        }
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let fact = &inputs[0];
        let shape = self.pool_spec.data_format.shape(fact.shape.iter().collect::<Vec<TDim>>())?;
        let mut axes = AxesMapping::disconnected(inputs, outputs)?
            .renaming((InOut::In(0), shape.c_axis()), 'I')?
            .renaming((InOut::Out(0), shape.c_axis()), 'O')?;
        if let Some(n_axis) = shape.n_axis() {
            axes = axes
                .renaming((InOut::In(0), n_axis), 'N')?
                .linking('N', (InOut::Out(0), n_axis))?;
        }
        let h_axis = shape.h_axis();
        let geo = "HWXYZ".chars().chain('a'..);
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let padding = self.pool_spec.computed_padding(shape.hw_dims());
        for ((ix, &dim), repr) in kernel_spatial_shape.iter().enumerate().zip(geo) {
            if dim == 1
                && self.pool_spec.dilation(ix) == 1
                && self.pool_spec.stride(ix) == 1
                && padding[ix].pad_before.is_zero()
                && padding[ix].pad_after.is_zero()
            {
                axes = axes
                    .renaming((InOut::In(0), ix + h_axis), repr)?
                    .linking(repr, (InOut::Out(0), ix + h_axis))?;
            }
        }
        if self.q_params.is_some() {
            for qp_ix in 0..6 {
                if inputs[qp_ix + 1].rank() == 1 {
                    axes = match qp_ix {
                        0 | 1 => axes.linking('O', (InOut::In(qp_ix + 1), 0))?,
                        2 | 3 => axes.linking('I', (InOut::In(qp_ix + 1), 0))?,
                        4 | 5 => axes.linking('O', (InOut::In(qp_ix + 1), 0))?,
                        _ => unreachable!(),
                    };
                }
            }
        }
        Ok(axes)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        macro_rules! pass {
            ($func:ident) => {
                if let Some(mut r) = self.$func(model, node).context(stringify!($func))? {
                    trace!(stringify!($func));
                    r.push_context(stringify!($func));
                    return Ok(Some(r));
                }
            };
        }
        pass!(declutter_stride_slice_to_downsample);
        pass!(declutter_as_einsum);
        pass!(declutter_channel_arithmetic_succ);
        pass!(declutter_precursor_padding);
        Ok(None)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        let kernel_spatial_shape =
            &self.kernel.shape()[self.kernel_fmt.h_axis()..][..shape.hw_rank()];
        let output_dims = self.pool_spec.padding.compute(
            shape.hw_dims(),
            kernel_spatial_shape,
            &self
                .pool_spec
                .dilations
                .clone()
                .unwrap_or_else(|| tvec!(1; kernel_spatial_shape.len())),
            &self.pool_spec.strides.clone().unwrap_or_else(|| tvec!(1; kernel_spatial_shape.len())),
        );
        let n_output_points: TDim =
            output_dims.iter().map(|d| d.convoluted.clone()).product::<TDim>();
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.iter().product::<usize>().to_dim();
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
        let kh_axis = self.kernel_fmt.h_axis();
        let (geo_adjusted, kernel_adjusted) = match change {
            Rm(a)
                if hw_axes.contains(a)
                    && hw_axes.len() > 1
                    && self.pool_spec.dilation(a - h_axis) == 1
                    && self.pool_spec.stride(a - h_axis) == 1
                    && self.pool_spec.kernel_shape[a - h_axis] == 1 =>
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
        let pool_spec = self.pool_spec.change_geo_axes(&geo_adjusted)?;
        let new_op = ConvUnary {
            pool_spec,
            kernel_fmt: self.kernel_fmt,
            kernel: kernel.into_arc_tensor(),
            group: self.group,
            bias: self.bias.clone(),
            q_params: self.q_params,
        };
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(new_op)),
            wire_changes: tvec!((InOut::In(0), change.clone()), (InOut::Out(0), change.clone())),
        }))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let DatumType::U8 = self.kernel.datum_type().unquantized() {
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.taps(model, &node.inputs)?;
            let new_op = self.kernel_offset_u8_as_i8(&mut wire, &mut patch)?.unwrap();
            let wire = patch.wire_node(&node.name, new_op, &wire)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            patch.obliterate(node.id)?;
            return Ok(Some(patch.with_context("kernel-u8-to-i8")));
        }

        let input_fact = model.outlet_fact(node.inputs[0])?;
        unsafe {
            let dt = input_fact.datum_type;
            if self.q_params.is_some() {
                let mut patch = TypedModelPatch::default();
                let inputs = patch.taps(model, &node.inputs)?;
                let wire = self
                    .wire_as_quant_im2col(&mut patch, &node.name, &inputs)
                    .context("in wire_as_quant_im2col")?;
                patch.shunt_outside(model, node.id.into(), wire[0])?;
                patch.obliterate(node.id)?;
                Ok(Some(patch.with_context("quantized-codegen")))
            } else if input_fact
                .shape
                .as_concrete()
                .map(|s| {
                    should_use_lazy(
                        &self.pool_spec.data_format.shape(s.into()).unwrap(),
                        &self.pool_spec,
                        self.group,
                    )
                })
                .unwrap_or(false)
            {
                let mut patch = TypedModelPatch::new("wire_as_lazy_im2col");
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                wire = self.wire_as_lazy_im2col(&mut patch, &node.name, wire)?[0];
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                patch.obliterate(node.id)?;
                Ok(Some(patch))
            } else if self.group != 1
                && self.group == self.output_channels()
                && self.group == self.input_channels()
                && input_fact.shape.as_concrete().is_some()
            {
                let op = dispatch_floatlike!(Self::to_depth_wise(dt)(self, input_fact))
                    .context("in to_depth_wise")?;
                Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?))
            } else {
                let mut patch = TypedModelPatch::default();
                let wire = patch.tap_model(model, node.inputs[0])?;
                let wire = self
                    .wire_as_im2col_pair(&mut patch, &node.name, &[wire])
                    .context("in wire_as_im2col_pair")?[0];
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                patch.obliterate(node.id)?;
                Ok(Some(patch))
            }
        }
    }

    as_op!();
}

fn should_use_lazy(_input_shape: &DataShape, pool_spec: &PoolSpec, group: usize) -> bool {
    group == 1 && pool_spec.kernel_shape.iter().product::<usize>() > 5
}

#[allow(non_snake_case)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::array::Pad;
    use DataFormat::*;

    #[test]
    fn onnx_basic_convinteger() {
        let op = ConvUnary {
            pool_spec: PoolSpec {
                data_format: NCHW,
                kernel_shape: tvec!(2, 2),
                padding: Valid,
                dilations: None,
                strides: None,
                output_channel_override: Some(1),
            },
            kernel_fmt: KernelFormat::OIHW,
            kernel: rctensor4(&[[[[1u8, 1], [1, 1]]]]),
            group: 1,
            bias: None,
            q_params: Some(i32::datum_type()),
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
        let input = input.into_iter().map(IntoTValue::into_tvalue).collect::<TVec<_>>();
        let output = op.eval(input).unwrap();
        assert_eq!(*output[0], tensor4(&[[[[8i32, 12], [20, 24]]]]));
    }

    #[test]
    fn valid_conv_absorbs_precursor_pad() -> TractResult<()> {
        let mut model = TypedModel::default();
        let wire = tvec!(model.add_source("source", f32::fact(dims!(1, 10)))?);
        let wire = model.wire_node(
            "pad",
            Pad {
                pads: vec![(0, 0), (1, 0)],
                mode: ops::array::PadMode::Constant(rctensor0(0f32)),
            },
            &wire,
        )?;
        let wire = model.wire_node(
            "conv",
            ConvUnary {
                pool_spec: PoolSpec {
                    data_format: crate::ops::nn::DataFormat::CHW,
                    dilations: None,
                    strides: None,
                    kernel_shape: tvec![2],
                    padding: Explicit(tvec![0], tvec![0]),
                    output_channel_override: Some(1),
                },
                kernel_fmt: crate::ops::cnn::KernelFormat::OIHW,
                kernel: rctensor3(&[[[1f32, 2f32]]]),
                group: 1,
                bias: None,
                q_params: None,
            },
            &wire,
        )?;
        model.set_output_outlets(&wire)?;
        model.declutter()?;
        assert_eq!(model.nodes().len(), 2); // source + conv
        let cv = model.nodes()[1].op_as::<ConvUnary>().unwrap();
        assert_eq!(cv.pool_spec.padding, Explicit(tvec![1], tvec![0])); // source + conv
        Ok(())
    }
}
