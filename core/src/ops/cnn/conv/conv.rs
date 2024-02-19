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
use crate::ops::cnn::wire_reshape_bias_for_bin;
use crate::ops::cnn::PaddingSpec::*;
use crate::ops::einsum::EinSum;
use crate::ops::math::{add, div, mul, sub};
use crate::ops::math::{Add, Div, Mul, Sub};
use crate::ops::matmul::lir_unary::AddMatMulGeometry;
use crate::ops::matmul::lir_unary::MapOutputAxisToInput;
use crate::ops::matmul::mir_quant::wire_ensure_q8_flavour;
use crate::ops::matmul::pack::MatMatMulPack;
use crate::ops::nn::Reduce;

use super::depth_wise::DepthWise;
use super::im2col::Im2Col;
use super::lazy_im2col::LazyIm2colSpec;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};
use crate::ops::matmul::lir_unary::{LirMatMulUnary, ProtoFusedSpec};
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape};

use tract_linalg::frame::Packer;
use tract_linalg::mmm::MatMatMul;

#[derive(Debug, Clone, new, Hash)]
pub struct Conv {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub group: usize,
    // None -> floats
    // Some(I32) -> output is I32 (use quantized kernels, but output will be i32). last 2 Q inputs
    // are ignored
    // Some(QXX) -> quantized XX, but parameters are ignored (I8, U8, or I32) in favor of last 2 Q inputs
    pub q_params: Option<DatumType>,
}

impl Conv {
    pub fn input_channels(&self) -> usize {
        self.pool_spec.input_channels
    }

    pub fn output_channels(&self) -> usize {
        self.pool_spec.output_channels
    }

    pub fn wire_kernel_as_g_o_ihw(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut kernel: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(kernel)?;
        for (ix, op) in self
            .kernel_fmt
            .kernel_as_group_o_ihw_ops(&fact.shape, self.group)
            .into_iter()
            .enumerate()
        {
            kernel = model.wire_node(format!("{name}.prep_kernel.{ix}"), op, &[kernel])?[0];
        }
        Ok(tvec!(kernel))
    }

    fn wire_pack_g_o_ihw(
        &self,
        model: &mut TypedModel,
        name: &str,
        packer: Packer,
        mut kernel: OutletId,
    ) -> TractResult<OutletId> {
        let kernel_shape = &model.outlet_fact(kernel)?.shape;
        let output_shape_fact = MatMatMulPack::output_shape(kernel_shape, &packer, 1, 2);
        kernel = model.wire_node(
            format!("{name}.prep_kernel.pack"),
            MatMatMulPack { packer, k_axis: 2, mn_axis: 1, output_shape_fact },
            &[kernel],
        )?[0];
        Ok(kernel)
    }

    // group,bias
    fn wire_bias_as_non_linear(
        &self,
        model: &mut TypedModel,
        name: &str,
        bias: OutletId,
        c_group_axis: usize,
    ) -> TractResult<(ProtoFusedSpec, OutletId)> {
        use tract_linalg::mmm::BinOp::Add;
        let fact = model.outlet_fact(bias)?;
        if fact.shape.volume().is_one() || fact.uniform.is_some() {
            Ok((ProtoFusedSpec::BinScalar(2, Add), bias))
        } else {
            let bias = AxisOp::wire_split_axis(
                model,
                &format!("{name}.reformat_bias"),
                bias,
                0,
                self.group,
            )?[0];
            let pfs =
                ProtoFusedSpec::BinPerRow(2, Add, MapOutputAxisToInput(tvec!((c_group_axis, 0))));
            Ok((pfs, bias))
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
        let &[mut x, mut kernel, bias, mut x0, x_scale, mut k0, mut k_scale, y0, y_scale] = wires
        else {
            bail!("Wrong number of inputs")
        };
        wire_ensure_q8_flavour(model, name, &mut kernel, "k", &mut k0, i8::datum_type())?;
        wire_ensure_q8_flavour(model, name, &mut x, "x", &mut x0, i8::datum_type())?;

        let a_fact = model.outlet_fact(kernel)?.clone();
        let b_fact = model.outlet_fact(x)?.clone();

        let (_, _, k, n, mmm) = self.compute_geo(&a_fact, &b_fact)?;
        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;

        if !model.outlet_fact(k_scale)?.shape.volume().is_one() {
            // requant is performed before geo_reshape, so we need at most one geo axis to the
            // right
            if !output_shape.fmt.c_is_last() {
                k_scale = model.wire_node(
                    format!("{name}.a_scale_axis_fix"),
                    AxisOp::Add(1),
                    &[k_scale],
                )?[0];
            }
        }

        let abc_scale = qmm::combine_scales(model, name, k_scale, x_scale, y_scale)?;

        let im2col = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[x, x0],
        )?[0];

        let g_o_ihw = self.wire_kernel_as_g_o_ihw(model, name, kernel)?;
        let g_o_ihw_as_i32 =
            model.wire_node(format!("{name}.kernel_as_i32"), cast(i32::datum_type()), &g_o_ihw)?;
        let sum_ker_g_c_k = model.wire_node(
            format!("{name}.sum_ker_g_c_k"),
            Reduce::new(tvec!(2), ops::nn::Reducer::Sum),
            &g_o_ihw_as_i32,
        )?;
        let sum_ker_a_g_c =
            model.wire_node(format!("{name}.rm_k"), AxisOp::Rm(2), &sum_ker_g_c_k)?;
        // align sum_A from G,C to "C" shape: N,HW,G,C (or N,G,C,HW)
        let sum_ker_n_g_c =
            model.wire_node(format!("{name}.sum_ker_n_g_c"), AxisOp::Add(0), &sum_ker_a_g_c)?;
        let hw_position = if self.pool_spec.data_format.c_is_last() { 1 } else { 3 };
        let sum_ker = model.wire_node(
            format!("{name}.sum_ker_n_g_c"),
            AxisOp::Add(hw_position),
            &sum_ker_n_g_c,
        )?;

        let mut sum_x = model.wire_node(
            format!("{name}.sum_x"),
            super::QSumB { n, r: mmm.b_pack().panel_width(), k },
            &[im2col],
        )?;
        // sum_b is N,G,HW. make it N,HW,G,C or N,G,C,HW
        sum_x = model.wire_node(format!("{name}.add_c"), AxisOp::Add(2), &sum_x)?;
        if self.pool_spec.data_format.c_is_last() {
            sum_x =
                model.wire_node(format!("{name}.transpose_sum_b"), AxisOp::Move(3, 1), &sum_x)?;
        }

        let x_dt = model.outlet_fact(x)?.datum_type;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let b_storage = unsafe { mmm.b_packed(x_dt.size_of(), k) };
        let bias =
            model.wire_node(format!("{name}.cast_bias"), cast(mmm.internal_type()), &[bias])?[0];
        let wire = self.wire_mm_weights_bias(
            model,
            name,
            im2col,
            g_o_ihw[0],
            bias,
            mmm,
            i32::datum_type(),
            mmm_output_shape.clone().into(),
            k,
            c_axis,
            h_axis,
            b_storage,
        )?;

        let wire = qmm::compensate_zero_points(
            model,
            name,
            wire[0],
            k.to_dim(),
            k0,
            x0,
            sum_ker[0],
            sum_x[0],
        )?;

        let wire = self.wire_remove_group(model, name, &[wire], &mmm_output_shape, c_axis)?;
        let wire = self.wire_rm_n_if_needed(model, name, &wire)?;
        let wire = qmm::requant(model, name, wire[0], c_dt, abc_scale, y0)?;
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
        let &[x, kernel, bias] = wire else { bail!("Wrong number of inputs") };
        let x_fact = model.outlet_fact(x)?.clone();
        let k_fact = model.outlet_fact(kernel)?.clone();
        let b_dt = x_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(x_fact.datum_type);

        let (_, _, k, _, mmm) = self.compute_geo(&k_fact, &x_fact)?;
        let geo_output_shape = self.pool_spec.output_shape(&x_fact.shape)?;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&geo_output_shape)?;

        let padding = model.add_const(format!("{name}.b0"), Tensor::zero_scalar_dt(b_dt)?)?;

        let mut wire: TVec<_> = wire.into();
        wire[0] = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &x_fact.shape, mmm.clone())?,
            &[wire[0], padding],
        )?[0];

        let b_storage = unsafe { mmm.b_packed(b_dt.size_of(), k) };

        let g_o_ihw = self.wire_kernel_as_g_o_ihw(model, name, wire[1])?;

        let wire = self
            .wire_mm_weights_bias(
                model,
                name,
                wire[0],
                g_o_ihw[0],
                bias,
                mmm,
                c_dt,
                mmm_output_shape.clone().into(),
                k.to_usize().unwrap(),
                c_axis,
                h_axis,
                b_storage,
            )
            .context("in wire_lir_matmatmul")?;

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
        wire: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let &[mut x, kernel, bias] = wire else { bail!("Wrong number of inputs") };
        let mut x_fact = model.outlet_fact(x)?.clone();
        let k_fact = model.outlet_fact(kernel)?.clone();
        let (geo, m, k, n, mmm) = self.compute_geo(&k_fact, &x_fact)?;
        debug!("{name} as lazy_im2col: m={m} k={k} n={n} {mmm}");
        let input_shape = x_fact.shape.as_concrete().unwrap().to_vec();
        let mut geo = geo.to_concrete(&input_shape)?.into_owned();
        let mut input_shape: DataShape = self.pool_spec.data_format.shape(input_shape.into())?;
        let padding = self.pool_spec.computed_padding(input_shape.hw_dims());
        if padding.iter().any(|axis| axis.pad_before != 0 || axis.pad_after != 0) {
            let mut pads = vec![(0, 0); x_fact.rank()];
            for (ix, ax) in padding.iter().enumerate() {
                pads[input_shape.h_axis() + ix] = (ax.pad_before, ax.pad_after);
            }
            let op = crate::ops::array::Pad {
                mode: crate::ops::array::PadMode::Constant(
                    Tensor::zero_scalar_dt(x_fact.datum_type)?.into_arc_tensor(),
                ),
                pads,
            };
            x = model.wire_node(format!("{name}.pad"), op, &[x])?[0];
            let valid_pool_spec = PoolSpec { padding: Valid, ..self.pool_spec.clone() };
            x_fact = model.outlet_fact(x)?.clone();
            let concrete_shape = x_fact.shape.as_concrete().unwrap();
            input_shape = valid_pool_spec.data_format.shape(concrete_shape.into())?;
            geo = valid_pool_spec
                .compute_geo(&x_fact.shape)?
                .to_concrete(concrete_shape)?
                .into_owned();
        }
        let c_dt = crate::ops::matmul::output_type(x_fact.datum_type);
        let c_stride = input_shape.c_stride();
        let size_of_b = x_fact.datum_type.size_of() as isize;
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
        let b_storage =
            Box::new(LazyIm2colSpec { packer: mmm.b_pack(), n_bytes_offsets, k_bytes_offsets });
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&geo.output_shape)?;

        let kernel = self.wire_kernel_as_g_o_ihw(model, name, kernel)?[0];
        let wire = self.wire_mm_weights_bias(
            model,
            name,
            x,
            kernel,
            bias,
            mmm,
            c_dt,
            mmm_output_shape.clone().into(),
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
        kernel_fact: &TypedFact,
        input_fact: &TypedFact,
    ) -> TractResult<(PoolGeometry, usize, usize, TDim, Box<dyn MatMatMul>)> {
        let a_dt = kernel_fact.datum_type;
        let b_dt = input_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_dt);

        let geo = self.pool_spec.compute_geo(&input_fact.shape)?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.input_channels() * self.pool_spec.kernel_shape.iter().product::<usize>()
            / self.group;
        let n: TDim =
            self.pool_spec.output_shape(&input_fact.shape)?.hw_dims().iter().cloned().product();

        let mmm = tract_linalg::ops()
            .mmm(a_dt, b_dt, c_dt, Some(m), Some(k), n.to_usize().ok())
            .with_context(|| format!("No multiplier for {a_dt:?}x{b_dt:?} to {c_dt:?}",))?;

        Ok((geo, m, k, n, mmm))
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_mm_weights_bias(
        &self,
        model: &mut TypedModel,
        name: &str,
        input: OutletId,
        g_o_ihw: OutletId,
        bias: OutletId,
        mmm: Box<dyn MatMatMul>,
        c_datum_type: DatumType,
        mmm_output_shape: ShapeFact,
        k: usize,
        c_m_axis: usize,
        c_n_axis: usize,
        b_storage: Box<dyn InputStoreSpec>,
    ) -> TractResult<TVec<OutletId>> {
        ensure!(model.outlet_fact(bias)?.datum_type == mmm.internal_type());
        let packed_ker = self
            .wire_pack_g_o_ihw(model, name, mmm.a_pack(), g_o_ihw)
            .context("in kernel_as_packed_as")?;
        let a_dt = model.outlet_fact(packed_ker)?.datum_type;
        let a_storage = unsafe { mmm.a_packed(a_dt.size_of(), k) };
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
        let mut wires: TVec<OutletId> = tvec!(input, packed_ker);
        let mut ops: Vec<ProtoFusedSpec> = vec![ProtoFusedSpec::AddMatMul(geo, 1, 0)];
        let bias_fact = model.outlet_fact(bias)?;
        if bias_fact.konst.is_none() || !bias_fact.konst.as_ref().unwrap().is_zero()? {
            let (fused, bias) = self.wire_bias_as_non_linear(model, name, bias, c_m_axis - 1)?;
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

    pub fn wire_as_depth_wise(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: &[OutletId],
    ) -> TractResult<OutletId> {
        let &[x, kernel, mut bias] = wire else { bail!("Wrong number of inputs") };
        let x_fact = model.outlet_fact(x)?.clone();
        let x_shape = x_fact.shape.as_concrete().unwrap();
        let ConcretePoolGeometry { input_shape, patch, output_shape } =
            self.pool_spec.compute_geo(&x_fact.shape)?.to_concrete(x_shape)?.into_owned();
        let kernel = self.wire_kernel_as_g_o_ihw(model, name, kernel)?;
        let c_axis = self.pool_spec.data_format.shape(x_shape)?.c_axis();
        bias = wire_reshape_bias_for_bin(
            model,
            name,
            bias,
            x_fact.rank(),
            c_axis,
            self.output_channels(),
        )?[0];
        let op = DepthWise::new(patch, input_shape, output_shape);
        Ok(model.wire_node(name, op, &[x, kernel[0], bias])?[0])
    }

    fn declutter_stride_slice_to_downsample(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let spatial_rank = self.pool_spec.rank();
        if let Some(axis) = (0..spatial_rank).find(|&ax| {
            self.pool_spec.stride(ax) > 1
                && self.pool_spec.padding.valid_dim(ax, self.pool_spec.stride(ax) == 1)
                && (self.pool_spec.kernel_shape[ax] == 1
                    || self.pool_spec.dilation(ax) % self.pool_spec.stride(ax) == 0)
        }) {
            let input_fact = model.outlet_fact(node.inputs[0])?;
            let downsample_factor = self.pool_spec.stride(axis);
            let mut new_op = self.clone();
            if new_op.pool_spec.dilation(axis) > 1 {
                new_op.pool_spec.dilations.as_mut().unwrap()[axis] /= downsample_factor;
            }
            new_op.pool_spec.strides.as_mut().unwrap()[axis] /= downsample_factor;
            let mut patch = TypedModelPatch::default();
            let mut taps = patch.taps(model, &node.inputs)?;
            let shape = self.pool_spec.data_format.shape(&input_fact.shape)?;
            taps[0] = patch.wire_node(
                format!("{}.downsample.{}", node.name, axis),
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor as isize, 0),
                &[taps[0]],
            )?[0];
            let id = patch.wire_node(&*node.name, new_op, &taps)?[0];
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
            && self.pool_spec.kernel_shape.iter().product::<usize>() == 1
            && self
                .pool_spec
                .computed_padding(input_shape.hw_dims())
                .iter()
                .all(|pad| pad.pad_after.is_zero() && pad.pad_before.is_zero())
        {
            let mut axes = self.axes_mapping(&input_facts, &output_facts)?;
            let mut patch = TypedModelPatch::new("declutter_as_einsum");
            let mut taps = patch.taps(model, &node.inputs)?;
            let name = &node.name;
            let co = self.output_channels();
            taps[1] =
                self.wire_kernel_as_g_o_ihw(&mut patch, &format!("{name}.filters"), taps[1])?[0];
            taps[1] =
                patch.wire_node(&format!("{name}.filters_as_co_ci"), AxisOp::Rm(0), &[taps[1]])?[0];

            while axes.rank(InOut::In(1)) > 0 {
                axes = axes.remove_axis_occurency(InOut::In(1), 0)?;
            }
            axes = axes
                .with_extra_axis_occurency('O', InOut::In(1), 0)?
                .with_extra_axis_occurency('I', InOut::In(1), 1)?;

            let bias_fact = input_facts[2];
            let wire = if self.q_params.is_some() {
                if bias_fact.rank() == 1 {
                    axes = axes.linking('O', (InOut::In(2), 0))?;
                }
                let op = EinSum { axes, operating_dt: i32::datum_type(), q_params: self.q_params };
                patch.wire_node(format!("{name}.einsum"), op, &taps)?[0]
            } else {
                axes = axes.remove_slot(InOut::In(2))?;
                let op = EinSum { axes, operating_dt: input_facts[0].datum_type, q_params: None };
                let mut wire = patch.wire_node(format!("{name}.einsum"), op, &taps[0..2])?[0];

                if !bias_fact.konst.as_ref().map(|f| f.is_zero()).transpose()?.unwrap_or(false) {
                    let bias_current_shape =
                        if bias_fact.rank() == 0 { tvec!() } else { tvec!(co.to_dim()) };
                    let mut bias_shape = tvec!(1.to_dim(); input_shape.rank());
                    if bias_fact.rank() > 0 {
                        bias_shape[input_shape.c_axis()] = co.to_dim();
                    }
                    let b = patch.wire_node(
                        format!("{name}.bias.reshape"),
                        AxisOp::Reshape(0, bias_current_shape, bias_shape),
                        &[taps[2]],
                    )?[0];
                    wire = patch.wire_node(
                        format!("{name}.bias"),
                        crate::ops::math::add(),
                        &[wire, b],
                    )?[0];
                }
                wire
            };
            patch.node_mut(wire.node).name = node.name.to_string();
            patch.shunt_outside(model, node.id.into(), wire)?;
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
        let &[succ_outlet] = &*node.outputs[0].successors else { return Ok(None) };
        let succ = model.node(succ_outlet.node);
        let Some(bin) = succ.op_as::<TypedBinOp>() else { return Ok(None) };
        let other_input = succ.inputs[1 - succ_outlet.slot];
        let axes_mapping = model.node_axes_mapping(succ.id)?;
        let input_shape =
            self.pool_spec.data_format.shape(&model.outlet_fact(node.inputs[0])?.shape)?;
        let conv_c_axis = input_shape.c_axis();
        if axes_mapping.axis((InOut::In(succ_outlet.slot), conv_c_axis))?.inputs
            [1 - succ_outlet.slot]
            .len()
            != 1
        {
            return Ok(None);
        };
        let mut other_expected_shape = tvec!(1.to_dim(); input_shape.rank());
        other_expected_shape[conv_c_axis] = self.output_channels().to_dim();
        if *other_expected_shape != *model.outlet_fact(other_input)?.shape {
            return Ok(None);
        }

        let mut patch = TypedModelPatch::default();
        let [input, mut kernel, mut bias] = &*patch.taps(model, &node.inputs)? else {
            panic!("Expect three inputs");
        };
        let name = &node.name;
        let succ_name = &succ.name;

        let operand = patch.tap_model(model, other_input)?;

        let renamed = format!("{name}.{succ_name}");
        bias = wire_reshape_bias_for_bin(
            &mut patch,
            format!("{renamed}.reshape_bias"),
            bias,
            1,
            0,
            self.output_channels(),
        )?[0];

        let operand = wire_reshape_bias_for_bin(
            &mut patch,
            format!("{renamed}.reshape_operand"),
            operand,
            1,
            0,
            self.output_channels(),
        )?[0];

        let operand_fact = patch.outlet_fact(operand)?.shape.to_tvec();
        let kernel_fact = patch.outlet_fact(kernel)?;
        let mut operand_shape_for_kernel = tvec!(1.to_dim(); 2 + input_shape.hw_rank());
        operand_shape_for_kernel[self.kernel_fmt.o_axis(&kernel_fact.shape)] =
            self.output_channels().to_dim();
        let operand_for_kernel = patch.wire_node(
            format!("{renamed}.reshape_operand_for_kernel"),
            AxisOp::Reshape(0, operand_fact, operand_shape_for_kernel),
            &[operand],
        )?[0];

        if bin.0.is::<Sub>() && succ_outlet.slot == 0 {
            bias = patch.wire_node(&renamed, sub(), &[bias, operand])?[0];
        } else if bin.0.is::<Sub>() {
            bias = patch.wire_node(&renamed, sub(), &[operand, bias])?[0];
        } else if bin.0.is::<Div>() && succ_outlet.slot == 0 {
            bias = patch.wire_node(&renamed, div(), &[bias, operand])?[0];
            kernel = patch.wire_node(&renamed, div(), &[kernel, operand_for_kernel])?[0];
        } else if bin.0.is::<Div>() {
            bias = patch.wire_node(&renamed, div(), &[operand, bias])?[0];
            kernel = patch.wire_node(&renamed, div(), &[operand_for_kernel, kernel])?[0];
        } else if bin.0.is::<Add>() {
            bias = patch.wire_node(&renamed, add(), &[bias, operand])?[0];
        } else if bin.0.is::<Mul>() {
            bias = patch.wire_node(&renamed, mul(), &[bias, operand])?[0];
            kernel = patch.wire_node(&renamed, mul(), &[kernel, operand_for_kernel])?[0];
        } else {
            return Ok(None);
        };
        let wire = patch.wire_node(&node.name, self.clone(), &[*input, kernel, bias])?[0];
        patch.shunt_outside(model, succ_outlet.node.into(), wire)?;
        Ok(Some(patch))
    }
}

impl Op for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.pool_spec.info();
        info.push(format!("Kernel {:?} (groups:{})", self.kernel_fmt, self.group));
        Ok(info)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for Conv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut model = TypedModel::default();
        let wire: TVec<OutletId> = inputs
            .iter()
            .enumerate()
            .map(|(ix, v)| model.add_source(format!("source.{ix}"), v.datum_type().fact(v.shape())))
            .collect::<TractResult<_>>()?;
        let wire = unsafe {
            if self.q_params.is_some() {
                self.wire_as_quant_im2col(&mut model, "im2col-adhoc", &wire)?
            } else {
                self.wire_as_im2col_pair(&mut model, "im2col-adhoc", &wire)?
            }
        };
        model.set_output_outlets(&wire)?;
        model.into_runnable()?.run(inputs)
    }
}

impl TypedOp for Conv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.q_params.is_some() || inputs[0].datum_type.is_float());
        let q_inputs = if self.q_params.is_some() { 6 } else { 0 };
        if inputs.len() != 3 + q_inputs {
            bail!("Wrong number of inputs: expected {} got {}", 3 + q_inputs, inputs.len());
        }
        if self.q_params.is_some() {
            ensure!(inputs[2].datum_type == i32::datum_type());
            ensure!(inputs[3].datum_type == i32::datum_type());
            ensure!(inputs[4].datum_type.is_float());
            ensure!(inputs[5].datum_type == i32::datum_type());
            ensure!(inputs[6].datum_type.is_float());
            ensure!(inputs[7].datum_type == i32::datum_type());
            ensure!(inputs[8].datum_type.is_float());
        }
        ensure!(self.pool_spec.rank() + 2 == inputs[1].rank());
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
        if let ExplicitOnnxPool(bef, after, _) | Explicit(bef, after) = &self.pool_spec.padding {
            anyhow::ensure!(bef.len() == self.pool_spec.rank());
            anyhow::ensure!(after.len() == self.pool_spec.rank());
        }
        ensure!(
            inputs[2].rank() == 0
                || (inputs[2].rank() == 1
                    && inputs[2].shape.volume() == self.output_channels().to_dim()),
            "Bias should be scalar or a vector with one value per output channel. Output channels is {}, bias is {:?}",
            self.output_channels(),
            inputs[2]
        );
        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);
        if let Some(dt) = self.q_params {
            fact.datum_type = dt;
        } else {
            ensure!(
                inputs[0].datum_type == inputs[1].datum_type,
                "Convolution input, weights and bias must have the same type, got {inputs:?}",
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
        let shape = self.pool_spec.data_format.shape(&fact.shape)?;
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
        let kernel_spatial_shape = &self.pool_spec.kernel_shape;
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
            for (qp_ix, qp) in inputs.iter().enumerate().skip(3) {
                if qp.rank() == 1 {
                    axes = match qp_ix {
                        3 | 4 => axes.linking('I', (InOut::In(qp_ix), 0))?,
                        5 | 6 => axes.linking('O', (InOut::In(qp_ix), 0))?,
                        7 | 8 => axes.linking('O', (InOut::In(qp_ix), 0))?,
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
        let kernel_spatial_shape = &self.pool_spec.kernel_shape;
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
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            shape.n().cloned().unwrap_or(one)
                * shape.c()
                * n_output_channels
                * n_output_points
                * kernel_surface
                / self.group
        )))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if io == InOut::In(1) {
            return Ok(None);
        }
        if io == InOut::In(2) {
            if let &AxisOp::Rm(_) = change {
                return Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(self.clone())),
                    wire_changes: tvec!(),
                }));
            }
        }
        let full_input_shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let shape = self.pool_spec.data_format.shape(full_input_shape.clone())?;
        // remove n
        if let Some(n) = shape.n_axis() {
            assert_eq!(n, 0);
            if change == &AxisOp::Rm(n) {
                let op = Conv { pool_spec: self.pool_spec.dispose_n_axis(), ..self.clone() };
                return Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(op)),
                    wire_changes: tvec!(
                        (InOut::In(0), change.clone()),
                        (InOut::Out(0), change.clone())
                    ),
                }));
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
                let geo_axis = a - h_axis;
                (Rm(geo_axis), Rm(kh_axis + geo_axis))
            }
            Add(a) if hw_axes.contains(a) => (Add(a - h_axis), Add(a - h_axis + kh_axis)),
            Move(f, t) if hw_axes.contains(f) && hw_axes.contains(t) => {
                (Move(f - h_axis, t - h_axis), Move(f - h_axis + kh_axis, t - h_axis + kh_axis))
            }
            _ => return Ok(None),
        };
        let pool_spec = self.pool_spec.change_geo_axes(&geo_adjusted)?;
        let new_op = Conv { pool_spec, ..self.clone() };
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(new_op)),
            wire_changes: tvec!(
                (InOut::In(0), change.clone()),
                (InOut::In(1), kernel_adjusted),
                (InOut::Out(0), change.clone())
            ),
        }))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        unsafe {
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
                let inputs = patch.taps(model, &node.inputs)?;
                let wire = self
                    .wire_as_lazy_im2col(&mut patch, &node.name, &inputs)
                    .context("wire_as_lazy_im2col")?[0];
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                patch.obliterate(node.id)?;
                Ok(Some(patch))
            } else if self.group != 1
                && self.group == self.output_channels()
                && self.group == self.input_channels()
                && input_fact.shape.as_concrete().is_some()
            {
                let mut patch = TypedModelPatch::default();
                let inputs = patch.taps(model, &node.inputs)?;
                let wire = self
                    .wire_as_depth_wise(&mut patch, &node.name, &inputs)
                    .context("wire_as_depth_wise")?;
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                patch.obliterate(node.id)?;
                Ok(Some(patch))
            } else {
                let mut patch = TypedModelPatch::default();
                let inputs = patch.taps(model, &node.inputs)?;
                let wire = self
                    .wire_as_im2col_pair(&mut patch, &node.name, &inputs)
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
        let op = Conv {
            pool_spec: PoolSpec {
                data_format: NCHW,
                kernel_shape: tvec!(2, 2),
                padding: Valid,
                dilations: None,
                strides: None,
                input_channels: 1,
                output_channels: 1,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: Some(i32::datum_type()),
        };
        let input = tvec!(
            rctensor4(&[[[[1u8, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            rctensor4(&[[[[1u8, 1], [1, 1]]]]),
            rctensor0(0u32),
            rctensor0(1u8),
            rctensor0(1.0f32),
            rctensor0(0u8),
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
        let kernel = model.add_const("kernel", rctensor3(&[[[1f32, 2f32]]]))?;
        let bias = model.add_const("bias", rctensor0(0f32))?;
        let wire = model.wire_node(
            "conv",
            Conv {
                pool_spec: PoolSpec {
                    data_format: crate::ops::nn::DataFormat::CHW,
                    dilations: None,
                    strides: None,
                    kernel_shape: tvec![2],
                    padding: Explicit(tvec![0], tvec![0]),
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: crate::ops::cnn::KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[wire[0], kernel, bias],
        )?;
        model.set_output_outlets(&wire)?;
        model.declutter()?;
        assert_eq!(model.nodes().len(), 4); // source + conv + kernel + bias
        let cv = model.nodes()[3].op_as::<Conv>().unwrap();
        assert_eq!(cv.pool_spec.padding, Explicit(tvec![1], tvec![0])); // source + conv
        Ok(())
    }
}
