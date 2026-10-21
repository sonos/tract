use crate::ops::change_axes::GpuAxisOp;
use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::cnn::{Deconv, KernelFormat};

/// Launches the backend's deconv kernel over `output`, one thread per output element.
pub type DispatchDeconvFn = fn(
    geometry: &DeconvGeometry,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()>;

/// The scalars a 2D gather deconv kernel takes, in its argument order. A 1D deconv is
/// laid out as 2D with a unit height (one row, one kernel row, unit stride, no padding).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeconvGeometry {
    pub n: i32,
    pub co: i32,
    pub ih: i32,
    pub iw: i32,
    pub oh: i32,
    pub ow: i32,
    pub kh: i32,
    pub kw: i32,
    pub ci_per_group: i32,
    pub co_per_group: i32,
    pub pad_h: i32,
    pub pad_w: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    pub dil_h: i32,
    pub dil_w: i32,
    pub input_strides: [i32; 4],
    pub weight_strides: [i32; 4],
    pub output_strides: [i32; 4],
}

impl DeconvGeometry {
    pub fn new(
        op: &Deconv,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<DeconvGeometry> {
        ensure!(op.kernel_format == KernelFormat::OIHW, "GPU deconv reads an OIHW kernel");
        let in_shape = op.pool_spec.data_format.shape(input.shape())?;
        let out_shape = op.pool_spec.data_format.shape(output.shape())?;
        let rank = in_shape.hw_rank();
        ensure!(rank == 1 || rank == 2, "GPU deconv is 1D or 2D");
        let padding = op.pool_spec.padding.compute_for_deconv(
            in_shape.hw_dims(),
            &op.pool_spec.kernel_shape,
            &op.pool_spec.dilations(),
            &op.pool_spec.strides(),
            &op.adjustments,
        )?;
        let strides = op.pool_spec.strides();
        let dilations = op.pool_spec.dilations();
        // Spatial values as (h, w); a 1D deconv gets the unit h.
        let hw = |unit: isize, v: &[isize]| if rank == 2 { (v[0], v[1]) } else { (unit, v[0]) };
        let to_isize = |v: &[usize]| v.iter().map(|x| *x as isize).collect::<TVec<_>>();
        let (ih, iw) = hw(1, &to_isize(in_shape.hw_dims()));
        let (oh, ow) = hw(1, &to_isize(out_shape.hw_dims()));
        let (kh, kw) = hw(1, &to_isize(&op.pool_spec.kernel_shape));
        let pads = padding.iter().map(|p| p.pad_before as isize).collect::<TVec<_>>();
        let (pad_h, pad_w) = hw(0, &pads);
        let (stride_h, stride_w) = hw(1, &to_isize(&strides));
        let (dil_h, dil_w) = hw(1, &to_isize(&dilations));
        let in_hw = hw(0, &to_isize(in_shape.hw_strides()));
        let out_hw = hw(0, &to_isize(out_shape.hw_strides()));
        let ws = weights.strides();
        ensure!(ws.len() == rank + 2, "GPU deconv kernel rank");
        let (w_sh, w_sw) = hw(0, &ws[2..]);
        let i = |v: isize| v as i32;
        Ok(DeconvGeometry {
            n: *in_shape.n().unwrap_or(&1) as i32,
            co: *out_shape.c() as i32,
            ih: i(ih),
            iw: i(iw),
            oh: i(oh),
            ow: i(ow),
            kh: i(kh),
            kw: i(kw),
            ci_per_group: (op.pool_spec.input_channels / op.group) as i32,
            co_per_group: (op.pool_spec.output_channels / op.group) as i32,
            pad_h: i(pad_h),
            pad_w: i(pad_w),
            stride_h: i(stride_h),
            stride_w: i(stride_w),
            dil_h: i(dil_h),
            dil_w: i(dil_w),
            input_strides: [
                *in_shape.n_stride().unwrap_or(&0) as i32,
                *in_shape.c_stride() as i32,
                i(in_hw.0),
                i(in_hw.1),
            ],
            weight_strides: [i(ws[0]), i(ws[1]), i(w_sh), i(w_sw)],
            output_strides: [
                *out_shape.n_stride().unwrap_or(&0) as i32,
                *out_shape.c_stride() as i32,
                i(out_hw.0),
                i(out_hw.1),
            ],
        })
    }

    /// The scalar arguments in kernel order, after the input, weights and output buffers.
    pub fn args(&self) -> [i32; 28] {
        let [in_sn, in_sc, in_sh, in_sw] = self.input_strides;
        let [w_so, w_si, w_sh, w_sw] = self.weight_strides;
        let [out_sn, out_sc, out_sh, out_sw] = self.output_strides;
        [
            self.n,
            self.co,
            self.ih,
            self.iw,
            self.oh,
            self.ow,
            self.kh,
            self.kw,
            self.ci_per_group,
            self.co_per_group,
            self.pad_h,
            self.pad_w,
            self.stride_h,
            self.stride_w,
            self.dil_h,
            self.dil_w,
            in_sn,
            in_sc,
            in_sh,
            in_sw,
            w_so,
            w_si,
            w_sh,
            w_sw,
            out_sn,
            out_sc,
            out_sh,
            out_sw,
        ]
    }

    /// Number of output elements, one thread each.
    pub fn output_len(&self) -> usize {
        self.n as usize * self.co as usize * self.oh as usize * self.ow as usize
    }
}

/// A transposed convolution on the device, without bias (see [`wire_gpu_deconv`]).
/// Takes 1D and 2D deconvs on f16 or f32 with an OIHW kernel, which
/// `rewrite_kernel_deconv_in_oihw` provides.
#[derive(Clone)]
pub struct GpuDeconv {
    pub op: Deconv,
    pub backend_name: &'static str,
    pub dispatch: DispatchDeconvFn,
}

impl GpuDeconv {
    pub fn is_supported(op: &Deconv, input_facts: &[&TypedFact]) -> bool {
        op.kernel_format == KernelFormat::OIHW
            && matches!(op.pool_spec.kernel_shape.len(), 1 | 2)
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
            && input_facts[1].datum_type == input_facts[0].datum_type
            && input_facts.iter().all(|f| f.is_plain())
    }
}

impl std::fmt::Debug for GpuDeconv {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Deconv({:?})", self.backend_name, self.op)
    }
}

impl PartialEq for GpuDeconv {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.op == other.op
    }
}
impl Eq for GpuDeconv {}

impl std::hash::Hash for GpuDeconv {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.op.hash(state);
    }
}

impl Op for GpuDeconv {
    fn name(&self) -> StaticName {
        format!("{}Deconv", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        self.op.info()
    }
    op_as_typed_op!();
}

impl EvalOp for GpuDeconv {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let weights = inputs[1].to_device_tensor()?;
        let output_shape = tract_core::ops::cnn::deconv::output_shape(
            &self.op.pool_spec,
            input.shape(),
            &self.op.adjustments,
        )?;
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), &output_shape)?;
        if output.len() > 0 {
            let geometry = DeconvGeometry::new(&self.op, input, weights, &output)?;
            (self.dispatch)(&geometry, input, weights, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuDeconv {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            self.op.output_facts(&[facts[0], facts[1], &zero])
        })
        .with_context(|| format!("Error while computing facts for {}Deconv", self.backend_name))
    }
}

/// Wires `deconv` in place of the core `Deconv` node, and adds the node's bias (when not
/// a constant zero) after it with `bias_add`, the backend's broadcasting Add.
pub fn wire_gpu_deconv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &[OutletId],
    deconv: GpuDeconv,
    bias_add: Box<dyn TypedOp>,
) -> TractResult<TVec<OutletId>> {
    let facts = source.node_input_facts(node.id)?;
    let need_bias = facts.len() > 2
        && !facts[2].konst.as_ref().is_some_and(|k| k.is_all_zero().unwrap_or(false));
    let name = &node.name;
    let data_shape = deconv.op.pool_spec.data_format.shape(&facts[0].shape)?;
    let wire = target.wire_node(
        if need_bias { format!("{name}.deconv") } else { name.clone() },
        deconv,
        &inputs[0..2],
    )?[0];
    if !need_bias {
        return Ok(tvec!(wire));
    }
    let mut bias_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
    bias_shape[data_shape.c_axis()] = facts[2].shape.volume();
    let bias = target.wire_node(
        format!("{name}.bias_reshaped"),
        GpuAxisOp::new(AxisOp::Reshape(0, facts[2].shape.to_tvec(), bias_shape)),
        &[inputs[2]],
    )?[0];
    target.wire_node(name, bias_add, &[wire, bias])
}
