use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::OnceLock;

use crate::context::metal_context;
use crate::kernels::matmul::{GemmKernel, GgmlGemm, MetalGemmImplKind, MfaGemm, MlxGemm};
use crate::{kernels, ops};
use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::cnn::conv::rewrite_kernel_conv_in_oihw;
use tract_core::ops::cnn::{Conv, rewrite_conv_with_n_axis};
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::rewrite_rules::rms_norm::remove_rms_norm_cast;
use tract_gpu::sync::{
    DeviceSync, DeviceSyncKind, sync_inputs_if_required, sync_model_outputs_if_required,
};
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::moe_ffn::{ExpertLayout, MoeFfn, RouteTopK, RoutedInputMode};

use crate::rewrite_rules;

/// A registered translator that can convert a core op into a Metal GPU op.
/// Each kernel module submits one (or more) of these via [`register_metal_op!`].
#[allow(clippy::type_complexity)]
pub struct MetalOpTranslator {
    pub type_id: TypeId,
    pub try_make: fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>,
}

inventory::collect!(MetalOpTranslator);

/// Register a translator for a core op type. The closure receives `(source, node, op)`
/// where `op` is already downcast to `$op_type`. Return `Ok(Some(gpu_op))` to translate,
/// `Ok(None)` to skip.
#[macro_export]
macro_rules! register_metal_op {
    ($op_type:ty, |$source:ident, $node:ident, $op:ident| $body:expr) => {
        inventory::submit! {
            $crate::transform::MetalOpTranslator {
                type_id: std::any::TypeId::of::<$op_type>(),
                try_make: |$source, $node| {
                    let Some($op) = $node.op_as::<$op_type>() else {
                        return Ok(None);
                    };
                    $body
                },
            }
        }
    };
}

/// Metal-local SDPA flattening: explode only the `Sdpa` nodes the MFA kernel
/// can't fuse, leaving fusable ones for the `MetalMfaSdpa` translator. (The
/// shared `tract_gpu` `rewire_sdpa` explodes all of them; cuda still uses it.)
fn flatten_unfused_sdpa(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &tract_transformers::ops::sdpa::Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    if crate::kernels::matmul::mfa::mfa_sdpa_supported(op, &in_facts) {
        Ok(None) // leave intact for the MetalMfaSdpa translator
    } else {
        op.patch_sdpa(model, node) // explode (same as the shared rewire_sdpa)
    }
}

fn rewire_sdpa_metal(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("flatten-unfused-sdpa", flatten_unfused_sdpa)
        .rewrite(&(), model)
}

impl MetalGemmImplKind {
    pub fn variants() -> Vec<MetalGemmImplKind> {
        vec![Self::Mlx, Self::Mfa, Self::Ggml]
    }

    pub fn variants_str() -> Vec<&'static str> {
        Self::variants().into_iter().map(|it| it.to_str()).collect()
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Mfa => "mfa",
            Self::Ggml => "ggml",
        }
    }
}

#[derive(Debug, Default)]
pub struct MetalTransform {
    pub gemm_impl: Option<MetalGemmImplKind>,
}

impl ModelTransform for MetalTransform {
    fn name(&self) -> StaticName {
        "metal-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        self.transform_up_to_phase(model, usize::MAX)
    }
}

impl FromStr for MetalTransform {
    type Err = TractError;
    fn from_str(str: &str) -> TractResult<Self> {
        let gemm_impl = match str {
            "mlx" => Some(MetalGemmImplKind::Mlx),
            "ggml" => Some(MetalGemmImplKind::Ggml),
            "mfa" => Some(MetalGemmImplKind::Mfa),
            "" => None,
            _ => bail!("Unknown backend"),
        };
        Ok(MetalTransform { gemm_impl })
    }
}

impl MetalTransform {
    pub fn transform_up_to_phase(
        &self,
        model: &mut TypedModel,
        stop_at_phase: usize,
    ) -> TractResult<()> {
        // Init Metal Context if not done previously
        metal_context();

        rewire_sdpa_metal(model)?;
        rewrite_einsum_to_prefix_matmul(model, false)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::<MetalTransform>::default()
            .with_rule_for("untranspose-matmul-output", rewrite_rules::untranspose_matmul_output)
            .with_rule_for("add-broadcast-pre-matmul", rewrite_rules::add_broadcast_pre_matmul)
            .rewrite(self, model)?;

        Rewriter::default()
            .with_rule_for("rewrite_kernel_conv_in_oihw", rewrite_kernel_conv_in_oihw)
            .with_rule_for("rewrite_conv_with_n_axis", rewrite_conv_with_n_axis)
            .with_rule_for("remove_rms_norm_cast", remove_rms_norm_cast)
            .rewrite(&(), model)?;

        if stop_at_phase == 1 {
            return Ok(());
        }

        *model = self.translate_model(model)?;

        if stop_at_phase == 2 {
            return Ok(());
        }

        Rewriter::default()
            .with_rule_for("fuse_move_axis", rewrite_rules::fuse_move_axis)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_axis_op", rewrite_rules::fuse_axis_op)
            .rewrite(&(), model)?;

        rewire_syncs(model)?;
        Ok(())
    }
}

/// Looks up the node's op TypeId in the inventory of registered `MetalOpTranslator`s.
/// Returns `Some(gpu_op)` if a translator matches and succeeds, `None` otherwise.
fn try_make_metal_op(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    type TranslateFn = fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>;
    static MAP: OnceLock<HashMap<TypeId, Vec<TranslateFn>>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m: HashMap<TypeId, Vec<TranslateFn>> = HashMap::new();
        for t in inventory::iter::<MetalOpTranslator> {
            m.entry(t.type_id).or_default().push(t.try_make);
        }
        m
    });

    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type)));

    // Copy-based ops are fully generic (no backend-specific dispatch needed).
    if let Some(op) = tract_gpu::ops::copy_based::try_make_copy_based_op(source, node)? {
        return Ok(Some(op));
    }

    if let Some(fns) = map.get(&(*node.op).type_id()) {
        for f in fns {
            if let Some(op) = f(source, node)? {
                return Ok(Some(op));
            }
        }
    }
    Ok(None)
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for MetalTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        // Special multi-node ops handled first
        let input_facts = source.node_input_facts(node.id)?;
        if let Some(op) = node.op_as::<PrefixMatMul>() {
            let facts: Vec<TypedFact> = input_facts.iter().map(|f| (*f).clone()).collect();
            if !op.transpose_c && op.quantize_output.is_none() && check_matmul_in_dts(&facts) {
                let mut device_inputs =
                    sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
                let outlet_ids = convert_matmul_to_metal(
                    source,
                    node,
                    target,
                    &mut device_inputs,
                    op,
                    self.gemm_impl,
                )?;
                return sync_model_outputs_if_required(source, node, target, outlet_ids);
            }
        }
        if let Some(conv) = node.op_as::<Conv>()
            && input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type))
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                ops::conv::wire_metal_conv(source, node, target, &device_inputs, conv)?;
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }
        if let Some(op) = node.op_as::<MoeFfn>()
            && let Some(outlet_ids) =
                convert_q40_moe_ffn_to_metal(source, node, target, mapping, op)?
        {
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }
        // Const: inline conversion, not a GPU op
        if let Some(op) = node.op_as::<Const>()
            && DeviceTensor::is_supported_dt(op.val().datum_type())
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                target.wire_node(node.name.clone(), convert_const(op)?, &device_inputs)?;
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }

        // Single-op translation.  See the matching CUDA path for rationale:
        // pre-check the gpu_op's output_facts against the already-translated
        // target-side input shapes before wiring, so a stale Reshape (e.g.
        // after pulsification has changed an upstream axis size) falls back
        // to CPU rather than aborting the whole Metal transform.
        let target_inputs: TVec<TypedFact> = node
            .inputs
            .iter()
            .map(|i| target.outlet_fact(mapping[i]).cloned())
            .collect::<TractResult<_>>()?;
        // Mirror sync_inputs_if_required(ToDevice): wrap non-device facts as
        // device facts so the GPU op's `output_facts` sees uniform device
        // inputs, matching what it'll receive after sync nodes are wired.
        // Mixed inputs (e.g. host kv-cache + device current activation) make
        // `output_facts` bail with "Inconsistent facts", wrongly tripping CPU
        // fallback.
        let target_inputs_post_sync: TVec<TypedFact> = target_inputs
            .iter()
            .map(|f| -> TractResult<TypedFact> {
                if f.as_device_fact().is_some() {
                    Ok(f.clone())
                } else {
                    Ok(tract_gpu::fact::DeviceFact::from_host(f.clone())?.into_exotic_fact())
                }
            })
            .collect::<TractResult<_>>()?;
        let target_input_post_sync_refs: TVec<&TypedFact> =
            target_inputs_post_sync.iter().collect();
        if let Some(gpu_op) = try_make_metal_op(source, node)?
            && gpu_op.output_facts(&target_input_post_sync_refs).is_ok()
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids = target.wire_node(node.name.clone(), gpu_op, &device_inputs)?;
            sync_model_outputs_if_required(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToHost)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}

fn sync_outlet_if_required(
    target: &mut TypedModel,
    name: impl Into<String>,
    outlet: OutletId,
    sync_kind: DeviceSyncKind,
) -> TractResult<OutletId> {
    let name = name.into();
    match sync_kind {
        DeviceSyncKind::ToHost if target.outlet_fact(outlet)?.as_device_fact().is_some() => {
            Ok(target.wire_node(name, DeviceSync::new(sync_kind), &[outlet])?[0])
        }
        DeviceSyncKind::ToDevice if target.outlet_fact(outlet)?.as_device_fact().is_none() => {
            let host_fact = target.outlet_fact(outlet)?.clone();
            if let Some(konst) = host_fact.konst.as_ref()
                && konst.as_device_tensor().is_none()
            {
                let device_konst = konst.as_ref().clone().into_device()?.into_tensor();
                let device_fact = DeviceFact::from_host(host_fact)?;
                let fact = target.outlet_fact_mut(outlet)?;
                *fact = device_fact.into_exotic_fact();
                fact.konst = Some(device_konst.into_arc_tensor());
                return Ok(outlet);
            }
            ensure!(
                host_fact.datum_type.is_copy(),
                "Only copy DatumType can be synced to Device: {:?}",
                host_fact.datum_type
            );
            Ok(target.wire_node(name, DeviceSync::new(sync_kind), &[outlet])?[0])
        }
        _ => Ok(outlet),
    }
}

fn q40_moe_activation_supported(op: &MoeFfn) -> bool {
    matches!(op.activation.as_str(), "silu")
        || (op.has_w3 && matches!(op.activation.as_str(), "swiglu"))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
}

fn convert_q40_moe_ffn_to_metal(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &HashMap<OutletId, OutletId>,
    op: &MoeFfn,
) -> TractResult<Option<TVec<OutletId>>> {
    let log_lowering = env_flag("TRACT_METAL_LOG_Q40_MOE");
    if env_flag("TRACT_METAL_DISABLE_Q40_MOE") {
        return Ok(None);
    }
    if op.has_wg_bias
        || op.has_w1_bias
        || op.has_w3_bias
        || op.has_w2_bias
        || op.act_limit_bits.is_some()
        || op.expert_layout != ExpertLayout::Linear
        || !q40_moe_activation_supported(op)
    {
        if log_lowering {
            eprintln!(
                "Metal Q40 MoE skip {}: metadata layout={:?} activation={} has_w3={} biases=[wg:{} w1:{} w3:{} w2:{}] act_limit={}",
                node.name,
                op.expert_layout,
                op.activation,
                op.has_w3,
                op.has_wg_bias,
                op.has_w1_bias,
                op.has_w3_bias,
                op.has_w2_bias,
                op.act_limit_bits.is_some()
            );
        }
        return Ok(None);
    }

    let facts = source.node_input_facts(node.id)?;
    let x_rank_ok =
        facts[0].rank() == 2 || (facts[0].rank() == 3 && facts[0].shape.dims()[0] == 1.to_dim());
    let x_dt_ok = matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16);
    let wg_dt_ok = matches!(facts[1].datum_type, DatumType::F32 | DatumType::F16);
    let w1_q40 = as_quant_fact(facts[2], &Q4_0).is_some();
    let w2_q40 = as_quant_fact(facts[3], &Q4_0).is_some();
    let w3_q40 =
        !op.has_w3 || facts.get(4).is_some_and(|fact| as_quant_fact(fact, &Q4_0).is_some());
    let facts_supported = x_rank_ok && x_dt_ok && wg_dt_ok && w1_q40 && w2_q40 && w3_q40;
    if log_lowering {
        eprintln!(
            "Metal Q40 MoE candidate {}: x_rank={} x_dt={:?} wg_dt={:?} w1_q40={} w2_q40={} w3_q40={} shapes x={:?} wg={:?} w1={:?} w2={:?}{}",
            node.name,
            facts[0].rank(),
            facts[0].datum_type,
            facts[1].datum_type,
            w1_q40,
            w2_q40,
            w3_q40,
            facts[0].shape,
            facts[1].shape,
            facts[2].shape,
            facts[3].shape,
            if op.has_w3 { format!(" w3={:?}", facts[4].shape) } else { String::new() }
        );
    }
    if !facts_supported {
        if log_lowering {
            eprintln!("Metal Q40 MoE skip {}: unsupported facts", node.name);
        }
        return Ok(None);
    }

    if log_lowering {
        eprintln!("Metal Q40 MoE lowering: {}", node.name);
    }

    let x_device = sync_outlet_if_required(
        target,
        format!("{}.x.to-device", node.name),
        mapping[&node.inputs[0]],
        DeviceSyncKind::ToDevice,
    )?;
    let x_device = if facts[0].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.x.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).unwrap(),
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let wg_device = sync_outlet_if_required(
        target,
        format!("{}.wg.to-device", node.name),
        mapping[&node.inputs[1]],
        DeviceSyncKind::ToDevice,
    )?;
    let wg_device = if facts[1].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.wg.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).unwrap(),
            &[wg_device],
        )?[0]
    } else {
        wg_device
    };

    let routes = if env_flag("TRACT_METAL_DISABLE_ROUTE_TOPK") {
        let x_host = sync_outlet_if_required(
            target,
            format!("{}.route-x-to-cpu", node.name),
            mapping[&node.inputs[0]],
            DeviceSyncKind::ToHost,
        )?;
        let wg_host = sync_outlet_if_required(
            target,
            format!("{}.route-wg-to-cpu", node.name),
            mapping[&node.inputs[1]],
            DeviceSyncKind::ToHost,
        )?;
        target.wire_node(
            format!("{}.route_topk", node.name),
            RouteTopK { k: op.k, gate: op.gate.clone() },
            &[x_host, wg_host],
        )?
    } else {
        target.wire_node(
            format!("{}.route_topk", node.name),
            ops::MetalRouteTopK { k: op.k, gate: op.gate.clone() },
            &[x_device, wg_device],
        )?
    };
    let route_token_ids = sync_outlet_if_required(
        target,
        format!("{}.route_token_ids.to-device", node.name),
        routes[0],
        DeviceSyncKind::ToDevice,
    )?;
    let route_expert_ids = sync_outlet_if_required(
        target,
        format!("{}.route_expert_ids.to-device", node.name),
        routes[1],
        DeviceSyncKind::ToDevice,
    )?;
    let route_weights = sync_outlet_if_required(
        target,
        format!("{}.route_weights.to-device", node.name),
        routes[2],
        DeviceSyncKind::ToDevice,
    )?;

    let x_shape_like = x_device;
    let x_expert_input = if facts[0].rank() == 3 {
        target.wire_node(
            format!("{}.x.rm-batch", node.name),
            tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Rm(0)),
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let w1_device = sync_outlet_if_required(
        target,
        format!("{}.w1.to-device", node.name),
        mapping[&node.inputs[2]],
        DeviceSyncKind::ToDevice,
    )?;
    let w2_device = sync_outlet_if_required(
        target,
        format!("{}.w2.to-device", node.name),
        mapping[&node.inputs[3]],
        DeviceSyncKind::ToDevice,
    )?;

    let h1 = target.wire_node(
        format!("{}.w1", node.name),
        ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::TokenRows },
        &[x_expert_input, w1_device, route_token_ids, route_expert_ids],
    )?[0];
    let activated = target.wire_node(
        format!("{}.activation", node.name),
        kernels::element_wise::metal_element_wise_op(Box::new(tract_core::ops::nn::Silu {})),
        &[h1],
    )?[0];

    let hidden = if op.has_w3 {
        let w3_device = sync_outlet_if_required(
            target,
            format!("{}.w3.to-device", node.name),
            mapping[&node.inputs[4]],
            DeviceSyncKind::ToDevice,
        )?;
        let gate = target.wire_node(
            format!("{}.w3", node.name),
            ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::TokenRows },
            &[x_expert_input, w3_device, route_token_ids, route_expert_ids],
        )?[0];
        target.wire_node(
            format!("{}.swiglu_mul", node.name),
            kernels::bin_ops::metal_bin_op(Box::new(tract_core::ops::math::Mul)),
            &[activated, gate],
        )?[0]
    } else {
        activated
    };

    let route_values = target.wire_node(
        format!("{}.w2", node.name),
        ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::RouteRows },
        &[hidden, w2_device, route_token_ids, route_expert_ids],
    )?[0];
    let output = target.wire_node(
        format!("{}.combine", node.name),
        ops::MetalRoutedCombine,
        &[x_shape_like, route_values, route_token_ids, route_weights],
    )?[0];
    let output = if facts[0].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.out.cast", node.name),
            metal_cast_new(facts[0].datum_type).unwrap(),
            &[output],
        )?[0]
    } else {
        output
    };

    Ok(Some(tvec![output]))
}

pub(crate) fn metal_cast_new(to: DatumType) -> Option<tract_gpu::ops::cast::GpuCast> {
    tract_gpu::ops::cast::GpuCast::new(
        to,
        "Metal",
        kernels::array::metal_cast_dispatch,
        kernels::array::Cast::is_supported_dt,
    )
}

fn check_matmul_in_dts(in_facts: &[TypedFact]) -> bool {
    MlxGemm.is_supported_dts(in_facts)
        || MfaGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(&[in_facts[1].clone(), in_facts[0].clone()])
}

fn is_input_broadcast(facts: TVec<&TypedFact>) -> bool {
    // Assume weights are in second postion
    let b_batch_dims: Vec<TDim> = if as_quant_fact(facts[1], &Q4_0).is_some() {
        facts[1].shape.dims().to_vec()
    } else {
        let rank = facts[1].rank();
        facts[1].shape.dims()[..rank - 2].to_vec()
    };

    let a_rank = facts[0].rank();
    let mut a_batch_dims = facts[0].shape[..(a_rank - 2)].to_vec();

    a_batch_dims.retain(|tdim| !matches!(tdim, TDim::Sym(_)) || b_batch_dims.contains(tdim));
    let symb_in_a = a_batch_dims != facts[0].shape[..(a_rank - 2)].to_vec();

    let a_batch_size = a_batch_dims.iter().product::<TDim>().gcd();
    let b_batch_size = b_batch_dims.iter().product::<TDim>().gcd();

    (a_batch_size % b_batch_size == 0) && ((a_batch_size != b_batch_size) || symb_in_a)
}

pub fn resolve_gemm_impl(
    gemm_impl: Option<MetalGemmImplKind>,
    input_facts: TVec<&TypedFact>,
) -> TractResult<MetalGemmImplKind> {
    if let Some(gemm) = gemm_impl {
        Ok(gemm)
    } else if as_quant_fact(input_facts[0], &Q4_0).is_some()
        || as_quant_fact(input_facts[1], &Q4_0).is_some()
        || input_facts[0].datum_type != input_facts[1].datum_type
        || is_input_broadcast(input_facts)
    {
        Ok(MetalGemmImplKind::Ggml)
    } else {
        Ok(MetalGemmImplKind::Mlx)
    }
}

fn convert_matmul_to_metal(
    model: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &mut [OutletId],
    op: &PrefixMatMul,
    gemm_impl: Option<MetalGemmImplKind>,
) -> TractResult<TVec<OutletId>> {
    let mut input_facts = model.node_input_facts(node.id)?;

    let expected_dt = model.node_output_facts(node.id)?[0].datum_type;
    let mut resolved_gemm_impl = resolve_gemm_impl(gemm_impl, input_facts.clone())?;
    if expected_dt == DatumType::F32
        && matches!(resolved_gemm_impl, MetalGemmImplKind::Mlx | MetalGemmImplKind::Mfa)
        && input_facts.iter().any(|fact| fact.datum_type == DatumType::F16)
    {
        // Mlx and Mfa only produce homogeneous f16 output here. Use Ggml for
        // PrefixMatMul nodes that require f32 output from f16 inputs.
        resolved_gemm_impl = MetalGemmImplKind::Ggml;
    }
    if matches!(resolved_gemm_impl, MetalGemmImplKind::Mlx | MetalGemmImplKind::Mfa)
        && (input_facts[0].datum_type != input_facts[1].datum_type)
    {
        ensure!(
            input_facts[0].datum_type == DatumType::F16
                || input_facts[1].datum_type == DatumType::F16
        );
        let inp_to_cast = if input_facts[0].datum_type == DatumType::F16 {
            &mut inputs[0]
        } else {
            &mut inputs[1]
        };
        *inp_to_cast = target.wire_node(
            node.name.clone() + ".cast_input",
            metal_cast_new(DatumType::F32).unwrap(),
            &[*inp_to_cast],
        )?[0];
    }

    let mut matmul_output = match resolved_gemm_impl {
        MetalGemmImplKind::Mlx => {
            let op = ops::MetalGemm::<MlxGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Mfa => {
            let op = ops::MetalGemm::<MfaGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Ggml => {
            let mut swap_inputs = false;
            if !GgmlGemm.is_supported_dts(&[input_facts[0].clone(), input_facts[1].clone()])
                && GgmlGemm.is_supported_dts(&[input_facts[1].clone(), input_facts[0].clone()])
            {
                input_facts.swap(0, 1);
                inputs.swap(0, 1);
                swap_inputs = true;
            }

            let a_pos = swap_inputs as usize;
            let b_pos = 1 - swap_inputs as usize;
            if op.transpose_a {
                ensure!(
                    as_quant_fact(input_facts[a_pos], &Q4_0).is_none(),
                    "Cannot transpose Q40 tensor"
                );

                let rank = input_facts[a_pos].rank();
                let perm_a_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                let perm_a_name = node.name.clone() + ".perm_a";
                inputs[a_pos] = target.wire_node(perm_a_name, perm_a_op, &[inputs[a_pos]])?[0];
            }

            // The GGML kernels now consume f16 activations directly (and emit
            // f16 output via output_dt), so no f16->f32 activation upcast is
            // inserted here anymore.

            if !op.transpose_b {
                ensure!(
                    as_quant_fact(input_facts[b_pos], &Q4_0).is_none(),
                    "Cannot transpose Q40 tensor"
                );

                let rank = input_facts[b_pos].rank();
                let perm_b_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                let perm_b_name = node.name.clone() + ".perm_b";
                inputs[b_pos] = target.wire_node(perm_b_name, perm_b_op, &[inputs[b_pos]])?[0];
            }
            let op = ops::MetalGemm::<GgmlGemm>::new(false, true);
            let mut matmul_output = target.wire_node(node.name.clone(), op, inputs)?;

            if swap_inputs {
                let out_fact = target.outlet_fact(matmul_output[0])?;
                let rank = &out_fact
                    .exotic_fact
                    .clone()
                    .map(|fact| fact.clarify_dt_shape().unwrap().1.len())
                    .unwrap();

                let perm_out_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                matmul_output = target.wire_node(
                    node.name.clone() + ".perm_out",
                    perm_out_op,
                    &matmul_output,
                )?;
            }
            matmul_output
        }
    };

    let out_fact = target.outlet_fact(matmul_output[0])?;
    let out_dt = out_fact.as_device_fact().map(|f| f.datum_type).unwrap_or(out_fact.datum_type);

    if out_dt != expected_dt {
        ensure!(
            kernels::array::Cast::is_supported_dt(out_dt),
            "Matmul output type cannot be casted to expected type"
        );
        let cast_op = metal_cast_new(model.node_output_facts(node.id)?[0].datum_type).unwrap();
        matmul_output =
            target.wire_node(node.name.clone() + ".out_cast", cast_op, &matmul_output)?
    }
    Ok(matmul_output)
}

fn convert_const(op: &Const) -> TractResult<Const> {
    let typed_fact: TypedFact = Arc::clone(op.val()).try_into()?;
    let metal_fact = if let Some(of) = op.exotic_fact() {
        DeviceFact::from_host(typed_fact.with_exotic_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };

    let metal_const = op.val().clone().into_device()?.into_tensor().into_arc_tensor();
    Const::new_with_exotic_fact(metal_const, Box::new(metal_fact))
}
