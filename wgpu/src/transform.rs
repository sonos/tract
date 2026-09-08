use std::any::TypeId;
use std::collections::HashMap;
use std::sync::OnceLock;

use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::cnn::conv::rewrite_kernel_conv_in_oihw;
use tract_core::ops::cnn::{Conv, Deconv, rewrite_conv_with_n_axis};
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::konst::Const;
use tract_core::ops::nn::Reduce;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
#[cfg(not(target_arch = "wasm32"))]
use tract_gpu::sync::sync_model_outputs_if_required;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::context::wgpu_context;
use crate::ops;

/// A registered translator that can convert a core op into a wgpu GPU op.
pub struct WgpuOpTranslator {
    pub type_id: TypeId,
    #[allow(clippy::type_complexity)]
    pub try_make: fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>,
}

inventory::collect!(WgpuOpTranslator);

/// Register a translator for a core op type. The closure receives `(source, node, op)`
/// where `op` is already downcast to `$op_type`. Return `Ok(Some(gpu_op))` to translate,
/// `Ok(None)` to skip.
#[macro_export]
macro_rules! register_wgpu_op {
    ($op_type:ty, |$source:ident, $node:ident, $op:ident| $body:expr) => {
        inventory::submit! {
            $crate::transform::WgpuOpTranslator {
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

#[derive(Debug, Default)]
pub struct WgpuTransform;

impl ModelTransform for WgpuTransform {
    fn name(&self) -> StaticName {
        "wgpu-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        crate::ops::pool::link_translators();
        wgpu_context();
        // Pointwise convolutions reach the backend as EinSum, so a segmenter is
        // mostly matmuls until this runs.
        rewrite_einsum_to_prefix_matmul(model, false)?;
        Rewriter::default()
            .with_rule_for("rewrite_kernel_conv_in_oihw", rewrite_kernel_conv_in_oihw)
            .with_rule_for("rewrite_conv_with_n_axis", rewrite_conv_with_n_axis)
            .with_rule_for("split_multi_axis_reduce", split_multi_axis_reduce)
            .rewrite(&(), model)?;
        *model = self.translate_model(model)?;
        Rewriter::default()
            .with_rule_for("start_elementwise_chain", crate::rewrite_rules::start_elementwise_chain)
            .with_rule_for("start_binary_chain", crate::rewrite_rules::start_binary_chain)
            .with_rule_for("grow_elementwise_chain", crate::rewrite_rules::grow_elementwise_chain)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_gemm_epilogue", crate::rewrite_rules::fuse_gemm_epilogue)
            .with_rule_for("fuse_conv_epilogue", crate::rewrite_rules::fuse_conv_epilogue)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_move_axis", crate::rewrite_rules::fuse_move_axis)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_axis_op", crate::rewrite_rules::fuse_axis_op)
            .rewrite(&(), model)?;
        rewire_syncs(model)?;
        Ok(())
    }
}

fn try_make_wgpu_op(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    type TranslateFn = fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>;
    static MAP: OnceLock<HashMap<TypeId, Vec<TranslateFn>>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m: HashMap<TypeId, Vec<TranslateFn>> = HashMap::new();
        for t in inventory::iter::<WgpuOpTranslator> {
            m.entry(t.type_id).or_default().push(t.try_make);
        }
        m
    });

    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type)));

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

fn convert_const(op: &Const) -> TractResult<Const> {
    let typed_fact: TypedFact = Arc::clone(op.val()).try_into()?;
    let wgpu_fact = if let Some(of) = op.exotic_fact() {
        DeviceFact::from_host(typed_fact.with_exotic_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };
    let wgpu_const = op.val().clone().into_device()?.into_tensor().into_arc_tensor();
    Const::new_with_exotic_fact(wgpu_const, Box::new(wgpu_fact))
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for WgpuTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input_facts = source.node_input_facts(node.id)?;
        if let Some(op) = node.op_as::<PrefixMatMul>()
            && op.quantize_output.is_none()
            && op.operating_dt.is_none_or(|dt| dt == input_facts[0].datum_type)
            && input_facts.iter().all(|f| crate::kernels::matmul::is_supported_dt(f.datum_type))
            && input_facts[0].datum_type == input_facts[1].datum_type
            && input_facts[0].rank() <= crate::kernels::matmul::MAX_RANK
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids = target.wire_node(
                node.name.clone(),
                ops::matmul::WgpuGemm { op: *op, epilogue: vec![] },
                &device_inputs,
            )?;
            return maybe_sync_outputs(source, node, target, outlet_ids);
        }
        if let Some(conv) = node.op_as::<Conv>()
            && input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type))
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
            && conv.pool_spec.kernel_shape.len() == 2
            && conv.q_params.is_none()
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids = ops::conv::wire_wgpu_conv(source, node, target, &device_inputs, conv)?;
            return maybe_sync_outputs(source, node, target, outlet_ids);
        }
        if let Some(deconv) = node.op_as::<Deconv>()
            && input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type))
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
            && deconv.pool_spec.kernel_shape.len() == 2
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                ops::deconv::wire_wgpu_deconv(source, node, target, &device_inputs, deconv)?;
            return maybe_sync_outputs(source, node, target, outlet_ids);
        }
        if let Some(gpu_op) = crate::kernels::resize::wgpu_resize(source, node)? {
            let mut input = mapping[&node.inputs[0]];
            if target.outlet_fact(input)?.as_device_fact().is_none() {
                input = target.wire_node(
                    format!("{}.to-device-0", node.name),
                    tract_gpu::sync::DeviceSync::new(DeviceSyncKind::ToDevice),
                    &[input],
                )?[0];
            }
            let outlet_ids = target.wire_node(node.name.clone(), gpu_op, &[input])?;
            return maybe_sync_outputs(source, node, target, outlet_ids);
        }
        if let Some(op) = node.op_as::<Const>()
            && DeviceTensor::is_supported_dt(op.val().datum_type())
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                target.wire_node(node.name.clone(), convert_const(op)?, &device_inputs)?;
            return maybe_sync_outputs(source, node, target, outlet_ids);
        }

        let target_inputs: TVec<TypedFact> = node
            .inputs
            .iter()
            .map(|i| target.outlet_fact(mapping[i]).cloned())
            .collect::<TractResult<_>>()?;
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
        if let Some(gpu_op) = try_make_wgpu_op(source, node)?
            && gpu_op.output_facts(&target_input_post_sync_refs).is_ok()
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids = target.wire_node(node.name.clone(), gpu_op, &device_inputs)?;
            maybe_sync_outputs(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToHost)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}

/// On wasm, `PollType::Wait` cannot complete a readback. Leave outputs on the
/// GPU; the caller awaits [`crate::to_host_async`] once after `run()`.
fn maybe_sync_outputs(
    src: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    outlet_ids: TVec<OutletId>,
) -> TractResult<TVec<OutletId>> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (src, node, target);
        Ok(outlet_ids)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        sync_model_outputs_if_required(src, node, target, outlet_ids)
    }
}

fn split_multi_axis_reduce(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.axes.len() > 1);
    use tract_core::ops::nn::Reducer::*;
    rule_if!(matches!(op.reducer, Sum | Prod | Min | Max | Any | All));
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[0])?;
    let mut axes = op.axes.clone();
    axes.sort();
    for (i, &axis) in axes.iter().rev().enumerate() {
        let single = Reduce { axes: tvec![axis], reducer: op.reducer };
        wire = patch.wire_node(format!("{node_name}.axis_{i}"), single, &[wire])?[0];
    }
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}
