use std::any::TypeId;
use std::collections::HashMap;
use std::sync::OnceLock;

use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::array::Gather;
use tract_core::ops::cnn::conv::{rewrite_kernel_conv_in_oihw, rewrite_kernel_deconv_in_oihw};
use tract_core::ops::cnn::{Conv, Deconv, rewrite_conv_1d_to_2d, rewrite_conv_with_n_axis};
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::konst::Const;
use tract_core::ops::logic::Iff;
use tract_core::ops::nn::resize::Resize;
use tract_core::ops::nn::{GroupNorm, Reduce, rewrite_nearest_upsample_to_broadcast};
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::float_gather_indices::float_gather_nd_indices;
use tract_gpu::rewrite_rules::layer_norm::fuse_layer_norm_affine;
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::rewrite_rules::rms_norm::fuse_rms_norm_residual;
use tract_gpu::rewrite_rules::rms_norm::fuse_rms_norm_scale;
use tract_gpu::rewrite_rules::rms_norm::fuse_rms_norm_split_scale;
use tract_gpu::rewrite_rules::rms_norm::fuse_scaled_rms_norm_in_cast;
use tract_gpu::rewrite_rules::rms_norm::fuse_scaled_rms_norm_out_cast;
use tract_gpu::rewrite_rules::rms_norm::remove_rms_norm_cast;
use tract_gpu::rewrite_rules::scaled_masked_softmax::drop_unit_axes_of_scaled_masked_softmax;
#[cfg(not(target_arch = "wasm32"))]
use tract_gpu::sync::sync_model_outputs_if_required;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::IntoDevice;

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
        crate::kernels::pool::link_translators();
        wgpu_context();
        Rewriter::default()
            .with_rule_for("flatten_unfused_sdpa", flatten_unfused_sdpa)
            .rewrite(&(), model)?;
        // Pointwise convolutions reach the backend as EinSum, so a segmenter is
        // mostly matmuls until this runs.
        rewrite_einsum_to_prefix_matmul(model, false)?;
        Rewriter::default()
            .with_rule_for("rewrite_kernel_conv_in_oihw", rewrite_kernel_conv_in_oihw)
            .with_rule_for("rewrite_kernel_deconv_in_oihw", rewrite_kernel_deconv_in_oihw)
            .with_rule_for("rewrite_conv_1d_to_2d", rewrite_conv_1d_to_2d)
            .with_rule_for("rewrite_conv_with_n_axis", rewrite_conv_with_n_axis)
            .with_rule_for("split_multi_axis_reduce", split_multi_axis_reduce)
            .with_rule_for("nearest_upsample_to_broadcast", rewrite_nearest_upsample_to_broadcast)
            .with_rule_for("fuse_layer_norm_affine", fuse_layer_norm_affine)
            .with_rule_for("fuse_rms_norm_scale", fuse_rms_norm_scale)
            .with_rule_for("fuse_rms_norm_split_scale", fuse_rms_norm_split_scale)
            .with_rule_for("remove_rms_norm_cast", remove_rms_norm_cast)
            .with_rule_for("fuse_scaled_rms_norm_in_cast", fuse_scaled_rms_norm_in_cast)
            .with_rule_for("fuse_scaled_rms_norm_out_cast", fuse_scaled_rms_norm_out_cast)
            .with_rule_for(
                "drop_unit_axes_of_scaled_masked_softmax",
                drop_unit_axes_of_scaled_masked_softmax,
            )
            .with_rule_for("float_gather_nd_indices", float_gather_nd_indices)
            .with_rule_for("conv_to_winograd_f43", crate::rewrite_rules::conv_to_winograd_f43)
            .rewrite(&(), model)?;
        *model = self.translate_model(model)?;
        Rewriter::default()
            .with_rule_for("start_elementwise_chain", crate::rewrite_rules::start_elementwise_chain)
            .with_rule_for("start_binary_chain", crate::rewrite_rules::start_binary_chain)
            .with_rule_for("grow_elementwise_chain", crate::rewrite_rules::grow_elementwise_chain)
            .rewrite(&(), model)?;
        // A residual add only folds once it has stayed a dispatch of its own.
        Rewriter::default()
            .with_rule_for("fuse_rms_norm_residual", fuse_rms_norm_residual)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_gemv_pair", crate::rewrite_rules::fuse_gemv_pair)
            .with_rule_for("fuse_gemv_pair_tail", crate::rewrite_rules::fuse_gemv_pair_tail)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_gemm_epilogue", crate::rewrite_rules::fuse_gemm_epilogue)
            .with_rule_for("fuse_conv_epilogue", crate::rewrite_rules::fuse_conv_epilogue)
            .with_rule_for("fuse_deconv_epilogue", crate::rewrite_rules::fuse_deconv_epilogue)
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

/// Explodes the `Sdpa` nodes the flash kernel cannot take into primitives,
/// which translate one by one.
fn flatten_unfused_sdpa(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &tract_transformers::ops::sdpa::Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(!crate::kernels::sdpa::flash_sdpa_supported(&model.node_input_facts(node.id)?, op));
    op.patch_sdpa(model, node)
}

/// The inputs a kernel reads as bool, bytes packed four to a word. Every other
/// input of a translated node is a float.
fn reads_bool_input(node: &TypedNode, slot: usize) -> bool {
    slot == 1 && node.op_is::<tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax>()
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
    rule_if!(input_facts.iter().enumerate().all(|(slot, f)| crate::utils::is_supported_fact(f)
        || (f.datum_type == bool::datum_type() && reads_bool_input(node, slot))));

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

pub(crate) fn convert_const(op: &Const) -> TractResult<Const> {
    let typed_fact: TypedFact = Arc::clone(op.val()).try_into()?;
    let wgpu_fact = if let Some(of) = op.exotic_fact() {
        DeviceFact::from_host(typed_fact.with_exotic_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };
    let wgpu_const = op.val().clone().into_device()?.into_tensor().into_arc_tensor();
    Const::new_with_exotic_fact(wgpu_const, Box::new(wgpu_fact))
}

/// Where each source outlet landed in the model being translated.
pub(crate) type Mapping = HashMap<OutletId, OutletId>;

/// Wires one node into `target` from its mapped inputs, moving to the device what it reads there,
/// or declines with `None`. For translations the per-op registry can't express: several nodes, an
/// input left on the host, or inputs the registry's dtype gate refuses.
type TranslateNodeFn =
    fn(&TypedModel, &TypedNode, &mut TypedModel, &Mapping) -> TractResult<Option<TVec<OutletId>>>;

fn translator_for(node: &TypedNode) -> Option<TranslateNodeFn> {
    static MAP: OnceLock<HashMap<TypeId, TranslateNodeFn>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        HashMap::from([
            (TypeId::of::<PrefixMatMul>(), ops::matmul::translate_gemm as TranslateNodeFn),
            (TypeId::of::<Conv>(), ops::conv::translate_conv),
            (TypeId::of::<Deconv>(), ops::deconv::translate_deconv),
            (TypeId::of::<GroupNorm>(), crate::kernels::group_norm::translate_group_norm),
            (TypeId::of::<Resize>(), crate::kernels::resize::translate_resize),
            (TypeId::of::<Gather>(), crate::kernels::gather::translate_gather),
            (TypeId::of::<Iff>(), crate::kernels::nan_replace::translate_nan_replace),
            (TypeId::of::<Const>(), translate_const),
        ])
    });
    map.get(&(*node.op).type_id()).copied()
}

/// The node's `ix`-th input, moved to the device unless it is there already.
pub(crate) fn input_on_device(
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
    ix: usize,
) -> TractResult<OutletId> {
    let input = mapping[&node.inputs[ix]];
    if target.outlet_fact(input)?.as_device_fact().is_some() {
        return Ok(input);
    }
    Ok(target.wire_node(
        format!("{}.to-device-{ix}", node.name),
        tract_gpu::sync::DeviceSync::new(DeviceSyncKind::ToDevice),
        &[input],
    )?[0])
}

fn translate_const(
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(op = node.op_as::<Const>());
    rule_if!(crate::utils::is_supported_dt(op.val().datum_type()) && op.exotic_fact().is_none());
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    target.wire_node(node.name.clone(), convert_const(op)?, &device_inputs).map(Some)
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for WgpuTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(translate) = translator_for(node)
            && let Some(outlet_ids) = translate(source, node, target, mapping)?
        {
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
            // A constant is read where it already is, on the host: bringing back the copy
            // uploaded for device consumers would be a readback, which a browser cannot block on.
            let mut cpu_inputs = tvec![];
            for (ix, input) in node.inputs.iter().enumerate() {
                let mapped = mapping[input];
                if let Some(konst) =
                    source.node(input.node).op_as::<Const>().filter(|k| k.exotic_fact().is_none())
                {
                    let name = format!("{}.host-const-{ix}", node.name);
                    cpu_inputs.push(target.add_const(name, konst.val().clone())?);
                } else if target.outlet_fact(mapped)?.as_device_fact().is_some() {
                    let sync = tract_gpu::sync::DeviceSync::new(DeviceSyncKind::ToHost);
                    cpu_inputs.push(
                        target.wire_node(format!("{}.to-cpu-{ix}", node.name), sync, &[mapped])?[0],
                    );
                } else {
                    cpu_inputs.push(mapped);
                }
            }
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
    rule_if!(matches!(op.reducer, Sum | Mean | Prod | Min | Max | Any | All));
    let fact = model.outlet_fact(node.inputs[0])?;
    rule_if!(
        !(crate::kernels::reduce::is_trailing_sum_run(op, fact.shape.dims())
            && fact.datum_type.is_float())
    );
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
