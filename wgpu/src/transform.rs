use std::any::TypeId;
use std::collections::HashMap;
use std::sync::OnceLock;

use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
#[cfg(not(target_arch = "wasm32"))]
use tract_gpu::sync::sync_model_outputs_if_required;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::IntoDevice;

use crate::context::wgpu_context;

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
        wgpu_context();
        *model = self.translate_model(model)?;
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
    rule_if!(input_facts.iter().all(|f| crate::utils::is_supported_fact(f)));

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
        if let Some(op) = node.op_as::<Const>()
            && crate::utils::is_supported_dt(op.val().datum_type())
            && op.exotic_fact().is_none()
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
