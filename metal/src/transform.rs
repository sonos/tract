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
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_sdpa::rewire_sdpa;
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::rewrite_rules::rms_norm::remove_rms_norm_cast;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required, sync_model_outputs_if_required};
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_gpu::utils::as_quant_fact;

use crate::rewrite_rules;

/// A registered translator that can convert a core op into a Metal GPU op.
/// Each kernel module submits one (or more) of these via [`register_metal_op!`].
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

        rewire_sdpa(model)?;
        rewrite_einsum_to_prefix_matmul(model, false)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::<MetalTransform>::default()
            .with_rule_for("untranspose-matmul-output", rewrite_rules::untranspose_matmul_output)
            .with_rule_for("add-broadcast-pre-matmul", rewrite_rules::add_broadcast_pre_matmul)
            .rewrite(self, model)?;

        Rewriter::default()
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
    if !input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type)) {
        return Ok(None);
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
        // Const: inline conversion, not a GPU op
        if let Some(op) = node.op_as::<Const>() {
            if DeviceTensor::is_supported_dt(op.val().datum_type()) {
                let device_inputs =
                    sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
                let outlet_ids =
                    target.wire_node(node.name.clone(), convert_const(op)?, &device_inputs)?;
                return sync_model_outputs_if_required(source, node, target, outlet_ids);
            }
        }

        // Single-op translation
        if let Some(gpu_op) = try_make_metal_op(source, node)? {
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

    let resolved_gemm_impl = resolve_gemm_impl(gemm_impl, input_facts.clone())?;
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
                let perm_a_op = tract_gpu::ops::change_axes::GpuAxisOp::new(
                    AxisOp::Move(rank - 2, rank - 1),
                    "Metal",
                    crate::kernels::array::metal_copy_nd_dispatch,
                );
                let perm_a_name = node.name.clone() + ".perm_a";
                inputs[a_pos] = target.wire_node(perm_a_name, perm_a_op, &[inputs[a_pos]])?[0];
            }

            if input_facts[0].datum_type == DatumType::F16 {
                let in_cast_op = metal_cast_new(DatumType::F32).unwrap();
                inputs[0] =
                    target.wire_node(node.name.clone() + ".in_cast", in_cast_op, &[inputs[0]])?[0];
            }

            if !op.transpose_b {
                ensure!(
                    as_quant_fact(input_facts[b_pos], &Q4_0).is_none(),
                    "Cannot transpose Q40 tensor"
                );

                let rank = input_facts[b_pos].rank();
                let perm_b_op = tract_gpu::ops::change_axes::GpuAxisOp::new(
                    AxisOp::Move(rank - 2, rank - 1),
                    "Metal",
                    crate::kernels::array::metal_copy_nd_dispatch,
                );
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

                let perm_out_op = tract_gpu::ops::change_axes::GpuAxisOp::new(
                    AxisOp::Move(rank - 2, rank - 1),
                    "Metal",
                    crate::kernels::array::metal_copy_nd_dispatch,
                );
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

    let expected_dt = model.node_output_facts(node.id)?[0].datum_type;

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
