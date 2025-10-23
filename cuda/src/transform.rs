use num_traits::Float;
use tract_core::dyn_clone::clone_box;
use tract_core::internal::*;
use tract_core::model::translator::Translate;
use tract_core::ops::array::{MultiBroadcastTo, PadMode, Slice, TypedConcat};
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::Comp;
use tract_core::ops::nn::{Reduce, Softmax};
use tract_core::tract_data::itertools::Itertools;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_sdpa::{causal_mask_as_extern, neutral_mask_for_full_attn};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::sync::{DeviceSync, DeviceSyncKind};
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::apply_rope::{ApplyRope, RotateHalf};
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCache;
use tract_transformers::ops::gelu_approximate::GeluApproximate;
use tract_transformers::ops::rms_norm::RmsNorm;
use tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax;
use tract_transformers::ops::sdpa::Sdpa;
use tract_transformers::ops::silu::Silu;

use crate::context::cuda_context;
use crate::kernels::matmul::GgmlGemm;
use crate::{kernels, ops, rewrite_rules};

#[derive(Debug, Default)]
pub struct CudaTransform;

impl ModelTransform for CudaTransform {
    fn name(&self) -> StaticName {
        "cuda-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        self.transform_up_to_phase(model, usize::MAX)
    }
}

impl CudaTransform {
    pub fn transform_up_to_phase(
        &self,
        model: &mut TypedModel,
        stop_at_phase: usize,
    ) -> TractResult<()> {
        // Init CUDA Context if not done previously
        cuda_context();

        rewrite_einsum_to_prefix_matmul(model, false)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::default()
            .with_rule_for("untranspose_matmul_output", rewrite_rules::untranspose_matmul_output)
            .with_rule_for("add_broadcast_pre_matmul", rewrite_rules::add_broadcast_pre_matmul)
            .with_rule_for("causal_mask_as_extern", causal_mask_as_extern)
            .with_rule_for("full_attn_mask_as_neutral", neutral_mask_for_full_attn)
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

        Rewriter::default()
            .with_rule_for("pad_q40_weights", rewrite_rules::pad_q40_weights)
            .rewrite(&(), model)?;
        Ok(())
    }

    fn sync_inputs_if_required(
        &self,
        model: &mut TypedModel,
        node: &TypedNode,
        mapping: &HashMap<OutletId, OutletId>,
        sync_kind: DeviceSyncKind,
    ) -> TractResult<TVec<OutletId>> {
        let mut mapped_inputs = tvec![];
        for (i_idx, i) in node.inputs.iter().enumerate() {
            let in_fact = model.outlet_fact_mut(mapping[i])?;
            match sync_kind {
                DeviceSyncKind::ToHost if in_fact.as_device_fact().is_some() => {
                    mapped_inputs.push(
                        model.wire_node(
                            format!("{}.to-cpu-{i_idx}", node.name),
                            DeviceSync::new(sync_kind),
                            &[mapping[i]],
                        )?[0],
                    );
                }
                DeviceSyncKind::ToDevice if in_fact.as_device_fact().is_none() => {
                    if let Some(ref konst) = in_fact.konst {
                        if konst.as_device_tensor().is_none() {
                            let device_konst =
                                konst.as_ref().clone().into_device()?.into_opaque_tensor();
                            let device_fact = DeviceFact::from_host(in_fact.clone())?;

                            *in_fact = TypedFact::dt_scalar(DatumType::Opaque)
                                .with_opaque_fact(device_fact);

                            in_fact.konst = Some(Arc::new(device_konst));
                            mapped_inputs.push(mapping[i]);
                            continue;
                        }
                    }
                    ensure!(
                        in_fact.datum_type.is_copy(),
                        "Only copy DatumType can be sync to Device: {:?}",
                        in_fact.datum_type
                    );

                    mapped_inputs.push(
                        model.wire_node(
                            format!("{}.to-device-{i_idx}", node.name),
                            DeviceSync::new(sync_kind),
                            &[mapping[i]],
                        )?[0],
                    );
                }
                _ => mapped_inputs.push(mapping[i]),
            }
        }
        Ok(mapped_inputs)
    }

    fn sync_model_outputs_if_required(
        &self,
        src: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        target_node_outlet_ids: TVec<OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut outputs = tvec![];
        for (o_idx, o) in target_node_outlet_ids.into_iter().enumerate() {
            // Add DeviceSync op for model output
            let is_src_output = src.outputs.contains(&OutletId::new(node.id, o_idx));
            if target.outlet_fact(o)?.as_device_fact().is_some() && is_src_output {
                let sync_output = target.wire_node(
                    format!("{}.to-host-{o_idx}-out", node.name),
                    DeviceSync::new(DeviceSyncKind::ToHost),
                    &[o],
                )?[0];
                outputs.push(sync_output);
            } else {
                outputs.push(o)
            }
        }
        Ok(outputs)
    }
}

fn can_translate_to_cuda_op(source: &TypedModel, node: &TypedNode) -> TractResult<bool> {
    let input_facts = source.node_input_facts(node.id)?.iter().map(|f| (*f).clone()).collect_vec();
    let input_dts = input_facts
        .iter()
        .map(|f| f.as_device_fact().map(|f| f.datum_type).unwrap_or(f.datum_type))
        .collect_vec();

    let in_dts_compatible =
        input_facts.iter().all(|fact| DeviceTensor::is_supported_dt(fact.datum_type));

    Ok(in_dts_compatible
        && (node
            .op_as::<Const>()
            .is_some_and(|op| DeviceTensor::is_supported_dt(op.val().datum_type()))
            || node
                .op_as::<Silu>()
                .is_some_and(|_| kernels::UnaryOps::is_supported_dt(input_dts[0]))
            || node.op_as::<ElementWiseOp>().is_some_and(|op| {
                kernels::UnaryOps::is_supported_dt(input_dts[0])
                    && map_element_wise_ops_to_cuda(op).is_some()
            })
            || node.op_as::<TypedBinOp>().is_some_and(|op| {
                map_binary_op_to_cuda(op).is_some_and(|op| op.0.is_supported_dt(input_dts[0]))
            })
            || node
                .op_as::<Comp>()
                .is_some_and(|op| convert_logic_op_to_cuda(op).0.is_supported_dt(input_dts[0]))
            || node
                .op_as::<Const>()
                .is_some_and(|op| DeviceTensor::is_supported_dt(op.val().datum_type()))
            || node.op_as::<Cast>().is_some_and(|op| {
                ops::CudaCast::is_supported_dt(input_dts[0]) && ops::CudaCast::new(op.to).is_some()
            })
            || node.op_is::<MultiBroadcastTo>()
            || node.op_is::<AxisOp>()
            || node.op_is::<Slice>()
            || node.op_is::<TypedConcat>()
            || node.op_is::<DynKeyValueCache>()
            || node.op_as::<Reduce>().is_some_and(|op| {
                kernels::nn::Reducer::is_supported_dt(input_dts[0])
                    && ops::CudaReduce::from_tract_core(op).is_ok()
            })
            || node.op_as::<Softmax>().is_some_and(|op| {
                kernels::nn::Softmax::is_supported_dt(input_dts[0])
                    && ops::CudaSoftmax::from_tract_core(op).is_ok()
            })
            || node
                .op_as::<ScaledMaskedSoftmax>()
                .is_some_and(|_| kernels::nn::ScaledMaskedSoftmax::is_supported_dt(input_dts[0]))
            || node
                .op_as::<RmsNorm>()
                .is_some_and(|_| kernels::nn::RmsNorm::is_supported_dt(input_dts[0]))
            || node
                .op_as::<RotateHalf>()
                .is_some_and(|_| kernels::array::RotateHalf::is_supported_dt(input_dts[0]))
            || node
                .op_as::<ApplyRope>()
                .is_some_and(|_| kernels::nn::ApplyRope::is_supported_dt(input_dts[0]))
            || node
                .op_as::<GeluApproximate>()
                .is_some_and(|_| kernels::nn::GeluApproximate::is_supported_dt(input_dts[0]))
            || node.op_as::<Sdpa>().is_some()
            || node.op_as::<PrefixMatMul>().is_some_and(|op| {
                !op.transpose_c
                    && op.quantize_output.is_none()
                    && (GgmlGemm.is_supported_dts(&input_facts)
                        || GgmlGemm
                            .is_supported_dts(&[input_facts[1].clone(), input_facts[0].clone()]))
            })))
}

fn convert_const(op: &Const) -> TractResult<Const> {
    let typed_fact: TypedFact = Arc::clone(op.val()).into();
    let cuda_fact = if let Some(of) = op.opaque_fact() {
        DeviceFact::from_host(typed_fact.with_opaque_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };

    let cuda_const = op.val().clone().into_device()?.into_opaque_tensor().into_arc_tensor();
    Const::new_with_opaque_fact(cuda_const, Box::new(cuda_fact))
}

macro_rules! map_unary_ops {
    ([$(($tract_unary_op:path, $cuda_unary_op:ident)),* $(,)?]) => {
        |op: &tract_core::ops::element_wise::ElementWiseOp| {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_unary_op>() {
                return Some($crate::ops::CudaUnaryOp(kernels::UnaryOps::$cuda_unary_op));
            })*
            return None;
        }
    };
}

fn map_element_wise_ops_to_cuda(op: &ElementWiseOp) -> Option<ops::CudaUnaryOp> {
    map_unary_ops!([
        (tract_core::ops::math::Abs, Abs),
        (tract_core::ops::math::Exp, Exp),
        (tract_core::ops::math::Ln, Ln),
        (tract_core::ops::nn::Sigmoid, Sigmoid),
        (tract_core::ops::math::Square, Sqr),
        (tract_core::ops::math::Sqrt, Sqrt),
        (tract_core::ops::math::Rsqrt, Rsqrt),
        (tract_core::ops::math::Recip, Recip),
        (tract_core::ops::math::Ceil, Ceil),
        (tract_core::ops::math::Floor, Floor),
        (tract_core::ops::math::Round, Round),
        (tract_core::ops::math::RoundHalfToEven, RoundHalfToEven),
        (tract_core::ops::math::Cos, Cos),
        (tract_core::ops::math::Acos, Acos),
        (tract_core::ops::math::Acosh, Acosh),
        (tract_core::ops::math::Cosh, Cosh),
        (tract_core::ops::math::Sin, Sin),
        (tract_core::ops::math::Asin, Asin),
        (tract_core::ops::math::Asinh, Asinh),
        (tract_core::ops::math::Sinh, Sinh),
        (tract_core::ops::math::Tan, Tan),
        (tract_core::ops::math::Atan, Atan),
        (tract_core::ops::math::Atanh, Atanh),
        (tract_core::ops::math::Tanh, Tanh),
        (tract_core::ops::math::Erf, Erf),
        (tract_core::ops::math::Neg, Neg),
    ])(op)
}

macro_rules! map_bin_ops {
    ([$(($tract_bin_op:path, $cuda_bin_op:ident)),* $(,)?]) => {
        |op: &TypedBinOp | {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::CudaBinOp(kernels::BinOps::$cuda_bin_op));
            })*
            return None;
        }
    };
}

#[allow(clippy::borrowed_box)]
fn map_binary_op_to_cuda(op: &TypedBinOp) -> Option<ops::CudaBinOp> {
    map_bin_ops!([
        (tract_core::ops::math::Mul, Mul),
        (tract_core::ops::math::Add, Add),
        (tract_core::ops::math::Div, Div),
        (tract_core::ops::math::Sub, Sub),
        (tract_core::ops::math::Pow, Pow),
        (tract_core::ops::logic::And, And),
        (tract_core::ops::logic::Or, Or),
    ])(op)
}

fn convert_logic_op_to_cuda(op: &Comp) -> ops::CudaBinOp {
    match op {
        Comp::Eq => ops::CudaBinOp(kernels::BinOps::Equals),
        Comp::NE => ops::CudaBinOp(kernels::BinOps::NotEquals),
        Comp::LT => ops::CudaBinOp(kernels::BinOps::Less),
        Comp::LTE => ops::CudaBinOp(kernels::BinOps::LessEqual),
        Comp::GT => ops::CudaBinOp(kernels::BinOps::Greater),
        Comp::GTE => ops::CudaBinOp(kernels::BinOps::GreaterEqual),
    }
}

fn convert_matmul_to_cuda(
    model: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &mut [OutletId],
    op: &PrefixMatMul,
) -> TractResult<TVec<OutletId>> {
    let mut input_facts = model.node_input_facts(node.id)?;
    // GGML kernel expects weights in second position and activations in first position
    // This avoid output transposition due to GGML column-major data expectations
    let mut swap_inputs = false;
    if !GgmlGemm.is_supported_dts(&[input_facts[0].clone(), input_facts[1].clone()])
        && GgmlGemm.is_supported_dts(&[input_facts[1].clone(), input_facts[0].clone()])
    {
        input_facts.swap(0, 1);
        inputs.swap(0, 1);
        swap_inputs = true;
    }

    let act_fact = input_facts[0];
    let weight_fact = input_facts[1];
    let outlets = inputs.split_at_mut(1);
    let act_outlet = &mut outlets.0[0];
    let weights_outlet = &mut outlets.1[0];

    let transpose_act = if swap_inputs { !op.transpose_b } else { op.transpose_a };
    let transpose_weight = if swap_inputs { !op.transpose_a } else { op.transpose_b };

    if transpose_act {
        let rank = act_fact.rank();
        let perm_act_op = ops::CudaAxisOp::from_tract_core(AxisOp::Move(rank - 2, rank - 1));
        let perm_act_name = node.name.clone() + ".perm_activs";
        *act_outlet = target.wire_node(perm_act_name, perm_act_op, &[*act_outlet])?[0];
    }

    if act_fact.datum_type == DatumType::F16 && as_quant_fact(weight_fact, &Q4_0).is_some() {
        let in_cast_op = ops::CudaCast::new(DatumType::F32).unwrap();
        *act_outlet =
            target.wire_node(node.name.clone() + ".in_cast", in_cast_op, &[*act_outlet])?[0];
    }

    if !transpose_weight {
        ensure!(as_quant_fact(weight_fact, &Q4_0).is_none(), "Cannot transpose Q40 tensor");

        let rank = weight_fact.rank();
        let perm_weights_op = ops::CudaAxisOp::from_tract_core(AxisOp::Move(rank - 2, rank - 1));
        let perm_weights_name = node.name.clone() + ".perm_weights";
        *weights_outlet =
            target.wire_node(perm_weights_name, perm_weights_op, &[*weights_outlet])?[0];
    }

    if as_quant_fact(weight_fact, &Q4_0).is_some() {
        let device_fact = target.outlet_fact(*act_outlet)?.to_device_fact()?;
        let quant_op = ops::CudaGgmlQuantQ81::new(device_fact.shape.clone())?;
        *act_outlet =
            target.wire_node(node.name.clone() + ".quant_activs", quant_op, &[*act_outlet])?[0];
    }
    let mut matmul_output =
        target.wire_node(node.name.clone(), *Box::new(ops::CudaGgmlGemm), inputs)?;

    if swap_inputs {
        let out_fact = target.outlet_fact(matmul_output[0])?;
        let rank = &out_fact
            .opaque_fact
            .clone()
            .map(|fact| fact.clarify_dt_shape().unwrap().1.len())
            .unwrap();

        let perm_out_op = ops::CudaAxisOp::from_tract_core(AxisOp::Move(rank - 2, rank - 1));
        matmul_output =
            target.wire_node(node.name.clone() + ".perm_out", perm_out_op, &matmul_output)?;
    }

    let out_fact = target.outlet_fact(matmul_output[0])?;
    let out_dt = out_fact.as_device_fact().map(|f| f.datum_type).unwrap_or(out_fact.datum_type);

    let expected_dt = model.node_output_facts(node.id)?[0].datum_type;
    if out_dt != expected_dt {
        ensure!(
            ops::CudaCast::is_supported_dt(out_dt),
            "Matmul output type cannot be casted to expected type"
        );
        let cast_op = ops::CudaCast::new(model.node_output_facts(node.id)?[0].datum_type).unwrap();
        matmul_output =
            target.wire_node(node.name.clone() + ".out_cast", cast_op, &matmul_output)?
    }
    Ok(matmul_output)
}

fn convert_sdpa_to_cuda_flash_attn(
    model: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &mut [OutletId],
    op: &Sdpa,
) -> TractResult<TVec<OutletId>> {
    let input_facts = model.node_input_facts(node.id)?;
    ensure!(!op.is_causal && input_facts.len() == 4);

    let q_fact = input_facts[0];
    let k_fact = input_facts[1];
    let v_fact = input_facts[2];
    let m_fact = input_facts[3];

    let splitted_i = inputs.split_at_mut(2);
    let qk = splitted_i.0.split_at_mut(1);
    let vm = splitted_i.1.split_at_mut(1);

    let q = &mut qk.0[0];
    let k = &mut qk.1[0];
    let v = &mut vm.0[0];
    let m = &mut vm.1[0];

    ensure!(k_fact.datum_type() == v_fact.datum_type());
    let cast_op = ops::CudaCast::new(DatumType::F16).unwrap();
    if k_fact.datum_type().unwrap() != DatumType::F16 {
        // Cast every inputs (except Q) to F16
        *k = target.wire_node(node.name.clone() + ".cast_k", cast_op.clone(), &[*k])?[0];
        *v = target.wire_node(node.name.clone() + ".cast_v", cast_op.clone(), &[*v])?[0];
    }

    let q_dt = q_fact.datum_type().unwrap();
    if q_dt != DatumType::F32 {
        *q = target.wire_node(
            node.name.clone() + ".cast_q",
            ops::CudaCast::new(DatumType::F32).unwrap(),
            &[*q],
        )?[0];
    }

    let q_rank = q_fact.rank();
    ensure!(input_facts[..3].iter().all(|i_f| i_f.rank() == q_rank));
    if q_rank != 4 {
        ensure!(q_fact.rank() == 3);
        let ax_op = ops::CudaAxisOp::from_tract_core(AxisOp::Add(1));
        *q = target.wire_node(node.name.clone() + ".reshape_q", ax_op.clone(), &[*q])?[0];
        *k = target.wire_node(node.name.clone() + ".reshape_k", ax_op.clone(), &[*k])?[0];
        *v = target.wire_node(node.name.clone() + ".reshape_v", ax_op, &[*v])?[0];
    }

    let out_dim = k_fact.shape[k_fact.rank() - 1].to_i64()?;
    ensure!(
        matches!(out_dim, 64 | 80 | 96 | 112 | 128 | 256),
        "Unsupported d for Cuda FlashAttn: {out_dim}"
    );

    // Pad S+P to next multiple of 256 for Flash Attention Op
    ensure!(k_fact.shape == v_fact.shape);

    let s_plus_p = k_fact.shape.dims()[q_fact.rank() - 2].clone();
    let s_plus_p_to_pad = ((s_plus_p.clone() + 255) / 256) * 256 - s_plus_p;
    let mut padding_kv = vec![(TDim::Val(0), TDim::Val(0)); 4];
    padding_kv[2].1 = s_plus_p_to_pad.clone();

    *k = target.wire_node(
        node.name.clone() + ".pad_k",
        ops::CudaPad::new(
            padding_kv.clone(),
            PadMode::Constant(tensor0(f16::from_f32(0f32)).into()),
        )?,
        &[*k],
    )?[0];
    *v = target.wire_node(
        node.name.clone() + ".pad_v",
        ops::CudaPad::new(padding_kv, PadMode::Constant(tensor0(f16::from_f32(0f32)).into()))?,
        &[*v],
    )?[0];

    // Handle mask,
    if m_fact.datum_type().unwrap() != DatumType::F16 {
        *m = target.wire_node(node.name.clone() + ".cast_m", cast_op, &[*m])?[0];
    }

    if m_fact.rank() != 4 {
        let ax_op = ops::CudaAxisOp::from_tract_core(AxisOp::Reshape(
            0,
            tvec![],
            tvec![TDim::Val(1); 4 - m_fact.rank()],
        ));
        *m = target.wire_node(node.name.clone() + ".reshape_m", ax_op.clone(), &[*m])?[0];
    }

    // Seq dimension for mask also need padding
    let s = q_fact.shape.dims()[q_fact.rank() - 2].clone();
    let mut padding_m = vec![(TDim::Val(0), TDim::Val(0)); 4];

    padding_m[2].1 = ((s.clone() + 15) / 16) * 16 - s;
    padding_m[3].1 = s_plus_p_to_pad;
    *m = target.wire_node(
        node.name.clone() + ".pad_m",
        ops::CudaPad::new(padding_m, PadMode::Constant(tensor0(-f16::infinity()).into()))?,
        &[*m],
    )?[0];

    let scale = op
        .scale
        .as_ref()
        .map(|s| *s.to_scalar::<f32>().unwrap())
        .unwrap_or(1.0 / (out_dim as f32).sqrt());
    let sdpa_op = ops::CudaFlashAttention::new(scale, false);

    let mut out = target.wire_node(node.name.clone(), sdpa_op, inputs)?;

    if q_rank != 4 {
        out = target.wire_node(
            node.name.clone() + ".reshape_out",
            ops::CudaAxisOp::from_tract_core(AxisOp::Rm(1)),
            &out,
        )?;
    }

    if q_dt != DatumType::F32 {
        out = target.wire_node(
            node.name.clone() + ".cast_out",
            ops::CudaCast::new(q_dt).unwrap(),
            &out,
        )?;
    }
    Ok(out)
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for CudaTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let translatable = can_translate_to_cuda_op(source, node)?;

        if translatable {
            let mut device_inputs =
                self.sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;

            let outlet_ids: TVec<OutletId> = if let Some(op) = node.op_as::<PrefixMatMul>() {
                convert_matmul_to_cuda(source, node, target, &mut device_inputs, op)?
            } else if let Some(op) = node.op_as::<Sdpa>() {
                convert_sdpa_to_cuda_flash_attn(source, node, target, &mut device_inputs, op)?
            } else {
                let op: Box<dyn TypedOp> = if let Some(op) = node.op_as::<Const>() {
                    Box::new(convert_const(op)?)
                } else if let Some(op) = node.op_as::<ElementWiseOp>() {
                    Box::new(map_element_wise_ops_to_cuda(op).unwrap())
                } else if let Some(op) = node.op_as::<TypedBinOp>() {
                    Box::new(map_binary_op_to_cuda(op).unwrap())
                } else if let Some(op) = node.op_as::<Comp>() {
                    Box::new(convert_logic_op_to_cuda(op))
                } else if let Some(_op) = node.op_as::<Silu>() {
                    Box::new(ops::CudaUnaryOp(kernels::UnaryOps::Silu))
                } else if let Some(op) = node.op_as::<MultiBroadcastTo>() {
                    Box::new(ops::CudaMultiBroadcastTo::new(op.shape.clone()))
                } else if let Some(op) = node.op_as::<Cast>() {
                    Box::new(ops::CudaCast::new(op.to).unwrap())
                } else if let Some(op) = node.op_as::<AxisOp>() {
                    let in_fact = source.node_input_facts(node.id)?[0];
                    Box::new(ops::CudaAxisOp::from_tract_core_with_fact(op.clone(), in_fact))
                } else if let Some(op) = node.op_as::<Slice>() {
                    Box::new(ops::CudaSlice::from_tract_core(op.clone()))
                } else if let Some(op) = node.op_as::<TypedConcat>() {
                    Box::new(ops::CudaConcat::from_tract_core(op))
                } else if let Some(op) = node.op_as::<DynKeyValueCache>() {
                    Box::new(ops::CudaDynKVCache::from_tract_transformers(op))
                } else if let Some(op) = node.op_as::<Reduce>() {
                    Box::new(ops::CudaReduce::from_tract_core(op)?)
                } else if let Some(op) = node.op_as::<Softmax>() {
                    Box::new(ops::CudaSoftmax::from_tract_core(op)?)
                } else if let Some(op) = node.op_as::<ScaledMaskedSoftmax>() {
                    Box::new(ops::CudaScaledMaskedSoftmax { scale: op.scale.clone() })
                } else if let Some(_op) = node.op_as::<RotateHalf>() {
                    Box::new(ops::CudaRotateHalf)
                } else if let Some(_op) = node.op_as::<ApplyRope>() {
                    Box::new(ops::CudaApplyRope)
                } else if let Some(op) = node.op_as::<RmsNorm>() {
                    Box::new(ops::CudaRmsNorm::new(op.axis, op.eps.clone()))
                } else if let Some(op) = node.op_as::<GeluApproximate>() {
                    Box::new(ops::CudaGeluApproximate { fast_impl: op.fast_impl })
                } else {
                    bail!("Failed to translate a supported CUDA Op")
                };
                target.wire_node(node.name.clone(), op, &device_inputs)?
            };
            self.sync_model_outputs_if_required(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                self.sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToHost)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_prefix_matmul_transform_f32_f16() -> TractResult<()> {
        let mut model = TypedModel::default();
        let (b, m, k, n) = (1, 16, 128, 32);

        let a_fact = TypedFact::dt_shape(DatumType::F32, &[b, m, k]);
        let b_fact = TypedFact::dt_shape(DatumType::F16, &[b, k, n]);

        let source_a = model.add_source("a", a_fact)?;
        let source_b = model.add_source("b", b_fact)?;

        let op = PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: Some(DatumType::F32),
        };

        let matmul_out = model.wire_node("matmul", op, &[source_a, source_b])?;
        model.set_output_outlets(&matmul_out)?;

        let tensor_a = Tensor::zero::<f32>(&[b, m, k])?;
        let tensor_b = Tensor::zero::<f16>(&[b, k, n])?;
        let inputs = tvec!(tensor_a.into(), tensor_b.into());

        let transform = CudaTransform::default();
        transform.transform(&mut model)?;

        let cuda_runnable = model.into_runnable()?;
        let _ = cuda_runnable.run(inputs)?;
        Ok(())
    }
}
