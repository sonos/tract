use crate::fact::MetalTypedFactExt;
use crate::kernels::array::RotateHalf;
use crate::kernels::matmul::{GemmKernel, GgmlGemm, MetalGemmImplKind, MfaGemm, MlxGemm};
use crate::kernels::nn::{
    ApplyRope, NewGelu, Reducer, RmsNorm, ScaledMaskedSoftmax, Silu, Softmax,
};
use crate::ops::{self, MetalSync, MetalSyncKind};

use crate::rewrite_rules;
use crate::rewrite_rules::{
    BasicApplyRope, BasicNewGelu, BasicRmsNorm, BasicRotateHalf, BasicScaledMaskedSoftmax,
    BasicSilu,
};
use crate::tensor::MetalTensorExt;
use crate::utils::{as_q40_fact, as_q40_tensor};
use crate::{IntoMetal, MetalFact, MetalTensor};
use std::borrow::Cow;
use std::fmt::Debug;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::array::{MultiBroadcastTo, Slice, TypedConcat};
use tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::{rewrite_einsums_as_matmul, BasicMatMul};
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::Comp;
use tract_core::ops::nn::{Reduce, Softmax as CoreSoftmax};
use tract_core::tract_linalg::frame::block_quant::BlockQuantValue;
use tract_core::transform::ModelTransform;
use tract_itertools::Itertools;

impl MetalGemmImplKind {
    pub fn variants() -> Vec<MetalGemmImplKind> {
        vec![Self::Mlx, Self::Mfa]
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
    pub gemm_impl: MetalGemmImplKind,
}

impl ModelTransform for MetalTransform {
    fn name(&self) -> Cow<str> {
        "metal-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        self.transform_up_to_phase(model, usize::MAX)
    }
}

impl MetalTransform {
    pub fn transform_up_to_phase(
        &self,
        model: &mut TypedModel,
        stop_at_phase: usize,
    ) -> TractResult<()> {
        rewrite_einsums_as_matmul(model)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::default()
            .with_rule_for("as-rms-norm", rewrite_rules::as_rms_norm_rule)
            .with_rule_for("remove_rms_norm_cast", rewrite_rules::remove_rms_norm_cast)
            .with_rule_for("as-silu", rewrite_rules::as_silu_rule)
            .with_rule_for("as-new-gelu", rewrite_rules::as_new_gelu_rule)
            .with_rule_for("as-rotate-half", rewrite_rules::as_rotate_half_rule)
            .with_rule_for("as-apply-rope", rewrite_rules::as_apply_rope_rule)
            .with_rule_for("as-scaled-masked-softmax", rewrite_rules::as_scaled_masked_softmax_rule)
            .with_rule_for("untranspose-matmul-output", rewrite_rules::untranspose_matmul_output)
            .rewrite(&(), model)?;

        if stop_at_phase == 1 {
            return Ok(());
        }

        *model = self.translate_model(model)?;

        if stop_at_phase == 2 {
            return Ok(());
        }

        Rewriter::default()
            .with_rule_for("rewire-metal-sync", rewrite_rules::rewire_metal_sync)
            .with_rule_for(
                "rewire-metal-sync-after-const",
                rewrite_rules::rewire_metal_sync_after_const,
            )
            .with_rule_for("fuse_axis_op", rewrite_rules::fuse_axis_op)
            .rewrite(&(), model)?;
        Ok(())
    }

    fn sync_inputs_if_required(
        &self,
        model: &mut TypedModel,
        node: &TypedNode,
        mapping: &HashMap<OutletId, OutletId>,
        sync_kind: MetalSyncKind,
    ) -> TractResult<TVec<OutletId>> {
        let mut mapped_inputs = tvec![];
        for (i_idx, i) in node.inputs.iter().enumerate() {
            let in_fact = model.outlet_fact_mut(mapping[i])?;
            match sync_kind {
                MetalSyncKind::ToCpu if in_fact.as_metal_fact().is_some() => {
                    mapped_inputs.push(
                        model.wire_node(
                            format!("{}.to-cpu-{i_idx}", node.name),
                            MetalSync::new(sync_kind),
                            &[mapping[i]],
                        )?[0],
                    );
                }
                MetalSyncKind::ToGpu if in_fact.as_metal_fact().is_none() => {
                    if let Some(ref konst) = in_fact.konst {
                        if konst.as_metal_tensor().is_none() {
                            let konst_metal =
                                konst.as_ref().clone().into_metal()?.into_opaque_tensor();
                            let metal_fact = MetalFact::from_cpu(in_fact.clone())?;

                            *in_fact = TypedFact::dt_scalar(DatumType::Opaque)
                                .with_opaque_fact(metal_fact);

                            in_fact.konst = Some(Arc::new(konst_metal));
                            mapped_inputs.push(mapping[i]);
                            continue;
                        }
                    }
                    ensure!(
                        in_fact.datum_type.is_copy(),
                        "Only copy DatumType can be sync to GPU: {:?}",
                        in_fact.datum_type
                    );

                    mapped_inputs.push(
                        model.wire_node(
                            format!("{}.to-gpu-{i_idx}", node.name),
                            MetalSync::new(sync_kind),
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
            // Add MetalSync op for model output
            let is_src_output = src.outputs.contains(&OutletId::new(node.id, o_idx));
            if target.outlet_fact(o)?.as_metal_fact().is_some() && is_src_output {
                let sync_output = target.wire_node(
                    format!("{}.to-cpu-{o_idx}-out", node.name),
                    MetalSync::new(MetalSyncKind::ToCpu),
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

fn can_translate_to_metal_op(
    source: &TypedModel,
    node: &TypedNode,
    gemm_impl: MetalGemmImplKind,
) -> TractResult<bool> {
    let input_facts = source.node_input_facts(node.id)?.iter().map(|f| (*f).clone()).collect_vec();
    let input_dts = input_facts
        .iter()
        .map(|f| f.as_metal_fact().map(|f| f.datum_type).unwrap_or(f.datum_type))
        .collect_vec();

    let in_dts_metal_compatible =
        input_facts.iter().all(|fact| MetalTensor::is_supported_dt(fact.datum_type));

    Ok(in_dts_metal_compatible
        && (node
            .op_as::<ElementWiseOp>()
            .is_some_and(|op| map_element_wise_ops_to_metal(op).is_some())
            || node.op_as::<TypedBinOp>().is_some_and(|op| map_bin_ops_to_metal(&op.0).is_some())
            || node.op_is::<Comp>()
            || node.op_is::<MultiBroadcastTo>()
            || node.op_as::<BasicMatMul>().is_some_and(|op| {
                !op.transpose_c
                    && op.quantize_output.is_none()
                    && check_matmul_in_dts(gemm_impl, &input_facts)
            })
            || node
                .op_as::<Const>()
                .is_some_and(|op| MetalTensor::is_supported_dt(op.0.datum_type()))
            || node.op_as::<Cast>().is_some_and(|op| {
                ops::MetalCast::is_supported_dt(input_dts[0])
                    && ops::MetalCast::new(op.to).is_some()
            })
            || node.op_is::<AxisOp>()
            || node.op_is::<Slice>()
            || node.op_is::<TypedConcat>()
            || node.op_as::<Reduce>().is_some_and(|op| {
                Reducer::is_supported_dt(input_dts[0])
                    && ops::MetalReduce::from_tract_core(op).is_ok()
            })
            || node.op_as::<CoreSoftmax>().is_some_and(|op| {
                Softmax::is_supported_dt(input_dts[0])
                    && ops::MetalSoftmax::from_tract_core(op).is_ok()
            })
            || node
                .op_as::<BasicScaledMaskedSoftmax>()
                .is_some_and(|_| ScaledMaskedSoftmax::is_supported_dt(input_dts[0]))
            || node
                .op_as::<BasicRmsNorm>()
                .is_some_and(|_| RmsNorm::is_supported_dt(input_dts[0]))
            || node
                .op_as::<BasicRotateHalf>()
                .is_some_and(|_| RotateHalf::is_supported_dt(input_dts[0]))
            || node
                .op_as::<BasicApplyRope>()
                .is_some_and(|_| ApplyRope::is_supported_dt(input_dts[0]))
            || node.op_as::<BasicSilu>().is_some_and(|_| Silu::is_supported_dt(input_dts[0]))
            || node
                .op_as::<BasicNewGelu>()
                .is_some_and(|_| NewGelu::is_supported_dt(input_dts[0]))))
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for MetalTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let translatable = can_translate_to_metal_op(source, node, self.gemm_impl)?;

        if translatable {
            let mut gpu_inputs =
                self.sync_inputs_if_required(target, node, mapping, MetalSyncKind::ToGpu)?;

            let outlet_ids: TVec<OutletId> = if let Some(op) = node.op_as::<BasicMatMul>() {
                convert_matmul_to_metal(source, node, target, &mut gpu_inputs, op, self.gemm_impl)?
            } else {
                let op: Box<dyn TypedOp> = if let Some(op) = node.op_as::<ElementWiseOp>() {
                    Box::new(map_element_wise_ops_to_metal(op).unwrap())
                } else if let Some(op) = node.op_as::<TypedBinOp>() {
                    Box::new(map_bin_ops_to_metal(&op.0).unwrap())
                } else if let Some(op) = node.op_as::<Comp>() {
                    Box::new(convert_logic_ops_to_metal(op))
                } else if let Some(op) = node.op_as::<MultiBroadcastTo>() {
                    Box::new(ops::MetalMultiBroadcastTo::new(op.shape.clone()))
                } else if let Some(op) = node.op_as::<Const>() {
                    Box::new(convert_const(op)?)
                } else if let Some(op) = node.op_as::<Cast>() {
                    Box::new(ops::MetalCast::new(op.to).unwrap())
                } else if let Some(op) = node.op_as::<AxisOp>() {
                    let in_fact = source.node_input_facts(node.id)?[0];
                    Box::new(ops::MetalAxisOp::from_tract_core_with_fact(op.clone(), in_fact))
                } else if let Some(op) = node.op_as::<Slice>() {
                    Box::new(ops::MetalSlice::from_tract_core(op.clone()))
                } else if let Some(op) = node.op_as::<TypedConcat>() {
                    Box::new(ops::MetalConcat::from_tract_core(op))
                } else if let Some(op) = node.op_as::<Reduce>() {
                    Box::new(ops::MetalReduce::from_tract_core(op).unwrap())
                } else if let Some(op) = node.op_as::<CoreSoftmax>() {
                    Box::new(ops::MetalSoftmax::from_tract_core(op).unwrap())
                } else if let Some(op) = node.op_as::<BasicScaledMaskedSoftmax>() {
                    Box::new(ops::MetalScaledMaskedSoftmax { scale: op.scale.clone() })
                } else if let Some(op) = node.op_as::<BasicRmsNorm>() {
                    Box::new(ops::MetalRmsNorm::new(op.axis, op.eps.clone()))
                } else if let Some(_op) = node.op_as::<BasicRotateHalf>() {
                    Box::new(ops::MetalRotateHalf)
                } else if let Some(_op) = node.op_as::<BasicApplyRope>() {
                    Box::new(ops::MetalApplyRope)
                } else if let Some(_op) = node.op_as::<BasicSilu>() {
                    Box::new(ops::MetalSilu)
                } else if let Some(_op) = node.op_as::<BasicNewGelu>() {
                    Box::new(ops::MetalNewGelu)
                } else {
                    bail!("Failed to translate a supported Metal Op")
                };
                target.wire_node(node.name.clone(), op, &gpu_inputs)?
            };
            self.sync_model_outputs_if_required(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                self.sync_inputs_if_required(target, node, mapping, MetalSyncKind::ToCpu)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}

macro_rules! map_bin_ops {
    ([$(($tract_bin_op:path, $metal_bin_op:ident)),* $(,)?]) => {
        |op: &Box<dyn tract_core::ops::binary::BinMiniOp >| {
            $(if let Some(_op) = op.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::binary::MetalBinOp($crate::ops::binary::BinOps::$metal_bin_op));
            })*
            return None;
        }
    };
}

macro_rules! map_element_wise_ops {
    ([$(($tract_bin_op:path, $metal_bin_op:ident)),* $(,)?]) => {
        |op: &tract_core::ops::element_wise::ElementWiseOp| {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::element_wise::MetalElementWiseOp($crate::ops::element_wise::ElementWiseOps::$metal_bin_op));
            })*
            return None;
        }
    };
}

fn check_matmul_in_dts(gemm_impl: MetalGemmImplKind, in_facts: &[TypedFact]) -> bool {
    match gemm_impl {
        MetalGemmImplKind::Mlx => MlxGemm.is_supported_dts(in_facts),
        MetalGemmImplKind::Mfa => MfaGemm.is_supported_dts(in_facts),
        MetalGemmImplKind::Ggml => {
            GgmlGemm.is_supported_dts(in_facts)
                || GgmlGemm.is_supported_dts(&[in_facts[1].clone(), in_facts[0].clone()])
        }
    }
}

fn convert_matmul_to_metal(
    model: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &mut [OutletId],
    op: &BasicMatMul,
    gemm_impl: MetalGemmImplKind,
) -> TractResult<TVec<OutletId>> {
    let mut matmul_output = match gemm_impl {
        MetalGemmImplKind::Mlx => {
            let op = ops::MetalGemm::<MlxGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Mfa => {
            let op = ops::MetalGemm::<MfaGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Ggml => {
            let mut input_facts: tract_smallvec::SmallVec<[&TypedFact; 4]> =
                model.node_input_facts(node.id)?;

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
                assert!(as_q40_fact(input_facts[b_pos]).is_none(), "Cannot transpose Q40 tensor");

                let rank = input_facts[a_pos].rank();
                let perm_a_op = ops::change_axes::MetalAxisOp::from_tract_core(AxisOp::Move(
                    rank - 2,
                    rank - 1,
                ));
                let perm_a_name = node.name.clone() + ".perm_a";
                inputs[a_pos] = target.wire_node(perm_a_name, perm_a_op, &[inputs[a_pos]])?[0];
            }

            if input_facts[0].datum_type == DatumType::F16 {
                let in_cast_op = ops::MetalCast::new(DatumType::F32).unwrap();
                inputs[0] =
                    target.wire_node(node.name.clone() + ".in_cast", in_cast_op, &[inputs[0]])?[0];
            }

            if !op.transpose_b {
                assert!(as_q40_fact(input_facts[b_pos]).is_none(), "Cannot transpose Q40 tensor");

                let rank = input_facts[b_pos].rank();
                let perm_b_op = ops::change_axes::MetalAxisOp::from_tract_core(AxisOp::Move(
                    rank - 2,
                    rank - 1,
                ));
                let perm_b_name = node.name.clone() + ".perm_b";
                inputs[b_pos] = target.wire_node(perm_b_name, perm_b_op, &[inputs[b_pos]])?[0];
            }
            let op = ops::MetalGemm::<GgmlGemm>::new(false, true);
            let mut matmul_output = target.wire_node(node.name.clone(), op, inputs)?;

            if swap_inputs {
                let out_fact = target.outlet_fact(matmul_output[0])?;
                let rank = &out_fact
                    .opaque_fact
                    .clone()
                    .map(|fact| fact.clarify_dt_shape().unwrap().1.len())
                    .unwrap();

                let perm_out_op = ops::change_axes::MetalAxisOp::from_tract_core(AxisOp::Move(
                    rank - 2,
                    rank - 1,
                ));
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
    let out_dt = out_fact.to_metal_fact().map(|f| f.datum_type).unwrap_or(out_fact.datum_type);

    let expected_dt = model.node_output_facts(node.id)?[0].datum_type;

    if out_dt != expected_dt {
        ensure!(
            ops::MetalCast::is_supported_dt(out_dt),
            "Matmul output type cannot be casted to expected type"
        );
        let cast_op = ops::MetalCast::new(model.node_output_facts(node.id)?[0].datum_type).unwrap();
        matmul_output =
            target.wire_node(node.name.clone() + ".out_cast", cast_op, &matmul_output)?
    }
    Ok(matmul_output)
}

#[allow(clippy::borrowed_box)]
fn map_bin_ops_to_metal(op: &Box<dyn BinMiniOp>) -> Option<ops::MetalBinOp> {
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

fn convert_logic_ops_to_metal(op: &Comp) -> ops::MetalBinOp {
    match op {
        Comp::Eq => ops::MetalBinOp(ops::binary::BinOps::Equals),
        Comp::NE => ops::MetalBinOp(ops::binary::BinOps::NotEquals),
        Comp::LT => ops::MetalBinOp(ops::binary::BinOps::Less),
        Comp::LTE => ops::MetalBinOp(ops::binary::BinOps::LessEqual),
        Comp::GT => ops::MetalBinOp(ops::binary::BinOps::Greater),
        Comp::GTE => ops::MetalBinOp(ops::binary::BinOps::GreaterEqual),
    }
}

fn convert_const(op: &Const) -> TractResult<Const> {
    let (tensor, metal_fact) = if let Some(curr_bqv) = as_q40_tensor(op.0.view().tensor) {
        let mut bqv = curr_bqv.clone();
        crate::utils::tract_to_gguf_q4_0_packing(&mut (bqv.value))?;

        let bqv = BlockQuantValue { value: bqv.value, fact: bqv.fact };
        (
            tensor0(Opaque(Arc::new(bqv))).broadcast_into_rank(op.0.rank())?.into_arc_tensor(),
            MetalFact::from_cpu(Arc::clone(&op.0).into())?,
        )
    } else {
        (op.0.clone(), MetalFact::from_cpu(Arc::clone(&op.0).into())?)
    };
    let metal_const = tensor.into_metal()?.into_opaque_tensor().into_arc_tensor();
    Ok(Const::new_with_opaque_fact(metal_const, Box::new(metal_fact)))
}

fn map_element_wise_ops_to_metal(op: &ElementWiseOp) -> Option<ops::MetalElementWiseOp> {
    map_element_wise_ops!([
        (tract_core::ops::math::Abs, Abs),
        (tract_core::ops::math::Exp, Exp),
        (tract_core::ops::math::Ln, Ln),
        (tract_core::ops::nn::Sigmoid, Sigmoid),
        (tract_core::ops::math::Square, Square),
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
