use crate::context::metal_context;
use crate::kernels::matmul::{GemmKernel, GgmlGemm, MetalGemmImplKind, MfaGemm, MlxGemm};
use crate::{kernels, ops};
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::sync::{DeviceSync, DeviceSyncKind};

use crate::rewrite_rules;
use std::borrow::Cow;
use std::fmt::Debug;
use std::str::FromStr;
use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::array::{MultiBroadcastTo, Slice, TypedConcat};
use tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::prefix_matmul::{rewrite_einsum_to_prefix_matmul, PrefixMatMul};
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::Comp;
use tract_core::ops::nn::{Reduce, Softmax as CoreSoftmax};
use tract_core::transform::ModelTransform;
use tract_gpu::fact::DeviceFact;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};
use tract_gpu::utils::as_q40_fact;
use tract_itertools::Itertools;
use tract_transformers::ops::apply_rope::{ApplyRope, RotateHalf};
use tract_transformers::ops::gelu_approximate::GeluApproximate;
use tract_transformers::ops::rms_norm::RmsNorm;
use tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax;
use tract_transformers::ops::silu::Silu;

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
    fn name(&self) -> Cow<str> {
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

        rewrite_einsum_to_prefix_matmul(model)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::<MetalTransform>::default()
            .with_rule_for("untranspose-matmul-output", rewrite_rules::untranspose_matmul_output)
            .with_rule_for("add-broadcast-pre-matmul", rewrite_rules::add_broadcast_pre_matmul)
            .rewrite(self, model)?;

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
                            let konst_metal =
                                konst.as_ref().clone().into_device()?.into_opaque_tensor();
                            let metal_fact = DeviceFact::from_host(in_fact.clone())?;

                            *in_fact = TypedFact::dt_scalar(DatumType::Opaque)
                                .with_opaque_fact(metal_fact);

                            in_fact.konst = Some(Arc::new(konst_metal));
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

fn can_translate_to_metal_op(source: &TypedModel, node: &TypedNode) -> TractResult<bool> {
    let input_facts = source.node_input_facts(node.id)?.iter().map(|f| (*f).clone()).collect_vec();
    let input_dts = input_facts
        .iter()
        .map(|f| f.as_device_fact().map(|f| f.datum_type).unwrap_or(f.datum_type))
        .collect_vec();

    let in_dts_metal_compatible =
        input_facts.iter().all(|fact| DeviceTensor::is_supported_dt(fact.datum_type));

    Ok(in_dts_metal_compatible
        && (node
            .op_as::<ElementWiseOp>()
            .is_some_and(|op| map_element_wise_ops_to_metal(op).is_some())
            || node.op_as::<TypedBinOp>().is_some_and(|op| map_bin_ops_to_metal(&op.0).is_some())
            || node.op_is::<Comp>()
            || node.op_is::<MultiBroadcastTo>()
            || node.op_as::<PrefixMatMul>().is_some_and(|op| {
                !op.transpose_c && op.quantize_output.is_none() && check_matmul_in_dts(&input_facts)
            })
            || node
                .op_as::<Const>()
                .is_some_and(|op| DeviceTensor::is_supported_dt(op.val().datum_type()))
            || node.op_as::<Cast>().is_some_and(|op| {
                ops::MetalCast::is_supported_dt(input_dts[0])
                    && ops::MetalCast::new(op.to).is_some()
            })
            || node.op_is::<AxisOp>()
            || node.op_is::<Slice>()
            || node.op_is::<TypedConcat>()
            || node.op_as::<Reduce>().is_some_and(|op| {
                kernels::nn::Reducer::is_supported_dt(input_dts[0])
                    && ops::MetalReduce::from_tract_core(op).is_ok()
            })
            || node.op_as::<CoreSoftmax>().is_some_and(|op| {
                kernels::nn::Softmax::is_supported_dt(input_dts[0])
                    && ops::MetalSoftmax::from_tract_core(op).is_ok()
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
                .op_as::<Silu>()
                .is_some_and(|_| kernels::nn::Silu::is_supported_dt(input_dts[0]))
            || node
                .op_as::<GeluApproximate>()
                .is_some_and(|_| kernels::nn::GeluApproximate::is_supported_dt(input_dts[0]))))
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for MetalTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let translatable = can_translate_to_metal_op(source, node)?;

        if translatable {
            let mut device_inputs =
                self.sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;

            let outlet_ids: TVec<OutletId> = if let Some(op) = node.op_as::<PrefixMatMul>() {
                convert_matmul_to_metal(
                    source,
                    node,
                    target,
                    &mut device_inputs,
                    op,
                    self.gemm_impl,
                )?
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
                } else if let Some(op) = node.op_as::<ScaledMaskedSoftmax>() {
                    Box::new(ops::MetalScaledMaskedSoftmax { scale: op.scale.clone() })
                } else if let Some(op) = node.op_as::<RmsNorm>() {
                    Box::new(ops::MetalRmsNorm::new(op.axis, op.eps.clone()))
                } else if let Some(_op) = node.op_as::<RotateHalf>() {
                    Box::new(ops::MetalRotateHalf)
                } else if let Some(_op) = node.op_as::<ApplyRope>() {
                    Box::new(ops::MetalApplyRope)
                } else if let Some(_op) = node.op_as::<Silu>() {
                    Box::new(ops::MetalSilu)
                } else if let Some(op) = node.op_as::<GeluApproximate>() {
                    Box::new(ops::MetalGeluApproximate { fast_impl: op.fast_impl })
                } else {
                    bail!("Failed to translate a supported Metal Op")
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

fn check_matmul_in_dts(in_facts: &[TypedFact]) -> bool {
    MlxGemm.is_supported_dts(in_facts)
        || MfaGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(&[in_facts[1].clone(), in_facts[0].clone()])
}

fn is_input_broadcast(facts: TVec<&TypedFact>) -> bool {
    // Assume weights are in second postion
    let b_batch_dims: Vec<TDim> = if as_q40_fact(facts[1]).is_some() {
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
    } else if as_q40_fact(input_facts[0]).is_some()
        || as_q40_fact(input_facts[1]).is_some()
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

    let mut matmul_output = match resolve_gemm_impl(gemm_impl, input_facts.clone())? {
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
                ensure!(as_q40_fact(input_facts[a_pos]).is_none(), "Cannot transpose Q40 tensor");

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
                ensure!(as_q40_fact(input_facts[b_pos]).is_none(), "Cannot transpose Q40 tensor");

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
    let out_dt = out_fact.to_device_fact().map(|f| f.datum_type).unwrap_or(out_fact.datum_type);

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
    let typed_fact: TypedFact = Arc::clone(op.val()).into();
    let metal_fact = if let Some(of) = op.opaque_fact() {
        DeviceFact::from_host(typed_fact.with_opaque_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };

    let metal_const = op.val().clone().into_device()?.into_opaque_tensor().into_arc_tensor();
    Const::new_with_opaque_fact(metal_const, Box::new(metal_fact))
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
