use tract_core::ops::logic::Comp;
use crate::kernels::nn::{ Reducer, Softmax, RmsNorm, Silu, NewGelu };
use crate::fact::MetalTypedFactExt;
use crate::ops;
use crate::ops::{ MetalSync, MetalSyncKind };
use crate::tensor::MetalTensorExt;
use crate::{ IntoMetal, MetalFact, MetalTensor };
use anyhow::Result;
use std::borrow::Cow;
use std::fmt::Debug;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::nn::{ Reduce, Softmax as CoreSoftmax };
use tract_core::ops::array::{ MultiBroadcastTo, Slice, TypedConcat };
use tract_core::ops::binary::{ TypedBinOp, BinMiniOp };
use tract_core::ops::einsum::{rewrite_einsums_as_matmul, BasicMatMul};
use crate::rewrite_rules::{ as_rms_norm_rule, rewire_metal_sync, BasicRmsNorm, as_silu_rule, BasicSilu, as_new_gelu_rule, BasicNewGelu };
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::transform::ModelTransform;

#[derive(Debug, Default)]
pub struct MetalTransform;

impl ModelTransform for MetalTransform {
    fn name(&self) -> Cow<str> {
        "metal-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        rewrite_einsums_as_matmul(model)?;
        Rewriter::default()
            .with_rule_for::<Reduce>("as-rms-norm", as_rms_norm_rule)
            .with_rule_for::<ElementWiseOp>("as-silu", as_silu_rule)
            .with_rule_for::<TypedBinOp>("as-new-gelu", as_new_gelu_rule)
            .rewrite(&(), model)?;

        let mut new = self.translate_model(model)?;

        Rewriter::default()
            .with_rule_for::<MetalSync>("rewire-metal-sync", rewire_metal_sync)
            .rewrite(&(), &mut new)?;
        *model = new;
        Ok(())
    }
}


impl MetalTransform {
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
                            let metal_fact = MetalFact::new(in_fact.clone())?;

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

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for MetalTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let in_dts_metal_compatible = source
            .node_input_facts(node.id)?
            .iter()
            .all(|f| MetalTensor::is_supported_dt(f.datum_type) || f.as_metal_fact().is_some());

        let new_metal_op: Option<Box<dyn TypedOp>> = if !in_dts_metal_compatible {
            None
        } else if let Some(op) = node.op_as::<ElementWiseOp>() {
            convert_element_wise_ops_to_metal(op).map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<TypedBinOp>() {
            convert_bin_ops_to_metal(&op.0).map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<Comp>() {
            Some(Box::new(convert_logic_ops_to_metal(op)))
        } else if let Some(op) = node.op_as::<BasicMatMul>() {
            convert_matmul_to_metal(source, node, op)?.map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<MultiBroadcastTo>() {
            Some(Box::new(ops::MetalMultiBroadcastTo::new(op.shape.clone())))
        } else if let Some(op) = node.op_as::<Const>() {
            ops::MetalConst::new(op.0.clone())?.map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<AxisOp>() {
            ops::MetalAxisOp::from_tract_core(op.clone()).map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<Slice>() {
            Some(Box::new(ops::MetalSlice::from_tract_core(op.clone())))
        } else if let Some(op) = node.op_as::<TypedConcat>() {
            Some(Box::new(ops::MetalConcat::from_tract_core(op)))
        } else if let Some(op) = node.op_as::<Reduce>() {
            check_in_dts_are_supported(source, node.id,  Reducer::is_supported_dt)?
                .then(|| ops::MetalReduce::from_tract_core(op).ok())
                .flatten()
                .map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<CoreSoftmax>() {
            check_in_dts_are_supported(source, node.id,  Softmax::is_supported_dt)?
                .then(|| ops::MetalSoftmax::from_tract_core(op).ok())
                .flatten()
                .map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(op) = node.op_as::<BasicRmsNorm>() {
            check_in_dts_are_supported(source, node.id,  RmsNorm::is_supported_dt)?
                .then(|| ops::MetalRmsNorm::new(op.axis, op.eps.clone()))
                .map(|o| -> Box<dyn TypedOp> { Box::new(o) })
        } else if let Some(_op) = node.op_as::<BasicSilu>() {
            check_in_dts_are_supported(source, node.id,  Silu::is_supported_dt)?
                .then(|| -> Box<dyn TypedOp> { Box::new(ops::MetalSilu) })
        } else if let Some(_op) = node.op_as::<BasicNewGelu>() {
            check_in_dts_are_supported(source, node.id,  NewGelu::is_supported_dt)?
                .then(|| -> Box<dyn TypedOp> { Box::new(ops::MetalNewGelu) })
        } else {
            None
        };

        match new_metal_op {
            Some(metal_op) => {
                let gpu_inputs =
                    self.sync_inputs_if_required(target, node, mapping, MetalSyncKind::ToGpu)?;
                let target_node_outlet_ids = target.wire_node(&node.name, metal_op, &gpu_inputs)?;
                self.sync_model_outputs_if_required(source, node, target, target_node_outlet_ids)
            }
            None => {
                let cpu_inputs =
                    self.sync_inputs_if_required(target, node, mapping, MetalSyncKind::ToCpu)?;
                target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
            }
        }
    }
}

fn check_in_dts_are_supported(model: &TypedModel, node_id: usize, is_supported_dt: impl Fn(DatumType) -> bool) -> TractResult<bool> {
    Ok(model
            .node_input_facts(node_id)?
            .iter()
            .all(|f| {
                (is_supported_dt)(f.datum_type) 
                    || f.as_metal_fact().map(|f| (is_supported_dt)(f.datum_type)).unwrap_or(false)
            }))
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

fn convert_matmul_to_metal(
    model: &TypedModel,
    node: &TypedNode,
    op: &BasicMatMul,
) -> Result<Option<ops::MetalGemm>> {
    if !op.transpose_c
        && op.quantize_output.is_none()
        && (model.node_input_facts(node.id)?.iter().all(|f| f.datum_type == f32::datum_type())
            || model.node_input_facts(node.id)?.iter().all(|f| f.datum_type == f16::datum_type()))
    {
        Ok(Some(ops::MetalGemm::new(op.transpose_a, op.transpose_b)))
    } else {
        Ok(None)
    }
}

pub fn matmul_to_gemm(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &BasicMatMul,
) -> Result<Option<TypedModelPatch>> {
    convert_matmul_to_metal(model, node, op)?
        .map(|metal_op| TypedModelPatch::replace_single_op(model, node, &node.inputs, metal_op))
        .transpose()
}

#[allow(clippy::borrowed_box)]
fn convert_bin_ops_to_metal(op: &Box<dyn BinMiniOp>) -> Option<ops::MetalBinOp> {
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

pub fn bin_ops_to_metal(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &TypedBinOp,
) -> Result<Option<TypedModelPatch>> {
    if op.1.is_some() {
        return Ok(None);
    }

    let input_facts = model.node_input_facts(node.id)?;
    let dt = input_facts[0].datum_type;

    // All input must have the same datum type and it has to be supported.
    if model.node_input_facts(node.id)?.iter().any(|f| f.datum_type != dt)
        || !crate::kernels::BinOps::is_supported_dt(dt)
    {
        return Ok(None);
    }

    convert_bin_ops_to_metal(&op.0)
        .map(|metal_op| TypedModelPatch::replace_single_op(model, node, &node.inputs, metal_op))
        .transpose()
}

fn convert_element_wise_ops_to_metal(op: &ElementWiseOp) -> Option<ops::MetalElementWiseOp> {
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

pub fn element_wise_ops_to_metal(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &ElementWiseOp,
) -> Result<Option<TypedModelPatch>> {
    if op.1.is_some() {
        return Ok(None);
    }

    let input_facts = model.node_input_facts(node.id)?;
    let dt = input_facts[0].datum_type;

    // All input must have the same datum type and it has to be supported.
    if model.node_input_facts(node.id)?.iter().any(|f| f.datum_type != dt)
        || !crate::kernels::ElementWiseOps::is_supported_dt(dt)
    {
        return Ok(None);
    }

    convert_element_wise_ops_to_metal(op)
        .map(|metal_op| TypedModelPatch::replace_single_op(model, node, &node.inputs, metal_op))
        .transpose()
}
