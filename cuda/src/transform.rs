use tract_core::dyn_clone::clone_box;
use tract_core::internal::*;
use tract_core::model::translator::Translate;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::einsum::prefix_matmul::rewrite_einsum_to_prefix_matmul;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::Comp;
use tract_core::tract_data::itertools::Itertools;
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::sync::{DeviceSync, DeviceSyncKind};
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_transformers::ops::silu::Silu;

use crate::context::cuda_context;
use crate::{kernels, ops};

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

        rewrite_einsum_to_prefix_matmul(model)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        if stop_at_phase == 1 {
            return Ok(());
        }

        *model = self.translate_model(model)?;

        if stop_at_phase == 2 {
            return Ok(());
        }

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
            || node.op_as::<Silu>().is_some_and(|_| kernels::UnaryOps::is_supported_dt(input_dts[0]))
            || node.op_as::<ElementWiseOp>().is_some_and(|op| kernels::UnaryOps::is_supported_dt(input_dts[0]) && map_element_wise_ops_to_cuda(op).is_some())
            || node.op_as::<TypedBinOp>().is_some_and(|op| kernels::BinOps::is_supported_dt(input_dts[0]) && map_binary_op_to_cuda(op).is_some())
            || node.op_as::<Comp>().is_some_and(|_| kernels::BinOps::is_supported_dt(input_dts[0]))
    )
    )
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

macro_rules! map_unary_ops {
    ([$(($tract_unary_op:path, $metal_unary_op:ident)),* $(,)?]) => {
        |op: &tract_core::ops::element_wise::ElementWiseOp| {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_unary_op>() {
                return Some($crate::ops::unary::CudaUnaryOp(kernels::UnaryOps::$metal_unary_op));
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
    ([$(($tract_bin_op:path, $metal_bin_op:ident)),* $(,)?]) => {
        |op: &TypedBinOp | {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::binary::CudaBinOp(kernels::BinOps::$metal_bin_op));
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
            let device_inputs =
                self.sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;

            let op: Box<dyn TypedOp> =  if let Some(op) = node.op_as::<Const>() {
                    Box::new(convert_const(op)?)
                }
            else if let Some(op) = node.op_as::<ElementWiseOp>() {
                Box::new(map_element_wise_ops_to_cuda(op).unwrap())
            }
            else if let Some(op) = node.op_as::<TypedBinOp>() {
                Box::new(map_binary_op_to_cuda(op).unwrap())
            }
            else if let Some(op) = node.op_as::<Comp>() {
                Box::new(convert_logic_op_to_cuda(op))
            }
            else if let Some(_op) = node.op_as::<Silu>() {
                Box::new(ops::CudaUnaryOp(kernels::UnaryOps::Silu))
            } else {
                bail!("Failed to translate a supported CUDA Op")
            };
            let outlet_ids = target.wire_node(node.name.clone(), op, &device_inputs)?;
            self.sync_model_outputs_if_required(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                self.sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToHost)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}
