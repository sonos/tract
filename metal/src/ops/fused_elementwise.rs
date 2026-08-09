use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};
use tract_gpu::utils::facts_to_device_facts;

use crate::kernels::element_wise::{FusedEwStepRaw, fused_ew_codes};

/// One step of a fused elementwise RPN program. `round_f16` reproduces the
/// original chain's numerics when the folded op computed in half precision:
/// the interpreter runs in f32 and rounds the step result through f16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedEwStep {
    PushInput(usize),
    /// f32 immediate, stored as bits so the op stays Eq + Hash.
    PushScalar(u32),
    Unary { code: u32, round_f16: bool },
    Binary { code: u32, round_f16: bool },
}

impl FusedEwStep {
    fn raw(&self) -> FusedEwStepRaw {
        match *self {
            FusedEwStep::PushInput(i) => FusedEwStepRaw {
                code: fused_ew_codes::PUSH_INPUT | (i as u32) << fused_ew_codes::SRC_SHIFT,
                imm: 0.0,
            },
            FusedEwStep::PushScalar(bits) => {
                FusedEwStepRaw { code: fused_ew_codes::PUSH_SCALAR, imm: f32::from_bits(bits) }
            }
            FusedEwStep::Unary { code, round_f16 } | FusedEwStep::Binary { code, round_f16 } => {
                FusedEwStepRaw {
                    code: code | if round_f16 { fused_ew_codes::FLAG_ROUND_F16 } else { 0 },
                    imm: 0.0,
                }
            }
        }
    }

    /// Stack effect of the step (+1 push, 0 unary, -1 binary).
    pub fn stack_effect(&self) -> isize {
        match self {
            FusedEwStep::PushInput(_) | FusedEwStep::PushScalar(_) => 1,
            FusedEwStep::Unary { .. } => 0,
            FusedEwStep::Binary { .. } => -1,
        }
    }
}

/// A chain of elementwise ops (unary float ops, float binary ops, float
/// casts) collapsed into a single kernel dispatch. Built by the
/// `fuse_elementwise_chain` Metal rewrite rules; evaluates a small RPN
/// program with one thread per output element.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalFusedElementwise {
    pub steps: TVec<FusedEwStep>,
    pub n_inputs: usize,
    pub out_dt: DatumType,
}

impl MetalFusedElementwise {
    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == self.n_inputs);
        let shapes: TVec<&[TDim]> = inputs.iter().map(|f| f.shape.as_ref()).collect();
        let out_shape = tract_core::broadcast::multi_broadcast(&shapes)?;
        Ok(tvec![self.out_dt.fact(out_shape)])
    }
}

impl Op for MetalFusedElementwise {
    fn name(&self) -> StaticName {
        "MetalFusedElementwise".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{} inputs, program: {:?}", self.n_inputs, self.steps)])
    }

    op_as_typed_op!();
}

impl EvalOp for MetalFusedElementwise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == self.n_inputs);
        let device_inputs: TVec<&DeviceTensor> = inputs
            .iter()
            .map(|i| i.to_device_tensor())
            .collect::<TractResult<_>>()
            .context("fused elementwise input is not a device tensor")?;
        let shapes: TVec<&[usize]> = device_inputs.iter().map(|t| t.shape()).collect();
        let out_shape = tract_core::broadcast::multi_broadcast(&shapes)?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            self.out_dt,
            &out_shape,
        )?;
        let raw_steps: TVec<FusedEwStepRaw> = self.steps.iter().map(|s| s.raw()).collect();
        crate::with_metal_stream(|stream| {
            crate::kernels::element_wise::dispatch_fused_elementwise_chain(
                stream,
                &raw_steps,
                &device_inputs,
                &output,
            )
        })?;
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalFusedElementwise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |facts| self.output_facts_inner(facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
