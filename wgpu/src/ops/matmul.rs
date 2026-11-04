use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::matmul::{Transposes, output_shape, wgpu_matmul_dispatch};
use crate::kernels::shaders::ChainStep;
use crate::transform::Mapping;

/// `PrefixMatMul` on the GPU, with any elementwise ops that followed it applied
/// to each result before it is stored. Inputs past the two operands belong to
/// the epilogue, one per binary step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuGemm {
    pub op: PrefixMatMul,
    pub epilogue: Vec<ChainStep>,
}

impl Op for WgpuGemm {
    fn name(&self) -> StaticName {
        "WgpuGemm".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = vec![format!(
            "transpose_a: {} transpose_b: {} transpose_c: {}",
            self.op.transpose_a, self.op.transpose_b, self.op.transpose_c
        )];
        if !self.epilogue.is_empty() {
            info.push(format!("epilogue: {:?}", self.epilogue));
        }
        Ok(info)
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuGemm {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let (a, b) = (inputs[0], inputs[1]);
        let shape = output_shape(
            a.shape(),
            b.shape(),
            self.op.transpose_a,
            self.op.transpose_b,
            self.op.transpose_c,
        )?;
        let output = tract_gpu::turn_handler::make_tensor_for_node(ctx, a.datum_type(), &shape)?;
        if output.len() > 0 {
            wgpu_matmul_dispatch(
                Transposes {
                    a: self.op.transpose_a,
                    b: self.op.transpose_b,
                    c: self.op.transpose_c,
                },
                &self.epilogue,
                a,
                b,
                &inputs[2..],
                &output,
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuGemm {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.op.output_facts(&facts[0..2]))
            .with_context(|| "Error while computing facts for WgpuGemm")
    }
}

/// A `Q4_0` or `Q2_0_T` weight on B, K last, block 32, rank 2 or 3.
/// Anything else stays on the CPU.
fn packed_weight_ok(op: &PrefixMatMul, facts: &[&TypedFact]) -> bool {
    if facts.len() < 2 || op.quantize_output.is_some() || !op.transpose_b {
        return false;
    }
    let (a, b) = (facts[0], facts[1]);
    if a.exotic_fact().is_some()
        || !crate::kernels::matmul::is_supported_dt(a.datum_type)
        || op.operating_dt.is_some_and(|dt| dt != a.datum_type)
        || a.rank() < 2
        || a.rank() > 3
        || a.rank() != b.rank()
    {
        return false;
    }
    let Some(bqf) = b.exotic_fact().and_then(|f| f.downcast_ref::<BlockQuantFact>()) else {
        return false;
    };
    let k = b.shape.dims().last().and_then(|d| d.to_usize().ok());
    k == Some(bqf.k())
        && bqf.k() % 32 == 0
        && bqf.format.block_len() == 32
        && (bqf.format.dyn_eq(&Q4_0) || bqf.format.dyn_eq(&Q2_0_T))
}

/// The same product with its operands exchanged: `(A B)^T = B^T A^T`.
fn swapped(op: &PrefixMatMul) -> PrefixMatMul {
    PrefixMatMul {
        transpose_a: !op.transpose_b,
        transpose_b: !op.transpose_a,
        transpose_c: !op.transpose_c,
        ..*op
    }
}

/// Translates a `PrefixMatMul` into a [`WgpuGemm`] with no epilogue yet.
///
/// Float operands must share a dtype. A `Q4_0` or `Q2_0_T` weight is accepted
/// on B when `transpose_b` is set, the rank is 2 or 3, and K is a multiple of
/// the block; on A, it is moved to B when that makes such a product. A packed
/// weight in any other position stays on the CPU.
pub fn translate_gemm(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    let mut input_facts = source.node_input_facts(node.id)?;
    rule_if_some!(op = node.op_as::<PrefixMatMul>());
    let mut op = *op;
    let on_a = input_facts.len() == 2 && input_facts[0].exotic_fact().is_some();
    if on_a {
        input_facts.swap(0, 1);
        op = swapped(&op);
    }
    if !packed_weight_ok(&op, &input_facts) {
        rule_if!(
            op.quantize_output.is_none()
                && op.operating_dt.is_none_or(|dt| dt == input_facts[0].datum_type)
                && input_facts.iter().all(|f| {
                    crate::kernels::matmul::is_supported_dt(f.datum_type)
                        && f.exotic_fact().is_none()
                })
                && input_facts[0].datum_type == input_facts[1].datum_type
                && input_facts[0].rank() <= crate::kernels::matmul::MAX_RANK
        );
    }
    let mut device_inputs =
        sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    if on_a {
        device_inputs.swap(0, 1);
    }
    let outlet_ids =
        target.wire_node(node.name.clone(), WgpuGemm { op, epilogue: vec![] }, &device_inputs)?;
    Ok(Some(outlet_ids))
}
