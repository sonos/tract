use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::*;
use tract_gpu::fact::DeviceFact;
use tract_gpu::rule_ensure;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;

use crate::Q40_ROW_PADDING;
use crate::ops::{CudaAxisOp, CudaFusedAxisOp, CudaGgmlGemm};
use crate::tensor::CudaTensor;
use crate::utils::pad_q40;

fn is_mm_weights(model: &TypedModel, node: &TypedNode) -> TractResult<bool> {
    let mut cursor = node;
    while let Some(succ) = model.single_succ(cursor.id)? {
        if succ.op_is::<CudaGgmlGemm>()
            || (succ.op_as::<CudaFusedAxisOp>().is_some_and(|fao| fao.op.is::<CudaGgmlGemm>()))
        {
            return Ok(true);
        }

        if succ.op_is::<CudaAxisOp>() {
            cursor = succ;
            continue;
        }
        break;
    }

    Ok(false)
}

// This rule is necessary GGML Q40 Matmul kernels that requires k % 512 == 0
pub fn pad_q40_weights(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &Const,
) -> TractResult<Option<TypedModelPatch>> {
    if !is_mm_weights(model, node)? {
        return Ok(None);
    }

    let Some(dev_tensor) = op.val().to_device_tensor().ok() else {
        return Ok(None);
    };

    let DeviceTensor::Owned(t) = dev_tensor else {
        return Ok(None);
    };
    let Some(cuda_tensor) = t.downcast_ref::<CudaTensor>() else {
        return Ok(None);
    };

    rule_ensure!(cuda_tensor.opaque_fact().is_some_and(|of| {
        of.downcast_ref::<BlockQuantFact>()
            .is_some_and(|bqf| bqf.format.same_as(&Q4_0) && bqf.k() % Q40_ROW_PADDING != 0)
    }));

    let host_tensor = dev_tensor.to_host()?.into_tensor();
    let bqs = as_q40_tensor(&host_tensor).expect("expected Q4_0 tensor view");
    let m: usize = host_tensor.shape()[..host_tensor.rank() - 1].iter().product();
    let k = *host_tensor.shape().last().unwrap();
    let padded_bqs = pad_q40(bqs, m, k)?;

    let typed_fact: TypedFact = Arc::clone(op.val()).into();
    // Preserve the original tensor's group structure in the padded shape
    let mut padded_shape: TVec<usize> = host_tensor.shape().into();
    let rank = padded_shape.len();
    padded_shape[rank - 1] = k.next_multiple_of(Q40_ROW_PADDING);
    let padded_bqf = BlockQuantFact::new(
        tract_core::dyn_clone::clone_box(padded_bqs.format()),
        padded_shape.clone(),
    );
    let padded_fact = typed_fact.with_opaque_fact(padded_bqf);

    let padded_tensor = padded_bqs.into_tensor_with_shape(&padded_shape).into_arc_tensor();

    let new_const = Const::new_with_opaque_fact(
        padded_tensor.into_device()?.into_tensor().into_arc_tensor(),
        Box::new(DeviceFact::from_host(padded_fact)?),
    )?;

    let mut patch = TypedModelPatch::default();
    let mut wire = patch.wire_node(&node.name, new_const, &[])?[0];

    // Rebuild the CudaAxisOp chain so that intermediate facts reflect the
    // padded k dimension.  Without this the memory pool would allocate buffers
    // using the stale (unpadded) fact shapes, causing a size mismatch at
    // runtime.
    let mut cursor = node;
    let mut obliterate_ids: TVec<usize> = tvec![node.id];
    while let Some(succ) = model.single_succ(cursor.id)? {
        if succ.op_is::<CudaAxisOp>() {
            wire = patch.wire_node(&succ.name, succ.op.clone(), &[wire])?[0];
            obliterate_ids.push(succ.id);
            cursor = succ;
            continue;
        }
        break;
    }

    // Rewire the downstream gemm node so that all facts are consistent with
    // the padded k dimension.
    let gemm_succ = model.single_succ(cursor.id)?.unwrap();
    let weight_outlet: OutletId = cursor.id.into();
    let mut gemm_inputs: TVec<OutletId> = tvec![];
    for input in &gemm_succ.inputs {
        if input.node == weight_outlet.node && input.slot == weight_outlet.slot {
            gemm_inputs.push(wire);
        } else {
            gemm_inputs.push(patch.tap_model(model, *input)?);
        }
    }
    let gemm_out = patch.wire_node(&gemm_succ.name, gemm_succ.op.clone(), &gemm_inputs)?;
    patch.shunt_outside(model, gemm_succ.id.into(), gemm_out[0])?;
    obliterate_ids.push(gemm_succ.id);
    for id in obliterate_ids {
        patch.obliterate(id)?;
    }
    Ok(Some(patch))
}
