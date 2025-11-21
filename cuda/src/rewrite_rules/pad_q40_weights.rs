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
    let q40_view = as_q40_tensor(&host_tensor).expect("expected Q4_0 tensor view");
    let padded_bqv = pad_q40(q40_view)?;

    let typed_fact: TypedFact = Arc::clone(op.val()).into();
    let padded_fact = typed_fact.with_opaque_fact(padded_bqv.fact.clone());

    let padded_tensor = tensor0(Opaque(Arc::new(padded_bqv)))
        .broadcast_into_rank(cuda_tensor.shape().len())?
        .into_arc_tensor();

    let new_const = Const::new_with_opaque_fact(
        padded_tensor.into_device()?.into_opaque_tensor().into_arc_tensor(),
        Box::new(DeviceFact::from_host(padded_fact)?),
    )?;

    Ok(Some(TypedModelPatch::replace_single_op(model, node, &[], new_const)?))
}
