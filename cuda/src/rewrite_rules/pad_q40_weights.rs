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

/// Walk the successor chain from a Const node to find a GEMM consumer.
/// Returns the effective weight shape as seen by the GEMM kernel (after all
/// axis ops, including CudaFusedAxisOp internal reshapes).  The last
/// dimension of the returned shape is the k dimension.
fn effective_gemm_shape(
    model: &TypedModel,
    node: &TypedNode,
    shape: &[usize],
) -> TractResult<Option<TVec<usize>>> {
    let mut cursor = node;
    let mut effective_shape: TVec<usize> = shape.into();
    while let Some(succ) = model.single_succ(cursor.id)? {
        if succ.op_is::<CudaGgmlGemm>() {
            return Ok(Some(effective_shape));
        }
        if let Some(fao) = succ.op_as::<CudaFusedAxisOp>() {
            if fao.op.is::<CudaGgmlGemm>() {
                // Apply the fused axis ops for the weight input slot.
                let weight_inlet = succ.inputs.iter().position(|i| i.node == cursor.id).unwrap();
                for axis_op in &fao.grouped_axis_ops[weight_inlet] {
                    axis_op.0.change_shape_array(&mut effective_shape, false)?;
                }
                return Ok(Some(effective_shape));
            }
        }
        if let Some(axis_op) = succ.op_as::<CudaAxisOp>() {
            axis_op.0.change_shape_array(&mut effective_shape, false)?;
            cursor = succ;
            continue;
        }
        break;
    }
    Ok(None)
}

// This rule is necessary GGML Q40 Matmul kernels that requires k % 512 == 0
pub fn pad_q40_weights(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &Const,
) -> TractResult<Option<TypedModelPatch>> {
    let Some(dev_tensor) = op.val().to_device_tensor().ok() else {
        return Ok(None);
    };

    let DeviceTensor::Owned(t) = dev_tensor else {
        return Ok(None);
    };
    let Some(cuda_tensor) = t.downcast_ref::<CudaTensor>() else {
        return Ok(None);
    };

    let bqf = cuda_tensor
        .opaque_fact()
        .and_then(|of| of.downcast_ref::<BlockQuantFact>())
        .filter(|bqf| bqf.format.same_as(&Q4_0));
    rule_ensure!(bqf.is_some());
    let bqf = bqf.unwrap();

    // Compute the effective weight shape as seen by the GEMM kernel (after
    // all axis ops and CudaFusedAxisOp internal reshapes).  The raw tensor
    // may have a per-head shape like [m, num_heads, head_dim] that gets
    // collapsed before the GEMM.
    let Some(effective_shape) = effective_gemm_shape(model, node, bqf.shape())? else {
        return Ok(None);
    };
    let effective_k = *effective_shape.last().unwrap();
    rule_ensure!(effective_k % Q40_ROW_PADDING != 0);

    let host_tensor = dev_tensor.to_host()?.into_tensor();
    let bqs = as_q40_tensor(&host_tensor).expect("expected Q4_0 tensor view");
    // Compute flat (m, k) matching the contiguous Q4_0 data layout.
    // The data can be viewed as flat_m rows of effective_k elements each.
    let total_elements: usize = bqf.shape().iter().product();
    let flat_m = total_elements / effective_k;
    let padded_bqs = pad_q40(bqs, flat_m, effective_k)?;

    // Build padded shape preserving the rank and batch structure expected by
    // the GEMM.  effective_shape already incorporates all CudaAxisOp and
    // CudaFusedAxisOp transformations, so we just replace k with padded_k.
    let padded_k = effective_k.next_multiple_of(Q40_ROW_PADDING);
    let mut padded_shape = effective_shape.clone();
    *padded_shape.last_mut().unwrap() = padded_k;
    let padded_bqf = BlockQuantFact::new(
        tract_core::dyn_clone::clone_box(padded_bqs.format()),
        padded_shape.clone(),
    );
    let padded_fact =
        TypedFact::dt_shape(DatumType::Opaque, &padded_shape).with_opaque_fact(padded_bqf);

    let padded_tensor =
        padded_bqs.into_tensor_with_shape(f32::datum_type(), &padded_shape).into_arc_tensor();

    let new_const = Const::new_with_opaque_fact(
        padded_tensor.into_device()?.into_tensor().into_arc_tensor(),
        Box::new(DeviceFact::from_host(padded_fact)?),
    )?;

    let mut patch = TypedModelPatch::default();
    let wire = patch.wire_node(&node.name, new_const, &[])?[0];

    // The padded const already has the effective GEMM shape (with padded k),
    // so skip the CudaAxisOp chain — those transforms are baked in.
    // Collect the intermediate node ids to obliterate.
    let mut cursor = node;
    let mut obliterate_ids: TVec<usize> = tvec![node.id];
    while let Some(succ) = model.single_succ(cursor.id)? {
        if succ.op_is::<CudaAxisOp>() {
            obliterate_ids.push(succ.id);
            cursor = succ;
            continue;
        }
        break;
    }

    // Rewire the downstream GEMM node.  If the consumer is a CudaFusedAxisOp,
    // clear the weight input's axis ops since the padded tensor is already in
    // the expected shape.
    let gemm_succ = model.single_succ(cursor.id)?.unwrap();
    let weight_outlet: OutletId = cursor.id.into();
    let weight_inlet = gemm_succ
        .inputs
        .iter()
        .position(|i| i.node == weight_outlet.node && i.slot == weight_outlet.slot)
        .unwrap();
    let mut gemm_inputs: TVec<OutletId> = tvec![];
    for (ix, input) in gemm_succ.inputs.iter().enumerate() {
        if ix == weight_inlet {
            gemm_inputs.push(wire);
        } else {
            gemm_inputs.push(patch.tap_model(model, *input)?);
        }
    }
    let gemm_op: Box<dyn TypedOp> = if let Some(fao) = gemm_succ.op_as::<CudaFusedAxisOp>() {
        let mut axis_ops = fao.grouped_axis_ops.clone();
        axis_ops[weight_inlet].clear();
        Box::new(CudaFusedAxisOp::new(axis_ops, fao.op.clone()))
    } else {
        gemm_succ.op.clone()
    };
    let gemm_out = patch.wire_node(&gemm_succ.name, gemm_op, &gemm_inputs)?;
    patch.shunt_outside(model, gemm_succ.id.into(), gemm_out[0])?;
    obliterate_ids.push(gemm_succ.id);
    for id in obliterate_ids {
        patch.obliterate(id)?;
    }
    Ok(Some(patch))
}
