pub mod apply_rope;
pub mod binary;
pub mod broadcast;
pub mod cast;
pub mod change_axes;
pub mod concat;
pub mod element_wise;
pub mod fused_axis_op;
pub mod gemm;
pub mod new_gelu;
pub mod reduce;
pub mod rms_norm;
pub mod rotate_half;
pub mod scaled_masked_softmax;
pub mod silu;
pub mod slice;
pub mod softmax;
pub mod sync;

pub use apply_rope::MetalApplyRope;
pub use binary::MetalBinOp;
pub use broadcast::MetalMultiBroadcastTo;
pub use cast::MetalCast;
pub use change_axes::MetalAxisOp;
pub use concat::MetalConcat;
pub use element_wise::MetalElementWiseOp;
pub use fused_axis_op::MetalFusedAxisOp;
pub use gemm::MetalGemm;
pub use new_gelu::MetalNewGelu;
pub use reduce::MetalReduce;
pub use rms_norm::MetalRmsNorm;
pub use rotate_half::MetalRotateHalf;
pub use scaled_masked_softmax::MetalScaledMaskedSoftmax;
pub use silu::MetalSilu;
pub use slice::MetalSlice;
pub use softmax::MetalSoftmax;
pub use sync::{MetalSync, MetalSyncKind};

use crate::{MetalContext, MetalTensor};
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;

pub trait MetalEvalOp: EvalOp + Op + Clone {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>>;
}

#[derive(Debug, Clone, new)]
pub struct MetalOpState<O: MetalEvalOp> {
    node_id: usize,
    op: O,
}

impl<O: MetalEvalOp> OpStateFreeze for MetalOpState<O> {
    fn freeze(&self) -> Box<(dyn FrozenOpState + 'static)> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> FrozenOpState for MetalOpState<O> {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> OpState for MetalOpState<O> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                if let Some(profiler) = context.profiler() {
                    profiler.borrow_mut().add_node_entry(self.node_id);
                };
                self.op.metal_eval(context, self.node_id, session, inputs)
            })
        })
    }
}

pub fn make_tensor_for_node(
    session: &SessionState,
    node_id: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<MetalTensor> {
    crate::session_handler::get_metal_mem_pool(session)
        .map(|mem| mem.tensor_for_node(node_id, dt, shape))
        .unwrap_or_else(|| unsafe { MetalTensor::uninitialized_dt(dt, shape) })
}
