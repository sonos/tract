use crate::context::CUDA_STREAM;
use crate::kernels::array::Pad;
use tract_core::internal::*;
use tract_core::ops::array as core_array;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Hash)]
pub struct CudaPad {
    pads: Vec<(TDim, TDim)>,
    mode: core_array::PadMode,
}

impl CudaPad {
    pub fn new(pads: Vec<(TDim, TDim)>, mode: core_array::PadMode) -> TractResult<Self> {
        ensure!(
            matches!(mode, core_array::PadMode::Constant(_)),
            "Only Constant padding supported for now"
        );

        Ok(CudaPad { pads, mode })
    }

    #[allow(unused)]
    pub fn from_tract_core(core_pad: &core_array::Pad) -> TractResult<Self> {
        ensure!(
            matches!(core_pad.mode, core_array::PadMode::Constant(_)),
            "Only Constant padding supported for now"
        );

        let pads = core_pad
            .pads
            .iter()
            .map(|(bef, aft)| (TDim::Val(*bef as i64), TDim::Val(*aft as i64)))
            .collect_vec();
        Ok(CudaPad { pads, mode: core_pad.mode.clone() })
    }
}

impl Op for CudaPad {
    fn name(&self) -> StaticName {
        "CudaPad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("padding: {:?} with mode: {:?}", self.pads, self.mode)])
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<CudaPad>() else { return false };
        self.pads == other.pads && self.mode == other.mode
    }

    op_as_typed_op!();
}

impl EvalOp for CudaPad {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque_input = args_1!(inputs);
        let input = opaque_input.to_device_tensor()?;
        let padding = &self.pads;
        ensure!(input.rank() == padding.len());

        let mut paddings_before: TVec<usize> = tvec![];
        let mut output_shape = input.shape().to_vec();
        for i in 0..input.rank() {
            let left = padding[i].0.eval_to_i64(&session.resolved_symbols)? as usize;
            let right = padding[i].1.eval_to_i64(&session.resolved_symbols)? as usize;
            paddings_before.push(left);
            output_shape[i] += left + right;
        }

        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &output_shape,
        )?;

        CUDA_STREAM.with(|stream| {
            Pad.dispatch_eval(stream, input, &output, paddings_before, self.mode.clone())
        })?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaPad {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let input = facts[0];
            let padding = self.pads.clone();
            ensure!(input.rank() == padding.len());

            let output_shape = Pad::output_shape(&input.shape, &padding)?;

            let dt = input.datum_type;
            Ok(tvec!(dt.fact(output_shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
}
