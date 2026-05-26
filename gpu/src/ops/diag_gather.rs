use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

/// `out[..., i, k] = in[..., i, offset + k - i]`, with zero-fill on
/// out-of-bounds reads. Mirrors `tract_transformers::ops::diag_gather::DiagGather`.
///
/// `offset` and `out_len` are TDim because the rel-pos table width and the
/// query-axis length may both be symbolic upstream; both are resolved against
/// `session.resolved_symbols` at eval time.
pub type DispatchDiagGatherFn =
    fn(input: &DeviceTensor, offset: i64, out_len: usize, output: &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuDiagGather {
    pub offset: TDim,
    pub out_len: TDim,
    pub backend_name: &'static str,
    pub dispatch: DispatchDiagGatherFn,
}

impl std::fmt::Debug for GpuDiagGather {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}DiagGather", self.backend_name)
    }
}

impl PartialEq for GpuDiagGather {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.offset == other.offset
            && self.out_len == other.out_len
    }
}
impl Eq for GpuDiagGather {}

impl std::hash::Hash for GpuDiagGather {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.offset.hash(state);
        self.out_len.hash(state);
    }
}

impl Op for GpuDiagGather {
    fn name(&self) -> StaticName {
        format!("{}DiagGather", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("offset={}, out_len={}", self.offset, self.out_len)])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuDiagGather {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_val = args_1!(inputs);
        let input = input_val.to_device_tensor()?;
        let offset = self.offset.eval(&session.resolved_symbols).to_i64()?;
        let out_len = self.out_len.eval(&session.resolved_symbols).to_usize()?;
        let mut out_shape: TVec<usize> = input.shape().into();
        let rank = out_shape.len();
        ensure!(rank >= 2);
        out_shape[rank - 1] = out_len;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &out_shape,
        )?;
        (self.dispatch)(input, offset, out_len, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuDiagGather {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 1);
            let mut shape: TVec<TDim> = facts[0].shape.to_tvec();
            ensure!(shape.len() >= 2);
            let rank = shape.len();
            shape[rank - 1] = self.out_len.clone();
            Ok(tvec!(facts[0].datum_type.fact(&shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}
