use crate::kernels::array::Concat;
use crate::ops::MetalEvalOp;
use crate::utils::with_borrowed_metal_stream;
use crate::MetalStream;
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCache;

#[derive(new, Debug, Clone, Hash)]
pub struct MetalDynKVCache {
    pub symbols: [TDim; 2],
    pub kernel: Concat,
}

impl MetalDynKVCache {
    pub fn from_tract_transformers(op: &DynKeyValueCache) -> Self {
        Self { kernel: Concat { axis: op.axis }, symbols: op.symbols.clone() }
    }

    pub fn axis(&self) -> usize {
        self.kernel.axis
    }
}

impl Op for MetalDynKVCache {
    fn name(&self) -> Cow<str> {
        "MetalDynKVCache".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis())])
    }

    op_as_typed_op!();
}

#[derive(Debug, Clone, new)]
pub struct MetalDynKVCacheState<O: MetalEvalOp> {
    node_id: usize,
    op: O,
    kv_cache: Option<DeviceTensor>
}

impl<O: MetalEvalOp> OpStateFreeze for MetalDynKVCacheState<O> {
    fn freeze(&self) -> Box<(dyn FrozenOpState + 'static)> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> FrozenOpState for MetalDynKVCacheState<O> {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> OpState for MetalDynKVCacheState<O> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1);
        if self.kv_cache.is_none() {
            self.kv_cache = Some(inputs[0].to_device_tensor().cloned()?);
            Ok(tvec!(inputs[0].clone()))
        } else {
            with_borrowed_metal_stream(|stream| {
                let inputs = tvec!(self.kv_cache.clone().unwrap().into_opaque_tensor().into_tvalue(), inputs[0].clone());
                let res = self.op.metal_eval(stream, self.node_id, session, inputs)?;
                self.kv_cache = Some(res[0].to_device_tensor()?.clone());
                Ok(res)
            })
        }
    }
}

impl EvalOp for MetalDynKVCache {
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut tract_core::internal::SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MetalDynKVCacheState::new(node_id, self.clone(), None))))
    }
}

impl MetalEvalOp for MetalDynKVCache {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        opaque_inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(opaque_inputs.len() == 2);
        let inputs = opaque_inputs
            .iter()
            .map(|it| it.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;

        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis()] = inputs.iter().map(|it| it.shape()[self.axis()]).sum();
        let output = crate::ops::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape,
        )?;
        self.kernel.dispatch_eval(stream, &inputs, &output)?;

        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalDynKVCache {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            fact.shape.set(self.axis(), self.symbols.iter().sum());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
}
