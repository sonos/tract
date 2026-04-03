use crate::fact::DeviceTypedFactExt;
use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_transformers::ops::dyn_kv_cache::{DynKeyValueCache, DynKeyValueCacheState};

#[derive(Debug, Clone, new)]
pub struct GpuDynKVCacheState {
    node_id: usize,
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<TValue>,
}

impl OpState for GpuDynKVCacheState {
    fn load_from(
        &mut self,
        state: &mut TurnState,
        states: &mut dyn Iterator<Item = TValue>,
    ) -> TractResult<()> {
        let kv_cache = states.next().context("Not enough state initializers")?;
        DynKeyValueCacheState::resolve_symbols(
            state,
            self.past_sequence_fact.clone(),
            Some(kv_cache.shape()),
        )?;
        self.kv_cache = Some(kv_cache.into_tensor().into_device()?.into_tensor().into_tvalue());
        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        if let Some(kv_cache) = &self.kv_cache {
            states.push(kv_cache.to_device_tensor()?.to_host()?.into_tensor().into_tvalue());
            Ok(())
        } else {
            bail!("KV cache {} was never initialized", self.name)
        }
    }

    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }

    fn resolve_symbols(&mut self, state: &mut TurnState) -> TractResult<()> {
        let shape = self
            .kv_cache
            .as_ref()
            .map(|kv_cache| kv_cache.to_device_tensor().expect("Expected GPU Tensor").shape());
        DynKeyValueCacheState::resolve_symbols(state, self.past_sequence_fact.clone(), shape)
    }

    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1);
        let mut op_inputs = TVec::new();

        if let Some(kv_cache) = self.kv_cache.take() {
            op_inputs.push(kv_cache);
        }

        op_inputs.push(inputs.into_iter().next().unwrap());

        let gpu_op =
            op.downcast_ref::<GpuDynKVCache>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let axis = gpu_op.axis;

        let inputs =
            op_inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[axis] = inputs.iter().map(|it| it.shape()[axis]).sum();
        let output = crate::session_handler::make_tensor_for_node(
            session,
            self.node_id,
            inputs[0].datum_type(),
            &output_shape,
        )?;

        // Concat inputs into output
        let ctx = crate::device::get_context()?;
        let mut cursor = 0usize;
        for input in &inputs {
            let slice_len = input.shape()[axis];
            if slice_len == 0 {
                continue;
            }
            let dst_offset =
                cursor * output.strides()[axis] as usize * output.datum_type().size_of();
            ctx.copy_nd(
                input,
                0,
                input.strides(),
                &output,
                dst_offset,
                input.shape(),
                output.strides(),
            )?;
            cursor += slice_len;
        }

        let res = output.into_tensor().into_tvalue();
        self.kv_cache = Some(res.clone());
        Ok(tvec!(res))
    }
}

impl GpuDynKVCacheState {
    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        if let Some(v) = &mut self.kv_cache {
            let mut t: Tensor = v.to_device_tensor()?.to_host()?.into_tensor();
            t = t.slice(self.axis, 0, len)?;
            *v = t.into_device()?.into_tensor().into_tvalue();
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FrozenGpuDynKVCacheState {
    node_id: usize,
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<DeviceTensor>,
}

impl OpStateFreeze for GpuDynKVCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState + 'static> {
        Box::new(FrozenGpuDynKVCacheState {
            node_id: self.node_id,
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.to_device_tensor().cloned().unwrap()),
        })
    }
}

impl FrozenOpState for FrozenGpuDynKVCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(GpuDynKVCacheState {
            node_id: self.node_id,
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.into_tensor().into_tvalue()),
        })
    }
}

#[derive(Clone)]
pub struct GpuDynKVCache {
    pub name: String,
    pub past_sequence_fact: TypedFact,
    pub input_sequence_fact: TypedFact,
    pub axis: usize,
}

impl GpuDynKVCache {
    pub fn from_tract_transformers(op: &DynKeyValueCache) -> Self {
        Self {
            name: op.name.clone(),
            axis: op.axis,
            past_sequence_fact: op.past_sequence_fact.clone(),
            input_sequence_fact: op.input_sequence_fact.clone(),
        }
    }
}

impl std::fmt::Debug for GpuDynKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GpuDynKVCache({}, axis={})", self.name, self.axis)
    }
}

impl PartialEq for GpuDynKVCache {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.axis == other.axis
            && self.past_sequence_fact == other.past_sequence_fact
            && self.input_sequence_fact == other.input_sequence_fact
    }
}

impl Eq for GpuDynKVCache {}

impl std::hash::Hash for GpuDynKVCache {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.axis.hash(state);
    }
}

impl Op for GpuDynKVCache {
    fn name(&self) -> StaticName {
        "GpuDynKVCache".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuDynKVCache {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuDynKVCacheState::new(
            node_id,
            self.name.clone(),
            self.axis,
            self.past_sequence_fact.clone(),
            None,
        ))))
    }
}

impl TypedOp for GpuDynKVCache {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut facts = crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            fact.shape.set(
                self.axis,
                self.past_sequence_fact.shape.dims()[self.axis].clone()
                    + self.input_sequence_fact.shape.dims()[self.axis].clone(),
            );
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))?;
        facts[0].as_device_fact_mut().unwrap().state_owned = true;
        Ok(facts)
    }

    as_op!();
}
