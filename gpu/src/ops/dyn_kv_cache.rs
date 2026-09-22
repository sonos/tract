use crate::device::get_context;
use crate::fact::DeviceTypedFactExt;
use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use derive_new::new;
use tract_core::internal::*;
use tract_transformers::ops::dyn_kv_cache::{DynKeyValueCache, DynKeyValueCacheState};

#[derive(Debug, Clone, new)]
pub struct GpuDynKVCacheState {
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    /// Spare capacity along `axis`; the cache is the `[0..len]` prefix of it.
    /// A decode appends one token per turn, so growing by concatenation would
    /// copy the whole past into a new buffer every turn -- quadratic over a
    /// decode. Doubling the buffer and writing at the cursor is the classic
    /// `Vec` trade: amortized O(1) per turn, and the tail is never read.
    #[new(default)]
    buffer: Option<DeviceTensor>,
    #[new(default)]
    len: usize,
    #[new(default)]
    reallocs: usize,
}

impl GpuDynKVCacheState {
    /// Extent of the buffer on the cache axis, live prefix and spare tail both.
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().map(|b| b.shape()[self.axis]).unwrap_or(0)
    }

    /// Shape of the live cache: the buffer's, cut to `len` on the cache axis.
    fn valid_shape(&self) -> Option<TVec<usize>> {
        let mut shape: TVec<usize> = self.buffer.as_ref()?.shape().into();
        shape[self.axis] = self.len;
        Some(shape)
    }

    /// The live cache as a tensor of its own, packed.
    ///
    /// Always a copy, even where the prefix fills the buffer exactly: the buffer
    /// is written by the next append, and what leaves here is held by whoever
    /// asked for as long as they like.
    fn valid(&self) -> TractResult<DeviceTensor> {
        let buffer = self.buffer.as_ref().context("KV cache was never initialized")?;
        let shape = self.valid_shape().unwrap();
        let out = DeviceTensor::uninitialized_dt(buffer.datum_type(), &shape)?;
        get_context()?.assign_slice(&out, 0..self.len, buffer, 0..self.len, self.axis)?;
        Ok(out)
    }

    /// Append `input` at the cursor, doubling the buffer when it no longer fits.
    fn push(&mut self, input: &DeviceTensor) -> TractResult<()> {
        let new = input.shape()[self.axis];
        if new == 0 {
            return Ok(());
        }
        if self.len + new > self.capacity() {
            self.grow(self.len + new, input)?;
        }
        let buffer = self.buffer.as_ref().unwrap();
        get_context()?.assign_slice(buffer, self.len..self.len + new, input, 0..new, self.axis)?;
        self.len += new;
        Ok(())
    }

    /// Re-seat the live prefix in a buffer of at least `needed` on the cache
    /// axis. `template` carries the other axes, which the first grow has no
    /// buffer to read them from.
    fn grow(&mut self, needed: usize, template: &DeviceTensor) -> TractResult<()> {
        let mut shape: TVec<usize> = self
            .buffer
            .as_ref()
            .map(|b| b.shape().into())
            .unwrap_or_else(|| template.shape().into());
        shape[self.axis] = (self.capacity() * 2).max(needed);
        let grown = DeviceTensor::uninitialized_dt(template.datum_type(), &shape)?;
        if let Some(buffer) = self.buffer.as_ref() {
            get_context()?.assign_slice(&grown, 0..self.len, buffer, 0..self.len, self.axis)?;
        }
        self.buffer = Some(grown);
        self.reallocs += 1;
        Ok(())
    }
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
        self.len = kv_cache.shape()[self.axis];
        self.buffer = Some(kv_cache.into_tensor().into_device()?);
        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        if self.buffer.is_some() {
            states.push(self.valid()?.to_host()?.into_tensor().into_tvalue());
            Ok(())
        } else {
            bail!("KV cache {} was never initialized", self.name)
        }
    }

    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }

    fn has_init_tensor_fact(&self) -> bool {
        true
    }

    fn resolve_symbols(&mut self, state: &mut TurnState) -> TractResult<()> {
        let shape = self.valid_shape();
        DynKeyValueCacheState::resolve_symbols(
            state,
            self.past_sequence_fact.clone(),
            shape.as_deref(),
        )
    }

    fn eval(
        &mut self,
        _ctx: &EvalContext,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1);
        let input = inputs.into_iter().next().unwrap();
        self.push(input.to_device_tensor()?)?;
        Ok(tvec!(self.valid()?.into_tensor().into_tvalue()))
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("GpuDynKVCache is not lane-aware: the cache has no lane axis")
    }
}

impl GpuDynKVCacheState {
    /// How many times the buffer has been re-seated: logarithmic in the number
    /// of tokens a decode appends, and the evidence that it is.
    pub fn reallocs(&self) -> usize {
        self.reallocs
    }

    /// Drop everything past `len` on the cache axis.
    ///
    /// The bytes stay where they are: the cache is the one tensor an application
    /// rolls back between runs, and a shorter prefix of the same buffer is the
    /// whole of it.
    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        if self.buffer.is_none() {
            return Ok(());
        }
        ensure!(len <= self.len, "Can not truncate a cache of {} to {len}", self.len);
        self.len = len;
        Ok(())
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
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuDynKVCacheState::new(
            self.name.clone(),
            self.axis,
            self.past_sequence_fact.clone(),
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
