//! A weight known by its name, standing in for bytes that live elsewhere.

use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, OnceLock};

use crate::TractResult;
use crate::datum::DatumType;
use crate::exotic::ExoticFact;
use crate::internal::*;
use crate::tensor::storage::{PlainStorage, TensorStorage};

/// Produces the bytes of a named weight, as a plain tensor of its datum type
/// and shape.
pub trait WeightSource: Send + Sync + fmt::Debug {
    fn load(&self, name: &str, dt: DatumType, shape: &[usize]) -> TractResult<Tensor>;
}

/// Storage for a weight that is known by its name and fetched from a
/// [`WeightSource`] the first time its bytes are read.
///
/// Plain in layout and out of ram until then, so the eager evaluations and
/// constant folding that read a constant's bytes leave it alone, and the name
/// it was loaded under survives model transformations. Without a source it
/// can never be read, which makes any attempt to fold it an error rather than
/// a silent rewrite.
pub struct NamedStorage {
    name: String,
    dt: DatumType,
    shape: TVec<usize>,
    source: Option<Arc<dyn WeightSource>>,
    host: OnceLock<Arc<Tensor>>,
}

impl NamedStorage {
    pub fn new(
        name: impl Into<String>,
        dt: DatumType,
        shape: &[usize],
        source: Option<Arc<dyn WeightSource>>,
    ) -> NamedStorage {
        NamedStorage { name: name.into(), dt, shape: shape.into(), source, host: OnceLock::new() }
    }

    /// A tensor of `dt` and `shape` over this storage.
    pub fn into_tensor(self) -> Tensor {
        let dt = self.dt;
        let shape = self.shape.clone();
        Tensor::from_storage(dt, &shape, self)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The same weight, fetched from `source`.
    pub fn with_source(&self, source: Arc<dyn WeightSource>) -> NamedStorage {
        NamedStorage::new(self.name.clone(), self.dt, &self.shape, Some(source))
    }

    pub fn is_materialized(&self) -> bool {
        self.host.get().is_some()
    }

    fn materialize(&self) -> TractResult<&Arc<Tensor>> {
        if let Some(host) = self.host.get() {
            return Ok(host);
        }
        let source = self
            .source
            .as_ref()
            .with_context(|| format!("Weight {:?} has no source to be read from", self.name))?;
        let tensor = source
            .load(&self.name, self.dt, &self.shape)
            .with_context(|| format!("Loading weight {:?}", self.name))?;
        ensure!(
            tensor.datum_type() == self.dt && tensor.shape() == &*self.shape,
            "Weight {:?} loaded as {:?} {:?}, expected {:?} {:?}",
            self.name,
            tensor.datum_type(),
            tensor.shape(),
            self.dt,
            self.shape
        );
        Ok(self.host.get_or_init(|| tensor.into_arc_tensor()))
    }
}

impl PartialEq for NamedStorage {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.dt == other.dt && self.shape == other.shape
    }
}

impl Eq for NamedStorage {}

impl fmt::Debug for NamedStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for NamedStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.is_materialized() { "loaded" } else { "not loaded" };
        write!(f, "Named({:?}, {state})", self.name)
    }
}

impl TensorStorage for NamedStorage {
    fn byte_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dt.size_of()
    }

    fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        let host = OnceLock::new();
        if let Some(h) = self.host.get() {
            let _ = host.set(Arc::clone(h));
        }
        Box::new(NamedStorage {
            name: self.name.clone(),
            dt: self.dt,
            shape: self.shape.clone(),
            source: self.source.clone(),
            host,
        })
    }

    fn as_plain_ram(&self) -> Option<&PlainStorage> {
        self.host.get().and_then(|h| h.as_plain_ram_storage())
    }

    fn as_plain_ram_mut(&mut self) -> Option<&mut PlainStorage> {
        self.materialize().ok()?;
        Arc::make_mut(self.host.get_mut()?).as_plain_ram_storage_mut()
    }

    fn into_plain_ram(self: Box<Self>) -> Option<PlainStorage> {
        let me = *self;
        me.materialize().ok()?;
        let host = me.host.into_inner()?;
        let host = Arc::try_unwrap(host).unwrap_or_else(|shared| (*shared).clone());
        host.into_blob().ok().map(PlainStorage::from)
    }

    fn dyn_hash(&self, mut state: &mut dyn std::hash::Hasher) {
        self.name.hash(&mut state);
    }

    fn exotic_fact(&self, _shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        Ok(None)
    }

    fn is_exotic(&self) -> bool {
        false
    }

    fn in_ram(&self) -> bool {
        self.is_materialized()
    }

    fn stays_lazy(&self) -> bool {
        true
    }

    fn materialize_plain_ram(&self) -> TractResult<&PlainStorage> {
        self.materialize()?
            .as_plain_ram_storage()
            .context("Weight source did not produce plain storage")
    }
}
