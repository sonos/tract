use std::fmt::{self, Display};
use std::sync::OnceLock;

use tract_core::internal::*;

use super::DeviceTensor;

/// Host storage for a tensor whose bytes are still on a device.
///
/// A device tensor crosses the model boundary wrapped in one of these. The
/// result is a host tensor as far as facts, datum type and shape go, but
/// nothing is copied back until something reads the bytes: `as_bytes`,
/// `as_slice`, `try_as_plain` and friends materialize it once and keep the
/// result, and a caller that only hands it to the next `run()` never pays a
/// readback at all, because `DeviceSync` takes the device tensor back out of
/// it.
///
/// Only owned device tensors belong here. An arena view is turn-scoped
/// storage, so it is copied out at the boundary rather than held.
#[derive(Debug)]
pub struct LazyHostStorage {
    /// Dropped once the host side is written to: a mutated tensor is a plain
    /// host tensor and must not alias a device buffer anyone else holds.
    device: Option<DeviceTensor>,
    host: OnceLock<Tensor>,
}

impl LazyHostStorage {
    pub fn new(device: DeviceTensor) -> Self {
        LazyHostStorage { device: Some(device), host: OnceLock::new() }
    }

    /// The device tensor still backing this storage, if it has not been
    /// detached by a host-side mutation.
    pub fn device(&self) -> Option<&DeviceTensor> {
        self.device.as_ref()
    }

    /// True once the bytes have been brought back to host.
    pub fn is_materialized(&self) -> bool {
        self.host.get().is_some()
    }

    pub fn datum_type(&self) -> DatumType {
        match (&self.device, self.host.get()) {
            (Some(d), _) => d.datum_type(),
            (None, Some(h)) => h.datum_type(),
            (None, None) => DatumType::U8,
        }
    }

    pub fn shape(&self) -> TVec<usize> {
        match (&self.device, self.host.get()) {
            (Some(d), _) => d.shape().into(),
            (None, Some(h)) => h.shape().into(),
            (None, None) => tvec![],
        }
    }

    /// Wrap into a `Tensor` carrying the device tensor's datum type and shape.
    pub fn into_tensor(self) -> Tensor {
        let dt = self.datum_type();
        let shape = self.shape();
        Tensor::from_storage(dt, &shape, self)
    }

    fn materialize(&self) -> TractResult<&Tensor> {
        if let Some(host) = self.host.get() {
            return Ok(host);
        }
        let device = self
            .device
            .as_ref()
            .context("Lazy host storage has neither a device tensor nor host bytes")?;
        let host = device.to_host().context("While materializing a lazy host tensor")?;
        // to_host hands out an Arc the device tensor may still hold a share of;
        // take it when we can, copy when we cannot.
        let host = Arc::try_unwrap(host).unwrap_or_else(|shared| (*shared).clone());
        Ok(self.host.get_or_init(|| host))
    }
}

impl PartialEq for LazyHostStorage {
    fn eq(&self, other: &Self) -> bool {
        match (&self.device, &other.device) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.host.get() == other.host.get(),
            _ => false,
        }
    }
}

impl Eq for LazyHostStorage {}

impl Display for LazyHostStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.host.get() {
            Some(host) => write!(f, "LazyHost(materialized): {{ {host:?} }}"),
            None => {
                write!(f, "LazyHost(on device): {{ {:?} {:?} }}", self.datum_type(), self.shape())
            }
        }
    }
}

impl TensorStorage for LazyHostStorage {
    fn byte_len(&self) -> usize {
        match (&self.device, self.host.get()) {
            (Some(d), _) => d.len() * d.datum_type().size_of(),
            (None, Some(h)) => h.len() * h.datum_type().size_of(),
            (None, None) => 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        let host = OnceLock::new();
        if let Some(h) = self.host.get() {
            let _ = host.set(h.clone());
        }
        Box::new(LazyHostStorage { device: self.device.clone(), host })
    }

    fn as_plain(&self) -> Option<&PlainStorage> {
        // Cheap predicate: answers what is here now, never triggers a readback.
        self.host.get().and_then(|h| h.as_plain_storage())
    }

    fn as_plain_mut(&mut self) -> Option<&mut PlainStorage> {
        self.materialize().ok()?;
        // Writing to the host side detaches the device tensor: from here on
        // this is an ordinary host tensor, and nothing aliases device memory.
        self.device = None;
        self.host.get_mut().and_then(|h| h.as_plain_storage_mut())
    }

    fn into_plain(self: Box<Self>) -> Option<PlainStorage> {
        let me = *self;
        me.materialize().ok()?;
        let host = me.host.into_inner()?;
        host.into_blob().ok().map(PlainStorage::from)
    }

    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {
        // no meaningful hash for device memory, as for DeviceTensor
    }

    fn exotic_fact(&self, _shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        // A lazily-materialized host tensor is a plain host tensor: it carries
        // its real datum type and shape, and no exotic fact.
        Ok(None)
    }

    fn materialize_plain(&self) -> TractResult<&PlainStorage> {
        self.materialize()?
            .as_plain_storage()
            .context("Device readback did not produce plain storage")
    }

    fn slice(
        &self,
        _dt: DatumType,
        _shape: &[usize],
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<Tensor>> {
        if self.host.get().is_some() {
            // Already back on host: the generic copy is cheaper than a readback.
            return Ok(None);
        }
        let Some(device) = self.device.as_ref() else { return Ok(None) };
        let Some(sliced) = device.dense_slice(axis, start, end)? else { return Ok(None) };
        Ok(Some(LazyHostStorage::new(sliced).into_tensor()))
    }
}
