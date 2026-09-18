use std::alloc::Layout;
use std::fmt;
use std::hash::Hash;

use crate::TractResult;
use crate::blob::Blob;
use crate::datum::DatumType;
use crate::dyn_eq::DynEq;
use crate::exotic::ExoticFact;
use crate::tensor::Tensor;
use downcast_rs::{Downcast, impl_downcast};

/// Trait abstracting over tensor storage backends.
///
/// Two independent axes describe one: layout, `is_exotic`, and placement,
/// `in_ram`. `PlainStorage` is the primary implementation, plain and in ram by
/// construction; every other backend is held behind
/// `StorageKind::Exotic(Box<dyn TensorStorage>)`, whichever pair of answers it
/// gives.
pub trait TensorStorage: Send + Sync + fmt::Debug + fmt::Display + DynEq + Downcast {
    fn byte_len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn deep_clone(&self) -> Box<dyn TensorStorage>;
    fn as_plain_ram(&self) -> Option<&PlainStorage>;
    fn as_plain_ram_mut(&mut self) -> Option<&mut PlainStorage>;
    fn into_plain_ram(self: Box<Self>) -> Option<PlainStorage>;
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
    /// Build the `ExoticFact` that describes this storage for use in `TypedFact`.
    ///
    /// Plain storage returns `None`. Exotic storages should return the
    /// appropriate fact so that `From<Arc<Tensor>> for TypedFact` preserves
    /// exotic-ness.
    fn exotic_fact(&self, shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>>;

    /// True when the tensor's datum type and shape do not describe it on their
    /// own, so a fact over it carries an `ExoticFact`.
    ///
    /// The layout axis, orthogonal to `in_ram`: all four combinations exist,
    /// a dense tensor in device memory being plain and out of ram,
    /// block-quant weights in device memory exotic and out of ram.
    ///
    /// Defaults to true: a storage is exotic until it says otherwise, so one
    /// that forgets is refused where a fact is required rather than silently
    /// mistyped.
    fn is_exotic(&self) -> bool {
        true
    }

    /// True when the bytes are in host memory, readable without a transfer.
    ///
    /// The placement axis. Defaults to true: storage holds its own bytes unless
    /// it says otherwise. Answers for the bytes in whatever layout the storage
    /// keeps them -- block-quant storage is in ram while packed -- so it takes
    /// both axes, `as_plain_ram`, for a plain read to be sure to work.
    fn in_ram(&self) -> bool {
        true
    }

    /// Plain storage for the tensor's bytes, producing it if this storage can.
    ///
    /// This is the accessor path: `Tensor::as_bytes` and friends go through it,
    /// so a storage that holds its bytes somewhere else (on a device, say) gets
    /// a chance to bring them back here, and to keep the result so the next
    /// access is free. `as_plain_ram` stays the cheap accessor: it answers with
    /// what is available right now and never produces anything.
    fn materialize_plain_ram(&self) -> TractResult<&PlainStorage> {
        self.as_plain_ram().ok_or_else(|| anyhow::anyhow!("Tensor storage is not plain"))
    }

    /// Slice along `axis`, if this storage can serve it in its own memory --
    /// for free where the slice is already a range it holds, by a copy it can
    /// make where it is otherwise.
    ///
    /// `None` means "not capable" and the caller falls back to a generic copy
    /// through host memory, so an implementation is free to refuse any case it
    /// cannot serve -- but storage whose bytes are not host bytes has no such
    /// fallback to decline to, and refusing there is a panic. What it must not
    /// do is return a tensor that is not a valid dense one: `Some` is a claim
    /// that the result stands on its own everywhere a tensor is accepted.
    fn slice(
        &self,
        _dt: DatumType,
        _shape: &[usize],
        _axis: usize,
        _start: usize,
        _end: usize,
    ) -> TractResult<Option<Tensor>> {
        Ok(None)
    }
}
impl_downcast!(TensorStorage);
crate::eq_trait_object!(TensorStorage);

/// Plain, contiguous storage backed by a `Blob`: plain in layout and in ram,
/// which is what every other storage is measured against.
#[derive(Eq)]
pub struct PlainStorage(pub(crate) Blob);

impl PlainStorage {
    #[inline]
    pub fn layout(&self) -> &Layout {
        self.0.layout()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_bytes().as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_bytes_mut().as_mut_ptr()
    }

    #[inline]
    pub fn into_blob(self) -> Blob {
        self.0
    }
}

impl Default for PlainStorage {
    #[inline]
    fn default() -> Self {
        PlainStorage(Blob::default())
    }
}

impl Clone for PlainStorage {
    #[inline]
    fn clone(&self) -> Self {
        PlainStorage(self.0.clone())
    }
}

impl Hash for PlainStorage {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for PlainStorage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<Blob> for PlainStorage {
    #[inline]
    fn from(blob: Blob) -> Self {
        PlainStorage(blob)
    }
}

impl std::ops::Deref for PlainStorage {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl std::ops::DerefMut for PlainStorage {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }
}

impl fmt::Debug for PlainStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for PlainStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl TensorStorage for PlainStorage {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn byte_len(&self) -> usize {
        self.0.len()
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(PlainStorage(self.0.clone()))
    }

    fn as_plain_ram(&self) -> Option<&PlainStorage> {
        Some(self)
    }

    fn as_plain_ram_mut(&mut self) -> Option<&mut PlainStorage> {
        Some(self)
    }

    fn into_plain_ram(self: Box<Self>) -> Option<PlainStorage> {
        Some(*self)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u8(0);
        state.write(self.0.as_bytes());
    }

    fn exotic_fact(&self, _shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        Ok(None)
    }

    fn is_exotic(&self) -> bool {
        false
    }
}

/// Inline enum replacing `Box<dyn TensorStorage>`.
///
/// The common `Plain` case stays inline (no heap alloc, no vtable indirection).
/// `Exotic` covers every other backend behind a single Box indirection, whether
/// or not it is exotic in the fact sense -- `is_exotic` answers that.
#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StorageKind {
    Plain(PlainStorage),
    Exotic(Box<dyn TensorStorage>),
}

impl StorageKind {
    #[inline]
    pub fn as_plain_ram(&self) -> Option<&PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.as_plain_ram(),
        }
    }

    #[inline]
    pub fn as_plain_ram_mut(&mut self) -> Option<&mut PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.as_plain_ram_mut(),
        }
    }

    #[inline]
    pub fn into_plain_ram(self) -> Option<PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.into_plain_ram(),
        }
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        match self {
            StorageKind::Plain(d) => d.0.len(),
            StorageKind::Exotic(o) => o.byte_len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            StorageKind::Plain(d) => d.0.is_empty(),
            StorageKind::Exotic(o) => o.is_empty(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn deep_clone(&self) -> StorageKind {
        match self {
            StorageKind::Plain(d) => StorageKind::Plain(d.clone()),
            StorageKind::Exotic(o) => StorageKind::Exotic(o.deep_clone()),
        }
    }

    #[inline]
    pub fn is_exotic(&self) -> bool {
        match self {
            StorageKind::Plain(_) => false,
            StorageKind::Exotic(o) => o.is_exotic(),
        }
    }

    #[inline]
    pub fn in_ram(&self) -> bool {
        match self {
            StorageKind::Plain(_) => true,
            StorageKind::Exotic(o) => o.in_ram(),
        }
    }

    #[inline]
    pub fn materialize_plain_ram(&self) -> TractResult<&PlainStorage> {
        match self {
            StorageKind::Plain(d) => Ok(d),
            StorageKind::Exotic(o) => o.materialize_plain_ram(),
        }
    }

    #[inline]
    pub fn as_storage(&self) -> &dyn TensorStorage {
        match self {
            StorageKind::Plain(d) => d,
            StorageKind::Exotic(o) => o.as_ref(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn as_storage_mut(&mut self) -> &mut dyn TensorStorage {
        match self {
            StorageKind::Plain(d) => d,
            StorageKind::Exotic(o) => o.as_mut(),
        }
    }

    pub fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        match self {
            StorageKind::Plain(d) => {
                state.write_u8(0);
                state.write(d.as_bytes())
            }
            StorageKind::Exotic(o) => o.dyn_hash(state),
        }
    }
}
