use std::alloc::Layout;
use std::fmt;
use std::hash::Hash;

use crate::blob::Blob;
use downcast_rs::{Downcast, impl_downcast};

/// Trait abstracting over tensor storage backends.
///
/// `PlainStorage` is the primary implementation backed by a contiguous `Blob`.
/// Non-plain backends are held behind `StorageKind::Exotic(Box<dyn TensorStorage>)`.
pub trait TensorStorage: Send + Sync + fmt::Debug + fmt::Display + Downcast {
    fn byte_len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn deep_clone(&self) -> Box<dyn TensorStorage>;
    fn as_plain(&self) -> Option<&PlainStorage>;
    fn as_plain_mut(&mut self) -> Option<&mut PlainStorage>;
    fn into_plain(self: Box<Self>) -> Option<PlainStorage>;
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
    fn same_as(&self, other: &dyn TensorStorage) -> bool;
}
impl_downcast!(TensorStorage);

/// Plain, contiguous storage backed by a `Blob`.
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

    fn as_plain(&self) -> Option<&PlainStorage> {
        Some(self)
    }

    fn as_plain_mut(&mut self) -> Option<&mut PlainStorage> {
        Some(self)
    }

    fn into_plain(self: Box<Self>) -> Option<PlainStorage> {
        Some(*self)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u8(0);
        state.write(self.0.as_bytes());
    }

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.as_plain() { self == other } else { false }
    }
}

/// Inline enum replacing `Box<dyn TensorStorage>`.
///
/// The common `Plain` case stays inline (no heap alloc, no vtable indirection).
/// `Exotic` covers non-plain backends behind a single Box indirection.
#[allow(dead_code)]
pub(crate) enum StorageKind {
    Plain(PlainStorage),
    Exotic(Box<dyn TensorStorage>),
}

impl StorageKind {
    #[inline]
    pub fn as_plain(&self) -> Option<&PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.as_plain(),
        }
    }

    #[inline]
    pub fn as_plain_mut(&mut self) -> Option<&mut PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.as_plain_mut(),
        }
    }

    #[inline]
    pub fn into_plain(self) -> Option<PlainStorage> {
        match self {
            StorageKind::Plain(d) => Some(d),
            StorageKind::Exotic(o) => o.into_plain(),
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

    pub fn same_as(&self, other: &StorageKind) -> bool {
        match (self, other) {
            (StorageKind::Plain(a), StorageKind::Plain(b)) => a == b,
            (StorageKind::Exotic(a), StorageKind::Exotic(b)) => a.same_as(b.as_ref()),
            _ => false,
        }
    }
}
