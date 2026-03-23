use std::alloc::Layout;
use std::fmt;
use std::hash::Hash;

use crate::blob::Blob;
use downcast_rs::{Downcast, impl_downcast};

/// Trait abstracting over tensor storage backends.
///
/// `DenseStorage` is the primary implementation backed by a contiguous `Blob`.
/// Non-dense backends are held behind `StorageKind::Other(Box<dyn TensorStorage>)`.
pub trait TensorStorage: Send + Sync + fmt::Debug + fmt::Display + Downcast {
    fn byte_len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn deep_clone(&self) -> Box<dyn TensorStorage>;
    fn as_dense(&self) -> Option<&DenseStorage>;
    fn as_dense_mut(&mut self) -> Option<&mut DenseStorage>;
    fn into_dense(self: Box<Self>) -> Option<DenseStorage>;
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
    fn same_as(&self, other: &dyn TensorStorage) -> bool;
}
impl_downcast!(TensorStorage);

/// Dense, contiguous storage backed by a `Blob`.
#[derive(Eq)]
pub struct DenseStorage(pub(crate) Blob);

impl DenseStorage {
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

impl Default for DenseStorage {
    #[inline]
    fn default() -> Self {
        DenseStorage(Blob::default())
    }
}

impl Clone for DenseStorage {
    #[inline]
    fn clone(&self) -> Self {
        DenseStorage(self.0.clone())
    }
}

impl Hash for DenseStorage {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for DenseStorage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<Blob> for DenseStorage {
    #[inline]
    fn from(blob: Blob) -> Self {
        DenseStorage(blob)
    }
}

impl std::ops::Deref for DenseStorage {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl std::ops::DerefMut for DenseStorage {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }
}

impl fmt::Debug for DenseStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for DenseStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl TensorStorage for DenseStorage {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn byte_len(&self) -> usize {
        self.0.len()
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(DenseStorage(self.0.clone()))
    }

    fn as_dense(&self) -> Option<&DenseStorage> {
        Some(self)
    }

    fn as_dense_mut(&mut self) -> Option<&mut DenseStorage> {
        Some(self)
    }

    fn into_dense(self: Box<Self>) -> Option<DenseStorage> {
        Some(*self)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.0.as_bytes());
    }

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.as_dense() { self == other } else { false }
    }
}

/// Inline enum replacing `Box<dyn TensorStorage>`.
///
/// The common `Dense` case stays inline (no heap alloc, no vtable indirection).
/// `Other` covers future non-dense backends behind a single Box indirection.
#[allow(dead_code)]
pub(crate) enum StorageKind {
    Dense(DenseStorage),
    Other(Box<dyn TensorStorage>),
}

impl StorageKind {
    #[inline]
    pub fn as_dense(&self) -> Option<&DenseStorage> {
        match self {
            StorageKind::Dense(d) => Some(d),
            StorageKind::Other(o) => o.as_dense(),
        }
    }

    #[inline]
    pub fn as_dense_mut(&mut self) -> Option<&mut DenseStorage> {
        match self {
            StorageKind::Dense(d) => Some(d),
            StorageKind::Other(o) => o.as_dense_mut(),
        }
    }

    #[inline]
    pub fn into_dense(self) -> Option<DenseStorage> {
        match self {
            StorageKind::Dense(d) => Some(d),
            StorageKind::Other(o) => o.into_dense(),
        }
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        match self {
            StorageKind::Dense(d) => d.0.len(),
            StorageKind::Other(o) => o.byte_len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            StorageKind::Dense(d) => d.0.is_empty(),
            StorageKind::Other(o) => o.is_empty(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn deep_clone(&self) -> StorageKind {
        match self {
            StorageKind::Dense(d) => StorageKind::Dense(d.clone()),
            StorageKind::Other(o) => StorageKind::Other(o.deep_clone()),
        }
    }

    #[inline]
    pub fn as_storage(&self) -> &dyn TensorStorage {
        match self {
            StorageKind::Dense(d) => d,
            StorageKind::Other(o) => o.as_ref(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn as_storage_mut(&mut self) -> &mut dyn TensorStorage {
        match self {
            StorageKind::Dense(d) => d,
            StorageKind::Other(o) => o.as_mut(),
        }
    }

    pub fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        match self {
            StorageKind::Dense(d) => state.write(d.as_bytes()),
            StorageKind::Other(o) => o.dyn_hash(state),
        }
    }

    pub fn same_as(&self, other: &StorageKind) -> bool {
        match (self, other) {
            (StorageKind::Dense(a), StorageKind::Dense(b)) => a == b,
            (StorageKind::Other(a), StorageKind::Other(b)) => a.same_as(b.as_ref()),
            _ => false,
        }
    }
}
