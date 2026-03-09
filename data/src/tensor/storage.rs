use std::alloc::Layout;
use std::fmt;
use std::hash::Hash;

use crate::blob::Blob;

/// Trait abstracting over tensor storage backends.
///
/// `Tensor` holds a `Box<dyn TensorStorage>`, dispatching through this trait.
/// `DenseStorage` is the primary implementation backed by a contiguous `Blob`.
pub trait TensorStorage: Send + Sync + fmt::Debug + fmt::Display {
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> &mut [u8];
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn layout(&self) -> &Layout;
    fn is_empty(&self) -> bool;
    fn byte_len(&self) -> usize;
    fn deep_clone(&self) -> Box<dyn TensorStorage>;
    fn same_as(&self, other: &dyn TensorStorage) -> bool;
    /// Attempt to convert the storage into a `Blob`. Returns `None` if the
    /// storage backend does not support this (e.g. non-dense storage).
    fn try_into_blob(self: Box<Self>) -> Option<Blob>;
}

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
    fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.0.as_bytes().as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_bytes_mut().as_mut_ptr()
    }

    #[inline]
    fn layout(&self) -> &Layout {
        self.0.layout()
    }

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

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        !self.0.is_empty()
            && self.0.as_bytes().as_ptr() == other.as_bytes().as_ptr()
            && self.0.len() == other.byte_len()
    }

    fn try_into_blob(self: Box<Self>) -> Option<Blob> {
        Some(self.0)
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
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            StorageKind::Dense(d) => d.0.as_bytes(),
            StorageKind::Other(o) => o.as_bytes(),
        }
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            StorageKind::Dense(d) => d.0.as_bytes_mut(),
            StorageKind::Other(o) => o.as_bytes_mut(),
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            StorageKind::Dense(d) => d.0.as_bytes().as_ptr(),
            StorageKind::Other(o) => o.as_ptr(),
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            StorageKind::Dense(d) => d.0.as_bytes_mut().as_mut_ptr(),
            StorageKind::Other(o) => o.as_mut_ptr(),
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        match self {
            StorageKind::Dense(d) => d.0.layout(),
            StorageKind::Other(o) => o.layout(),
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
    pub fn same_as(&self, other: &StorageKind) -> bool {
        match (self, other) {
            (StorageKind::Dense(a), StorageKind::Dense(b)) => {
                !a.0.is_empty()
                    && a.0.as_bytes().as_ptr() == b.0.as_bytes().as_ptr()
                    && a.0.len() == b.0.len()
            }
            _ => false,
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

    pub fn try_into_blob(self) -> Option<Blob> {
        match self {
            StorageKind::Dense(d) => Some(d.0),
            StorageKind::Other(o) => o.try_into_blob(),
        }
    }
}
