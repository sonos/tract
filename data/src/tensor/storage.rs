use std::alloc::Layout;
use std::fmt;
use std::hash::Hash;

use crate::blob::Blob;

/// Trait abstracting over tensor storage backends.
///
/// `Tensor` holds a `Box<dyn TensorStorage>`, dispatching through this trait.
/// `DenseStorage` is the primary implementation backed by a contiguous `Blob`.
pub trait TensorStorage: Send + Sync + fmt::Debug + fmt::Display {
    fn byte_len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn deep_clone(&self) -> Box<dyn TensorStorage>;
    fn same_as(&self, other: &dyn TensorStorage) -> bool;
    fn as_dense(&self) -> Option<&DenseStorage>;
    fn as_dense_mut(&mut self) -> Option<&mut DenseStorage>;
    fn into_dense(self: Box<Self>) -> Option<DenseStorage>;
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

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.as_dense() {
            !self.0.is_empty()
                && self.0.as_bytes().as_ptr() == other.as_ptr()
                && self.0.len() == other.byte_len()
        } else {
            false
        }
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
}
