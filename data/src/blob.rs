use num_traits::Zero;

use crate::{TractError, TractResult};
use std::alloc::*;
use std::fmt::Display;
use std::hash::Hash;
use std::ptr::null_mut;

#[derive(Eq)]
pub struct Blob {
    layout: std::alloc::Layout,
    data: *mut u8,
}

impl Default for Blob {
    #[inline]
    fn default() -> Blob {
        Blob::from_bytes(&[]).unwrap()
    }
}

impl Clone for Blob {
    #[inline]
    fn clone(&self) -> Self {
        Blob::from_bytes_alignment(self, self.layout.align()).unwrap()
    }
}

impl Drop for Blob {
    #[inline]
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe { dealloc(self.data, self.layout) }
        }
    }
}

impl PartialEq for Blob {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.layout == other.layout && self.as_bytes() == other.as_bytes()
    }
}

impl Hash for Blob {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.layout.align().hash(state);
        self.as_bytes().hash(state);
    }
}

impl Blob {
    #[inline]
    pub unsafe fn new_for_size_and_align(size: usize, align: usize) -> Blob {
        Self::for_layout(Layout::from_size_align_unchecked(size, align))
    }

    #[inline]
    pub unsafe fn ensure_size_and_align(&mut self, size: usize, align: usize) {
        if size > self.layout.size() || align > self.layout.align() {
            if !self.data.is_null() {
                std::alloc::dealloc(self.data as _, self.layout);
            }
            self.layout = Layout::from_size_align_unchecked(size, align);
            self.data = std::alloc::alloc(self.layout);
            assert!(!self.data.is_null());
        }
    }

    #[inline]
    pub unsafe fn for_layout(layout: Layout) -> Blob {
        let mut data = null_mut();
        if layout.size() > 0 {
            data = unsafe { alloc(layout) };
            assert!(!data.is_null(), "failed to allocate {layout:?}");
        }
        Blob { layout, data }
    }

    #[inline]
    pub fn from_bytes(s: &[u8]) -> TractResult<Blob> {
        Self::from_bytes_alignment(s, 128)
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        if self.data.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.layout.size()) }
        }
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        if self.data.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.data, self.layout.size()) }
        }
    }

    #[inline]
    pub fn from_bytes_alignment(s: &[u8], alignment: usize) -> TractResult<Blob> {
        unsafe {
            let layout = Layout::from_size_align(s.len(), alignment)?;
            let blob = Self::for_layout(layout);
            if s.len() > 0 {
                std::ptr::copy_nonoverlapping(s.as_ptr(), blob.data, s.len());
            }
            Ok(blob)
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl std::ops::Deref for Blob {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::ops::DerefMut for Blob {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

impl std::fmt::Display for Blob {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        assert!(self.data.is_null() == self.layout.size().is_zero());
        write!(
            fmt,
            "Blob of {} bytes (align @{}): {} {}",
            self.len(),
            self.layout.align(),
            String::from_utf8(
                self.iter().take(20).copied().flat_map(std::ascii::escape_default).collect::<Vec<u8>>()
            )
            .unwrap(),
            if self.len() >= 20 { "[...]" } else { "" }
        )
    }
}

impl std::fmt::Debug for Blob {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        <Self as Display>::fmt(self, fmt)
    }
}

impl<'a> TryFrom<&'a [u8]> for Blob {
    type Error = TractError;
    #[inline]
    fn try_from(s: &[u8]) -> Result<Blob, Self::Error> {
        Blob::from_bytes(s)
    }
}

unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}
