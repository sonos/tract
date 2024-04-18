use crate::TractResult;
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
    fn default() -> Blob {
        Blob::from_bytes(&[]).unwrap()
    }
}

impl Clone for Blob {
    fn clone(&self) -> Self {
        Blob::from_bytes_alignment(self, self.layout.align()).unwrap()
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe { dealloc(self.data, self.layout) }
        }
    }
}

impl PartialEq for Blob {
    fn eq(&self, other: &Self) -> bool {
        self.layout == other.layout && self.as_bytes() == other.as_bytes()
    }
}

impl Hash for Blob {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.layout.align().hash(state);
        self.as_bytes().hash(state);
    }
}

impl Blob {
    pub fn from_bytes(s: &[u8]) -> TractResult<Blob> {
        Self::from_bytes_alignment(s, 128)
    }

    fn as_bytes(&self) -> &[u8] {
        if self.data.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.layout.size()) }
        }
    }

    pub fn from_bytes_alignment(s: &[u8], alignment: usize) -> TractResult<Blob> {
        unsafe {
            let layout = Layout::from_size_align_unchecked(s.len(), alignment);
            let mut data = null_mut();
            if layout.size() > 0 {
                data = alloc(layout);
                std::ptr::copy_nonoverlapping(s.as_ptr(), data, s.len());
            }
            Ok(Blob { layout, data })
        }
    }
}

impl std::ops::Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::fmt::Display for Blob {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt,
            "Blob of {} bytes (align @{}): {}",
            self.len(),
            self.layout.align(),
            String::from_utf8_lossy(self)
        )
    }
}

impl std::fmt::Debug for Blob {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        <Self as Display>::fmt(self, fmt)
    }
}

impl<'a> TryFrom<&'a [u8]> for Blob {
    type Error = anyhow::Error;
    fn try_from(s: &[u8]) -> Result<Blob, Self::Error> {
        Blob::from_bytes(s)
    }
}

unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}
