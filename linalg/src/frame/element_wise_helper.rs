use crate::LADatum;
use std::alloc::*;
use tract_data::TractResult;

pub(crate) fn map_slice_with_alignment<T>(
    vec: &mut [T],
    f: impl Fn(&mut [T]),
    nr: usize,
    alignment_bytes: usize,
) -> TractResult<()>
where
    T: LADatum,
{
    if vec.is_empty() {
        return Ok(());
    }
    unsafe {
        TMP.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            let tmp = std::slice::from_raw_parts_mut(buffer.buffer as *mut T, nr);
            let mut compute_via_temp_buffer = |slice: &mut [T]| {
                tmp[..slice.len()].copy_from_slice(slice);
                f(tmp);
                slice.copy_from_slice(&tmp[..slice.len()])
            };
            let prefix_len = vec.as_ptr().align_offset(alignment_bytes).min(vec.len());
            if prefix_len > 0 {
                compute_via_temp_buffer(&mut vec[..prefix_len]);
            }
            let aligned_len = (vec.len() - prefix_len) / nr * nr;
            if aligned_len > 0 {
                f(&mut vec[prefix_len..][..aligned_len]);
            }
            if prefix_len + aligned_len < vec.len() {
                compute_via_temp_buffer(&mut vec[prefix_len + aligned_len..]);
            }
        })
    }
    Ok(())
}

pub(crate) fn reduce_slice_with_alignment<T>(
    vec: &[T],
    f: impl Fn(&[T]) -> T,
    nr: usize,
    alignment_bytes: usize,
    neutral: T,
    reduce: impl Fn(T, T) -> T,
) -> TractResult<T>
where
    T: LADatum,
{
    if vec.is_empty() {
        return Ok(neutral);
    }
    let mut red = neutral;
    unsafe {
        TMP.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            let tmp = std::slice::from_raw_parts_mut(buffer.buffer as *mut T, nr);
            let mut compute_via_temp_buffer = |slice: &[T], red: &mut T| {
                tmp[..slice.len()].copy_from_slice(slice);
                tmp[slice.len()..].fill(neutral);
                *red = reduce(*red, f(tmp));
            };
            let prefix_len = vec.as_ptr().align_offset(alignment_bytes).min(vec.len());
            if prefix_len > 0 {
                compute_via_temp_buffer(&vec[..prefix_len], &mut red);
            }
            let aligned_len = (vec.len() - prefix_len) / nr * nr;
            if aligned_len > 0 {
                let t = f(&vec[prefix_len..][..aligned_len]);
                red = reduce(red, t);
            }
            if prefix_len + aligned_len < vec.len() {
                compute_via_temp_buffer(&vec[prefix_len + aligned_len..], &mut red);
            }
        })
    }
    Ok(red)
}

pub(crate) fn map_reduce_slice_with_alignment<T>(
    vec: &mut [T],
    f: impl Fn(&mut [T]) -> T,
    nr: usize,
    alignment_bytes: usize,
    map_neutral: T,
    neutral: T,
    reduce: impl Fn(T, T) -> T,
) -> TractResult<T>
where
    T: LADatum,
{
    if vec.is_empty() {
        return Ok(neutral);
    }
    let mut red = neutral;
    unsafe {
        TMP.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            let tmp = std::slice::from_raw_parts_mut(buffer.buffer as *mut T, nr);
            let mut compute_via_temp_buffer = |slice: &mut [T], red: &mut T| {
                tmp[..slice.len()].copy_from_slice(slice);
                tmp[slice.len()..].fill(map_neutral);
                *red = reduce(*red, f(tmp));
                slice.copy_from_slice(&tmp[..slice.len()]);
            };
            let prefix_len = vec.as_ptr().align_offset(alignment_bytes).min(vec.len());
            if prefix_len > 0 {
                compute_via_temp_buffer(&mut vec[..prefix_len], &mut red);
            }
            let aligned_len = (vec.len() - prefix_len) / nr * nr;
            if aligned_len > 0 {
                let t = f(&mut vec[prefix_len..][..aligned_len]);
                red = reduce(red, t);
            }
            if prefix_len + aligned_len < vec.len() {
                compute_via_temp_buffer(&mut vec[prefix_len + aligned_len..], &mut red);
            }
        })
    }
    Ok(red)
}

std::thread_local! {
    static TMP: std::cell::RefCell<TempBuffer> = std::cell::RefCell::new(TempBuffer::default());
}

pub struct TempBuffer {
    pub layout: Layout,
    pub buffer: *mut u8,
}

impl Default for TempBuffer {
    fn default() -> Self {
        TempBuffer { layout: Layout::new::<()>(), buffer: std::ptr::null_mut() }
    }
}

impl TempBuffer {
    pub fn ensure(&mut self, size: usize, alignment: usize) {
        unsafe {
            if size > self.layout.size() || alignment > self.layout.align() {
                let size = size.max(self.layout.size());
                let alignment = alignment.max(self.layout.align());
                if !self.buffer.is_null() {
                    std::alloc::dealloc(self.buffer, self.layout);
                }
                self.layout = Layout::from_size_align_unchecked(size, alignment);
                self.buffer = std::alloc::alloc(self.layout);
                assert!(!self.buffer.is_null());
            }
        }
    }
}

impl Drop for TempBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                std::alloc::dealloc(self.buffer, self.layout);
            }
        }
    }
}
