use std::{alloc, fmt, mem, ops, slice};

pub struct Buffer<T> {
    ptr: *mut T,
    items: usize,
    layout: alloc::Layout,
}

impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", unsafe { slice::from_raw_parts(self.ptr, self.items) })
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.ptr as _, self.layout) };
    }
}

impl<T> ops::Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.items) }
    }
}

impl<T> ops::DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.items) }
    }
}

impl<T> Buffer<T> {
    pub fn uninitialized(items: usize, alignment_bytes: usize) -> Buffer<T> {
        let layout =
            alloc::Layout::from_size_align(items * mem::size_of::<T>(), alignment_bytes).unwrap();
        let ptr = unsafe { alloc::alloc(layout) } as *mut T;
        Buffer { ptr, items, layout }
    }
}

impl<T: Copy> Buffer<T> {
    pub fn realign_data(data: &[T], alignment_bytes: usize) -> Buffer<T> {
        let mut buf = Buffer::uninitialized(data.len(), alignment_bytes);
        buf.copy_from_slice(data);
        buf
    }
}
