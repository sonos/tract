use std::alloc;
use std::mem;

pub unsafe fn alloc_bytes(size: usize, alignment: usize) -> *mut u8 {
    alloc::alloc(alloc::Layout::from_size_align(size, alignment).unwrap())
}

pub unsafe fn alloc_zeroed_bytes(size: usize, alignment: usize) -> *mut u8 {
    alloc::alloc_zeroed(alloc::Layout::from_size_align(size, alignment).unwrap())
}

pub unsafe fn vec_bytes(capacity: usize, alignment: usize) -> Vec<u8> {
    let aligned_buffer = alloc_bytes(capacity, alignment);
    Vec::from_raw_parts(aligned_buffer as _, 0, capacity)
}

pub unsafe fn uninitialized_bytes(size: usize, alignment: usize) -> Vec<u8> {
    let aligned_buffer = alloc_bytes(size, alignment);
    Vec::from_raw_parts(aligned_buffer as _, size, size)
}

pub unsafe fn uninitialized<T>(size: usize, alignment_bytes: usize) -> Vec<T> {
    let aligned_buffer = alloc_bytes(size * mem::size_of::<T>(), alignment_bytes);
    Vec::from_raw_parts(aligned_buffer as _, size, size)
}

pub unsafe fn zeroed<T>(size: usize, alignment_bytes: usize) -> Vec<T> {
    let aligned_buffer = alloc_zeroed_bytes(size * mem::size_of::<T>(), alignment_bytes);
    Vec::from_raw_parts(aligned_buffer as _, size, size)
}

/*
fn realign_slice_bytes(v: &[u8], alignment: usize) -> Vec<u8> {
    assert!(
        (alignment as u32).count_ones() == 1,
        "Invalid alignment required ({})",
        alignment
    );
    if v.len() == 0 {
        return vec![];
    }
    unsafe {
        let aligned_buffer = alloc_bytes(v.len(), alignment);
        let mut output = Vec::from_raw_parts(aligned_buffer as _, v.len(), v.len());
        output.copy_from_slice(v);
        output
    }
}
*/

pub fn realign_slice<T: Copy>(v: &[T], alignment: usize) -> Vec<T> {
    if v.len() == 0 {
        return vec![];
    }
    unsafe {
        let t = mem::size_of::<T>();
        let aligned = alloc_bytes(v.len() * t, alignment);
        let mut result = Vec::from_raw_parts(aligned as _, v.len(), v.len());
        result.copy_from_slice(&v);
        result
    }
}

pub fn realign_vec<T: Copy>(v: Vec<T>, alignment: usize) -> Vec<T> {
    if v.len() == 0 || v.as_ptr() as usize % alignment == 0 {
        return v;
    }
    realign_slice(&v, alignment)
}
