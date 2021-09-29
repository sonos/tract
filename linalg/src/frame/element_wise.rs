use std::alloc::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::anyhow;

struct TempBuffer {
    layout: Layout,
    buffer: *mut u8,
}

impl Default for TempBuffer {
    fn default() -> Self {
        TempBuffer { layout: Layout::new::<()>(), buffer: std::ptr::null_mut() }
    }
}

impl TempBuffer {
    fn ensure(&mut self, size: usize, alignment: usize) {
        unsafe {
            if size > self.layout.size() || alignment > self.layout.align() {
                let size = size.max(self.layout.size());
                let alignment = alignment.max(self.layout.align());
                if !self.buffer.is_null() {
                    std::alloc::dealloc(self.buffer, self.layout);
                }
                self.layout = Layout::from_size_align_unchecked(size, alignment);
                self.buffer = std::alloc::alloc(self.layout);
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

std::thread_local! {
    static TMP:  std::cell::RefCell<TempBuffer> = std::cell::RefCell::new(TempBuffer::default());
}

pub trait ElementWise<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn run(&self, vec: &mut [T]) -> anyhow::Result<()>;
}

dyn_clone::clone_trait_object!(<T> ElementWise<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct ElementWiseImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync,
    K: ElementWiseKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> ElementWise<T> for ElementWiseImpl<K, T>
where
    T: crate::Datum + Copy + Debug + PartialEq + Send + Sync,
    K: ElementWiseKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) -> anyhow::Result<()> {
        if vec.len() == 0 {
            return Ok(());
        }
        unsafe {
            TMP.with(|buffer| {
                let mut buffer = buffer.borrow_mut();
                buffer.ensure(K::nr() * T::datum_type().size_of(), K::alignment_bytes());
                let mut tmp = std::slice::from_raw_parts_mut(buffer.buffer as *mut T, K::nr());
                let mut compute_via_temp_buffer = |slice: &mut [T]| {
                    tmp[..slice.len()].copy_from_slice(slice);
                    K::run(&mut tmp);
                    slice.copy_from_slice(&tmp[..slice.len()])
                };
                let prefix_len = vec.as_ptr().align_offset(K::alignment_bytes()).min(vec.len());
                if prefix_len > 0 {
                    compute_via_temp_buffer(&mut vec[..prefix_len]);
                }
                let aligned_len = (vec.len() - prefix_len) / K::nr() * K::nr();
                if aligned_len > 0 {
                    K::run(&mut vec[prefix_len..][..aligned_len]);
                }
                if prefix_len + aligned_len < vec.len() {
                    compute_via_temp_buffer(&mut vec[prefix_len + aligned_len..]);
                }
            })
        }
        Ok(())
    }
}

pub trait ElementWiseKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T]);
}
