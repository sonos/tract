use std::alloc::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::anyhow;

use crate::LADatum;

macro_rules! ew_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize) -> ());
            }

            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl ElementWiseKer<$ti> for $func {
                #[inline(always)]
                fn name() -> &'static str {
                    stringify!($func)
                }
                #[inline(always)]
                fn nr() -> usize {
                    $nr
                }
                #[inline(always)]
                fn alignment_items() -> usize {
                    $alignment_items
                }
                #[inline(always)]
                fn alignment_bytes() -> usize {
                    $alignment_items * std::mem::size_of::<$ti>()
                }
                #[inline(never)]
                fn run(buf: &mut [$ti]) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len()) }
                }
            }
        }
    };
}

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

std::thread_local! {
    static TMP: std::cell::RefCell<TempBuffer> = std::cell::RefCell::new(TempBuffer::default());
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
    T: LADatum,
    K: ElementWiseKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> ElementWise<T> for ElementWiseImpl<K, T>
where
    T: LADatum,
    K: ElementWiseKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) -> anyhow::Result<()> {
        if vec.is_empty() {
            return Ok(());
        }
        unsafe {
            TMP.with(|buffer| {
                let mut buffer = buffer.borrow_mut();
                buffer.ensure(K::nr() * T::datum_type().size_of(), K::alignment_bytes());
                let tmp = std::slice::from_raw_parts_mut(buffer.buffer as *mut T, K::nr());
                let mut compute_via_temp_buffer = |slice: &mut [T]| {
                    tmp[..slice.len()].copy_from_slice(slice);
                    K::run(tmp);
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

pub trait ElementWiseKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T]);
    fn ew() -> Box<dyn ElementWise<T>> {
        Box::new(ElementWiseImpl::<Self, T>::new())
    }
}

#[cfg(test)]
pub mod test {
    use crate::{frame::element_wise::*, LADatum};
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    pub fn test_element_wise<K: ElementWiseKer<T>, T: LADatum, F: Fn(T) -> T>(
        values: &[T],
        reference: F,
    ) -> TestCaseResult {
        crate::setup_test_logger();
        let op = ElementWiseImpl::<K, T>::new();
        let mut values = values.to_vec();
        while values.len() < K::nr() {
            values.push(T::zero());
        }
        let expected = values.iter().copied().map(reference).collect::<Vec<_>>();
        let mut found = values;
        op.run(&mut found).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
