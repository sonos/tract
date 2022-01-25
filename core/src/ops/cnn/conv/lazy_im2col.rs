use crate::internal::*;
use tract_linalg::mmm::{VirtualInput, VirtualInputSpec};

#[derive(Clone, Debug, Hash)]
pub struct LazyIm2colSpec {
    pub n_item_offsets: Vec<isize>,
    pub k_item_offsets: Vec<isize>,
}

impl_dyn_hash!(LazyIm2colSpec);

impl LazyIm2colSpec {
    fn wrap_t<T: Datum + Copy>(&self, view: &TensorView) -> Box<dyn VirtualInput> {
        let input = LazyIm2col::<T> {
            ptr: view.as_ptr().unwrap(),
            n_byte_offsets: self
                .n_item_offsets
                .iter()
                .map(|&x| x * T::datum_type().size_of() as isize)
                .collect(),
            k_byte_offsets: self
                .k_item_offsets
                .iter()
                .map(|&x| x * T::datum_type().size_of() as isize)
                .collect(),
        };
        Box::new(input)
    }
}

impl VirtualInputSpec for LazyIm2colSpec {
    fn wrap(&self, view: &TensorView) -> Box<dyn VirtualInput> {
        assert_eq!(view.datum_type(), f32::datum_type());
        dispatch_copy!(Self::wrap_t(view.datum_type())(self, view))
    }
}

#[derive(Clone, Debug)]
struct LazyIm2col<T: Datum + Copy> {
    ptr: *const T,
    n_byte_offsets: Vec<isize>,
    k_byte_offsets: Vec<isize>,
}

unsafe impl<T: Datum + Copy> Send for LazyIm2col<T> {}
unsafe impl<T: Datum + Copy> Sync for LazyIm2col<T> {}

impl<T: Datum + Copy> VirtualInput for LazyIm2col<T> {
    fn input(
        &self,
        packer: &tract_linalg::frame::Packer,
        packed: *mut u8,
        k_range: std::ops::Range<usize>,
        mn_range: std::ops::Range<usize>,
    ) {
        let mn_end = mn_range.end.min(self.n_byte_offsets.len());
        let n_range = mn_range.start..mn_end;
        unsafe {
            let mut writer =
                packer.write_with_k_outer(packed as *mut T, k_range.len(), n_range.len());
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.get_unchecked(k));
                let mut n = n_range.start;
                while n + 8 <= n_range.end {
                    let o1 = *self.n_byte_offsets.get_unchecked(n);
                    let o2 = *self.n_byte_offsets.get_unchecked(n + 1);
                    let o3 = *self.n_byte_offsets.get_unchecked(n + 2);
                    let o4 = *self.n_byte_offsets.get_unchecked(n + 3);
                    let o5 = *self.n_byte_offsets.get_unchecked(n + 4);
                    let o6 = *self.n_byte_offsets.get_unchecked(n + 5);
                    let o7 = *self.n_byte_offsets.get_unchecked(n + 6);
                    let o8 = *self.n_byte_offsets.get_unchecked(n + 7);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    let v5 = *(ptr.offset(o5) as *const T);
                    let v6 = *(ptr.offset(o6) as *const T);
                    let v7 = *(ptr.offset(o7) as *const T);
                    let v8 = *(ptr.offset(o8) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    writer.write(v5);
                    writer.write(v6);
                    writer.write(v7);
                    writer.write(v8);
                    n += 8;
                }
                while n + 6 <= n_range.end {
                    let o1 = *self.n_byte_offsets.get_unchecked(n);
                    let o2 = *self.n_byte_offsets.get_unchecked(n + 1);
                    let o3 = *self.n_byte_offsets.get_unchecked(n + 2);
                    let o4 = *self.n_byte_offsets.get_unchecked(n + 3);
                    let o5 = *self.n_byte_offsets.get_unchecked(n + 4);
                    let o6 = *self.n_byte_offsets.get_unchecked(n + 5);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    let v5 = *(ptr.offset(o5) as *const T);
                    let v6 = *(ptr.offset(o6) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    writer.write(v5);
                    writer.write(v6);
                    n += 6;
                }
                while n + 4 <= n_range.end {
                    let o1 = *self.n_byte_offsets.get_unchecked(n);
                    let o2 = *self.n_byte_offsets.get_unchecked(n + 1);
                    let o3 = *self.n_byte_offsets.get_unchecked(n + 2);
                    let o4 = *self.n_byte_offsets.get_unchecked(n + 3);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    n += 4;
                }
                while n < n_range.end {
                    let o1 = *self.n_byte_offsets.get_unchecked(n);
                    let v1 = *(ptr.offset(o1) as *const T);
                    writer.write(v1);
                    n += 1;
                }
            }
        }
    }
}
