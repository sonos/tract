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
            n_item_offsets: self.n_item_offsets.clone(),
            k_item_offsets: self.k_item_offsets.clone(),
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
    n_item_offsets: Vec<isize>,
    k_item_offsets: Vec<isize>,
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
        let mn_end = mn_range.end.min(self.n_item_offsets.len());
        let n_range = mn_range.start..mn_end;
        unsafe {
            let mut writer =
                packer.write_with_k_outer(packed as *mut T, k_range.len(), n_range.len());
            for k in k_range.start..k_range.end {
                let ptr = self.ptr.offset(*self.k_item_offsets.get_unchecked(k));
                for n in n_range.start..n_range.end {
                    writer.write(*ptr.offset(*self.n_item_offsets.get_unchecked(n)))
                }
            }
        }
    }
}
