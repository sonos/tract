use crate::internal::*;
use std::fmt::Debug;
use std::ops::Range;
use tract_linalg::frame::PackingWriter;
use tract_linalg::mmm::{VirtualInput, VirtualInputSpec};

#[derive(Clone, Hash)]
pub struct LazyIm2colSpec {
    pub n_bytes_offsets: Vec<isize>,
    pub k_bytes_offsets: Vec<isize>,
}

impl Debug for LazyIm2colSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LazyIm2colSpec {{...}}")
    }
}

impl LazyIm2colSpec {
    fn wrap_t<T: Datum + Copy>(&self, view: &TensorView) -> Box<dyn VirtualInput> {
        let input = LazyIm2col::<T> {
            ptr: view.as_ptr().unwrap(),
            n: self.n_bytes_offsets.len(),
            n_byte_offsets: self.n_bytes_offsets.as_ptr(),
            k_byte_offsets: self.k_bytes_offsets.as_ptr(),
        };
        Box::new(input)
    }
}

impl VirtualInputSpec for LazyIm2colSpec {
    fn wrap(&self, view: &TensorView) -> Box<dyn VirtualInput> {
        dispatch_copy!(Self::wrap_t(view.datum_type())(self, view))
    }
}

#[derive(Clone, Debug)]
struct LazyIm2col<T: Datum + Copy> {
    ptr: *const T,
    n: usize,
    n_byte_offsets: *const isize,
    k_byte_offsets: *const isize,
}

unsafe impl<T: Datum + Copy> Send for LazyIm2col<T> {}
unsafe impl<T: Datum + Copy> Sync for LazyIm2col<T> {}

impl<T: Datum + Copy> LazyIm2col<T> {
    fn input_8n(&self, writer: &mut impl PackingWriter<T>, k_range: Range<isize>, n: isize) {
        unsafe {
            let o1 = *self.n_byte_offsets.offset(n);
            let o2 = *self.n_byte_offsets.offset(n + 1);
            let o3 = *self.n_byte_offsets.offset(n + 2);
            let o4 = *self.n_byte_offsets.offset(n + 3);
            let o5 = *self.n_byte_offsets.offset(n + 4);
            let o6 = *self.n_byte_offsets.offset(n + 5);
            let o7 = *self.n_byte_offsets.offset(n + 6);
            let o8 = *self.n_byte_offsets.offset(n + 7);
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.offset(k));
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
            }
        }
    }

    fn input_6n(&self, writer: &mut impl PackingWriter<T>, k_range: Range<isize>, n: isize) {
        unsafe {
            let o1 = *self.n_byte_offsets.offset(n);
            let o2 = *self.n_byte_offsets.offset(n + 1);
            let o3 = *self.n_byte_offsets.offset(n + 2);
            let o4 = *self.n_byte_offsets.offset(n + 3);
            let o5 = *self.n_byte_offsets.offset(n + 4);
            let o6 = *self.n_byte_offsets.offset(n + 5);
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.offset(k));
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
            }
        }
    }

    fn input_4n(&self, writer: &mut impl PackingWriter<T>, k_range: Range<isize>, n: isize) {
        unsafe {
            let o1 = *self.n_byte_offsets.offset(n);
            let o2 = *self.n_byte_offsets.offset(n + 1);
            let o3 = *self.n_byte_offsets.offset(n + 2);
            let o4 = *self.n_byte_offsets.offset(n + 3);
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                let v3 = *(ptr.offset(o3) as *const T);
                let v4 = *(ptr.offset(o4) as *const T);
                writer.write(v1);
                writer.write(v2);
                writer.write(v3);
                writer.write(v4);
            }
        }
    }

    fn input_2n(&self, writer: &mut impl PackingWriter<T>, k_range: Range<isize>, n: isize) {
        unsafe {
            let o1 = *self.n_byte_offsets.offset(n);
            let o2 = *self.n_byte_offsets.offset(n + 1);
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                writer.write(v1);
                writer.write(v2);
            }
        }
    }

    fn write(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: std::ops::Range<isize>,
        mn_range: std::ops::Range<isize>,
    ) {
        let mn_end = mn_range.end.min(self.n as isize);
        let n_range = mn_range.start..mn_end;
        match n_range.len() {
            8 => return self.input_8n(writer, k_range, n_range.start),
            6 => return self.input_6n(writer, k_range, n_range.start),
            4 => return self.input_4n(writer, k_range, n_range.start),
            2 => return self.input_2n(writer, k_range, n_range.start),
            _ => (),
        }
        unsafe {
            for k in k_range.start..k_range.end {
                let ptr = (self.ptr as *const u8).offset(*self.k_byte_offsets.offset(k));
                let mut n = n_range.start;
                while n + 8 <= n_range.end {
                    let o1 = *self.n_byte_offsets.offset(n);
                    let o2 = *self.n_byte_offsets.offset(n + 1);
                    let o3 = *self.n_byte_offsets.offset(n + 2);
                    let o4 = *self.n_byte_offsets.offset(n + 3);
                    let o5 = *self.n_byte_offsets.offset(n + 4);
                    let o6 = *self.n_byte_offsets.offset(n + 5);
                    let o7 = *self.n_byte_offsets.offset(n + 6);
                    let o8 = *self.n_byte_offsets.offset(n + 7);
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
                    let o1 = *self.n_byte_offsets.offset(n);
                    let o2 = *self.n_byte_offsets.offset(n + 1);
                    let o3 = *self.n_byte_offsets.offset(n + 2);
                    let o4 = *self.n_byte_offsets.offset(n + 3);
                    let o5 = *self.n_byte_offsets.offset(n + 4);
                    let o6 = *self.n_byte_offsets.offset(n + 5);
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
                    let o1 = *self.n_byte_offsets.offset(n);
                    let o2 = *self.n_byte_offsets.offset(n + 1);
                    let o3 = *self.n_byte_offsets.offset(n + 2);
                    let o4 = *self.n_byte_offsets.offset(n + 3);
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
                    let o1 = *self.n_byte_offsets.offset(n);
                    let v1 = *(ptr.offset(o1) as *const T);
                    writer.write(v1);
                    n += 1;
                }
            }
        }
    }
}

impl<T: Datum + Copy> VirtualInput for LazyIm2col<T> {
    fn input(
        &self,
        packer: &tract_linalg::frame::Packer,
        packed: *mut u8,
        k_range: std::ops::Range<usize>,
        mn_range: std::ops::Range<usize>,
    ) {
        let mn_end = mn_range.end.min(self.n) as isize;
        let n_range = mn_range.start as isize..mn_end;
        if n_range.len() == packer.r && mn_range.start % packer.r == 0 {
            let mut writer = packer.write_single_panel_with_k_outer(packed as *mut T);
            self.write(
                &mut writer,
                k_range.start as isize..k_range.end as isize,
                mn_range.start as isize..n_range.end,
            )
        } else {
            let mut writer =
                packer.write_with_k_outer(packed as *mut T, k_range.len(), n_range.len());
            self.write(
                &mut writer,
                k_range.start as isize..k_range.end as isize,
                mn_range.start as isize..n_range.end,
            )
        }
    }
}
