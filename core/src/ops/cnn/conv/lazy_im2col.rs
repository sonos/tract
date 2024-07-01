use crate::internal::*;
use std::fmt::{Debug, Display};
use std::ops::Range;
use tract_linalg::frame::{PackedFormat, PackingWriter};
use tract_linalg::mmm::MMMInputValue;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LazyIm2colParams {
    pub packer: PackedFormat,
    pub n_byte_offsets: Vec<isize>,
    pub k_byte_offsets: Vec<isize>,
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LazyIm2Col {
    pub params: Arc<LazyIm2colParams>,
}

impl Op for LazyIm2Col {
    fn name(&self) -> Cow<str> {
        "LazyIm2col".into()
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for LazyIm2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let tensor = args_1!(inputs);
        let input: Box<dyn MMMInputValue> =
            Box::new(LazyIm2colInput { tensor, im2col: self.params.clone() });
        let input = Opaque(Arc::new(input));
        Ok(tvec!(tensor2(&[[input]]).into_tvalue()))
    }
}

impl TypedOp for LazyIm2Col {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(Opaque::fact([1, 1])))
    }

    as_op!();
}

#[derive(Clone, Debug)]
struct LazyIm2colInput {
    tensor: TValue,
    im2col: Arc<LazyIm2colParams>,
}

impl Display for LazyIm2colInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Hash for LazyIm2colInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.tensor.as_bytes(), &self.im2col).hash(state);
    }
}

unsafe impl Send for LazyIm2colInput {}
unsafe impl Sync for LazyIm2colInput {}

impl LazyIm2colInput {
    fn input_8n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
        let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            let o7 = *n_byte_offsets.offset(n + 6);
            let o8 = *n_byte_offsets.offset(n + 7);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
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

    fn input_6n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
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

    fn input_4n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
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

    fn input_2n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                writer.write(v1);
                writer.write(v2);
            }
        }
    }

    fn write<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: std::ops::Range<isize>,
        mn_range: std::ops::Range<isize>,
    ) {
        let mn_end = mn_range.end.min(self.im2col.n_byte_offsets.len() as isize);
        let n_range = mn_range.start..mn_end;
        match n_range.len() {
            8 => return self.input_8n(writer, k_range, n_range.start),
            6 => return self.input_6n(writer, k_range, n_range.start),
            4 => return self.input_4n(writer, k_range, n_range.start),
            2 => return self.input_2n(writer, k_range, n_range.start),
            _ => (),
        }
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let mut n = n_range.start;
                while n + 8 <= n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
                    let o5 = *n_byte_offsets.offset(n + 4);
                    let o6 = *n_byte_offsets.offset(n + 5);
                    let o7 = *n_byte_offsets.offset(n + 6);
                    let o8 = *n_byte_offsets.offset(n + 7);
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
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
                    let o5 = *n_byte_offsets.offset(n + 4);
                    let o6 = *n_byte_offsets.offset(n + 5);
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
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
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
                    let o1 = *n_byte_offsets.offset(n);
                    let v1 = *(ptr.offset(o1) as *const T);
                    writer.write(v1);
                    n += 1;
                }
            }
        }
    }
}

impl MMMInputValue for LazyIm2colInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        let k = self.im2col.k_byte_offsets.len();
        Some(self.im2col.packer.single_panel_layout(k, self.tensor.datum_type().size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        Ok(dispatch_copy!(Self::do_panel(self.tensor.datum_type())(self, i, buffer)))
    }

    fn k(&self) -> usize {
        self.im2col.k_byte_offsets.len()
    }

    fn mn(&self) -> usize {
        self.im2col.n_byte_offsets.len()
    }

    fn r(&self) -> usize {
        self.im2col.packer.r
    }
}

impl LazyIm2colInput {
    fn do_panel<T: Datum + Copy>(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let r = self.im2col.packer.r;
        let mn_start = i * r;
        let mn_end = (mn_start + self.im2col.packer.r).min(self.im2col.n_byte_offsets.len());
        let k = self.im2col.k_byte_offsets.len();
        let mn_range = mn_start as isize..mn_end as isize;
        let k_range = 0..k as isize;
        let packed = buffer.unwrap();
        if mn_range.len() == r && mn_start % r == 0 {
            let mut writer = self.im2col.packer.write_single_panel_with_k_outer(packed as *mut T);
            self.write(&mut writer, k_range, mn_range);
        } else {
            let mut writer = self.im2col.packer.write_with_k_outer(
                packed as *mut T,
                k_range.len(),
                mn_range.len(),
            );
            self.write(&mut writer, k_range, mn_range);
        }
        packed
    }
}
