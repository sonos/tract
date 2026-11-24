use crate::internal::*;
use crate::ops::matmul::pack::DynPackedExoticFact;
use std::fmt::{Debug, Display};
use std::ops::Range;
use tract_linalg::WeightType;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue, PackedMatrixStorage};
use tract_linalg::pack::{PackedFormat, PackingWriter};

/// The gather plan of a lazy im2col: value `(k, n)` is read at
/// `k_byte_offsets[k] + n_byte_offsets[n]` from the convolution input.
///
/// A padded convolution reads its unpadded input directly. `axis_tap_masks[ax][o]` then holds,
/// bit `t` set, the kernel taps whose coordinate along spatial axis `ax` lands inside the input
/// for output coordinate `o`, the output points `n` running row-major over those axes. A tap is
/// inside for a point when it is inside along every axis, and row `k` is tap `k % taps`. A tap
/// outside reads as zero and its offset is never dereferenced. An empty `axis_tap_masks` means
/// every tap of every point is inside, as for an unpadded input.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LazyIm2colParams {
    pub packer: PackedFormat,
    pub n_byte_offsets: Vec<isize>,
    pub k_byte_offsets: Vec<isize>,
    pub taps: usize,
    pub axis_tap_masks: Vec<Vec<u64>>,
    /// Per spatial axis, the output coordinates whose every tap is inside along that axis.
    pub axis_interior: Vec<Range<usize>>,
}

impl LazyIm2colParams {
    /// Whether every tap of every point of `n_range` is inside the input, answered from the
    /// first point's coordinates when the range stays within one run of the last axis.
    fn interior(&self, n_range: Range<usize>) -> bool {
        let Some((last, outer)) = self.axis_tap_masks.split_last() else { return true };
        let mut rest = n_range.start / last.len();
        let first = n_range.start % last.len();
        for (axis, inside) in outer.iter().zip(&self.axis_interior).rev() {
            if !inside.contains(&(rest % axis.len())) {
                return false;
            }
            rest /= axis.len();
        }
        let inside = self.axis_interior.last().unwrap();
        inside.start <= first && first + n_range.len() <= inside.end
    }
}

/// The tap masks of output points `n_range`, one per point into `masks`, from the per-axis
/// masks of [`LazyIm2colParams::axis_tap_masks`]; whether one of the points misses a tap.
pub(crate) fn point_tap_masks(
    axis_tap_masks: &[Vec<u64>],
    taps: usize,
    n_range: Range<usize>,
    masks: &mut [u64],
) -> bool {
    let all = u64::MAX >> (64 - taps);
    let mut coords = [0usize; 8];
    let mut rest = n_range.start;
    for (ax, axis) in axis_tap_masks.iter().enumerate().rev() {
        coords[ax] = rest % axis.len();
        rest /= axis.len();
    }
    let mut border = false;
    for mask in masks.iter_mut().take(n_range.len()) {
        *mask = axis_tap_masks.iter().zip(&coords).fold(all, |m, (axis, &c)| m & axis[c]);
        border |= *mask != all;
        for (ax, axis) in axis_tap_masks.iter().enumerate().rev() {
            coords[ax] += 1;
            if coords[ax] < axis.len() {
                break;
            }
            coords[ax] = 0;
        }
    }
    border
}

impl MMMInputFormat for LazyIm2colParams {
    fn r(&self) -> usize {
        self.packer.r
    }

    fn precursor(&self) -> WeightType {
        self.packer.precursor()
    }

    fn prepare_tensor(&self, _t: &Tensor, _k_axis: usize, _mn_axis: usize) -> TractResult<Tensor> {
        bail!("Unexpected call to prepare_tensor on LazyIm2Col")
    }

    fn k_alignment(&self) -> usize {
        1
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        k * mn * self.packer.dt.size_of()
    }

    fn extract_at_mn_f16(
        &self,
        _data: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _slice: &mut [f16],
    ) -> TractResult<()> {
        unimplemented!()
    }
    fn extract_at_mn_f32(
        &self,
        _data: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _slice: &mut [f32],
    ) -> TractResult<()> {
        unimplemented!()
    }

    fn prepare_one_view(
        &self,
        _t: &TensorView,
        _k_axis: usize,
        _mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        bail!("Unexpected call to prepare_one_view on LazyIm2Col")
    }

    fn prepare_one(
        &self,
        _t: &Tensor,
        _k_axis: usize,
        _mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        bail!("Unexpected call to prepare_one on LazyIm2Col")
    }
}

impl Display for LazyIm2colParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LazyIm2Col")
    }
}

impl ExoticFact for LazyIm2colParams {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(MMMInputFormat::mem_size(
            self,
            self.k_byte_offsets.len().to_dim(),
            self.n_byte_offsets.len().to_dim(),
        ))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LazyIm2Col {
    pub params: Arc<LazyIm2colParams>,
    /// Number of groups for grouped convolution (1 if ungrouped). The output is then a
    /// `[1, group]`-batched packed tensor; each group reads from a different slice of the
    /// input via a per-group offset of `g * ci_per_group * c_stride * size_of`.
    pub group: usize,
    /// Byte stride between consecutive groups in the input tensor's flat byte buffer.
    /// Equal to `ci_per_group * c_stride * size_of(input)`. Unused (0) when `group == 1`.
    pub group_stride_bytes: isize,
}

impl Op for LazyIm2Col {
    fn name(&self) -> StaticName {
        "LazyIm2col".into()
    }

    op_as_typed_op!();
}

impl EvalOp for LazyIm2Col {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let tensor = args_1!(inputs);
        let dt = tensor.datum_type();
        let mut values: TVec<Box<dyn MMMInputValue>> = TVec::with_capacity(self.group);
        for g in 0..self.group {
            let group_offset_bytes = g as isize * self.group_stride_bytes;
            values.push(Box::new(LazyIm2colInput {
                tensor: tensor.clone(),
                im2col: self.params.clone(),
                group_offset_bytes,
            }));
        }
        let output = PackedMatrixStorage::new_batched(&[1, self.group], values).into_tensor(dt);
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for LazyIm2Col {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let exotic_fact = DynPackedExoticFact {
            k: self.params.k_byte_offsets.len().to_dim(),
            mn: self.params.n_byte_offsets.len().to_dim(),
            packers: vec![Box::new(self.params.packer.clone()) as Box<dyn MMMInputFormat>],
        };
        Ok(tvec!(inputs[0].datum_type.fact([1, self.group]).with_exotic_fact(exotic_fact)))
    }

    as_op!();
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LazyIm2colInput {
    tensor: TValue,
    im2col: Arc<LazyIm2colParams>,
    /// Per-group base byte offset added to every gather. For group g this is
    /// `g * ci_per_group * c_stride * size_of(input)`. Zero for ungrouped convs.
    group_offset_bytes: isize,
}

impl Display for LazyIm2colInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Hash for LazyIm2colInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.tensor.as_bytes(), &self.im2col, self.group_offset_bytes).hash(state);
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
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            let o7 = *n_byte_offsets.offset(n + 6);
            let o8 = *n_byte_offsets.offset(n + 7);
            for k in k_range.start..k_range.end {
                let ptr = ptr.wrapping_offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.wrapping_offset(o1) as *const T);
                let v2 = *(ptr.wrapping_offset(o2) as *const T);
                let v3 = *(ptr.wrapping_offset(o3) as *const T);
                let v4 = *(ptr.wrapping_offset(o4) as *const T);
                let v5 = *(ptr.wrapping_offset(o5) as *const T);
                let v6 = *(ptr.wrapping_offset(o6) as *const T);
                let v7 = *(ptr.wrapping_offset(o7) as *const T);
                let v8 = *(ptr.wrapping_offset(o8) as *const T);
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
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            for k in k_range.start..k_range.end {
                let ptr = ptr.wrapping_offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.wrapping_offset(o1) as *const T);
                let v2 = *(ptr.wrapping_offset(o2) as *const T);
                let v3 = *(ptr.wrapping_offset(o3) as *const T);
                let v4 = *(ptr.wrapping_offset(o4) as *const T);
                let v5 = *(ptr.wrapping_offset(o5) as *const T);
                let v6 = *(ptr.wrapping_offset(o6) as *const T);
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
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            for k in k_range.start..k_range.end {
                let ptr = ptr.wrapping_offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.wrapping_offset(o1) as *const T);
                let v2 = *(ptr.wrapping_offset(o2) as *const T);
                let v3 = *(ptr.wrapping_offset(o3) as *const T);
                let v4 = *(ptr.wrapping_offset(o4) as *const T);
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
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            for k in k_range.start..k_range.end {
                let ptr = ptr.wrapping_offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.wrapping_offset(o1) as *const T);
                let v2 = *(ptr.wrapping_offset(o2) as *const T);
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
        if !self.im2col.axis_tap_masks.is_empty()
            && !self.im2col.interior(n_range.start as usize..n_range.end as usize)
        {
            let mut masks = [0u64; 64];
            let masks = &mut masks[..n_range.len()];
            let points = n_range.start as usize..n_range.end as usize;
            if point_tap_masks(&self.im2col.axis_tap_masks, self.im2col.taps, points, masks) {
                return self.input_checked(writer, k_range, n_range, masks);
            }
        }
        match n_range.len() {
            8 => return self.input_8n(writer, k_range, n_range.start),
            6 => return self.input_6n(writer, k_range, n_range.start),
            4 => return self.input_4n(writer, k_range, n_range.start),
            2 => return self.input_2n(writer, k_range, n_range.start),
            _ => (),
        }
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            for k in k_range.start..k_range.end {
                let ptr = ptr.wrapping_offset(*k_byte_offsets.offset(k));
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
                    let v1 = *(ptr.wrapping_offset(o1) as *const T);
                    let v2 = *(ptr.wrapping_offset(o2) as *const T);
                    let v3 = *(ptr.wrapping_offset(o3) as *const T);
                    let v4 = *(ptr.wrapping_offset(o4) as *const T);
                    let v5 = *(ptr.wrapping_offset(o5) as *const T);
                    let v6 = *(ptr.wrapping_offset(o6) as *const T);
                    let v7 = *(ptr.wrapping_offset(o7) as *const T);
                    let v8 = *(ptr.wrapping_offset(o8) as *const T);
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
                    let v1 = *(ptr.wrapping_offset(o1) as *const T);
                    let v2 = *(ptr.wrapping_offset(o2) as *const T);
                    let v3 = *(ptr.wrapping_offset(o3) as *const T);
                    let v4 = *(ptr.wrapping_offset(o4) as *const T);
                    let v5 = *(ptr.wrapping_offset(o5) as *const T);
                    let v6 = *(ptr.wrapping_offset(o6) as *const T);
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
                    let v1 = *(ptr.wrapping_offset(o1) as *const T);
                    let v2 = *(ptr.wrapping_offset(o2) as *const T);
                    let v3 = *(ptr.wrapping_offset(o3) as *const T);
                    let v4 = *(ptr.wrapping_offset(o4) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    n += 4;
                }
                while n < n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let v1 = *(ptr.wrapping_offset(o1) as *const T);
                    writer.write(v1);
                    n += 1;
                }
            }
        }
    }
}

impl LazyIm2colInput {
    /// The gather of a border panel: a tap outside the input writes zero. Most taps of a border
    /// panel are inside the input for all of its points, so each tap's points are summed up
    /// once and a fully inside tap row gathers without a per-element check.
    fn input_checked<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n_range: Range<isize>,
        masks: &[u64],
    ) {
        let taps = self.im2col.taps;
        let n_offsets = &self.im2col.n_byte_offsets[n_range.start as usize..n_range.end as usize];
        let mut tap_inside = [true; 64];
        for (t, inside) in tap_inside.iter_mut().enumerate().take(taps) {
            *inside = masks.iter().all(|m| m & (1 << t) != 0);
        }
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>().offset(self.group_offset_bytes);
            for k in k_range {
                let t = k as usize % taps;
                let ptr =
                    ptr.wrapping_offset(*self.im2col.k_byte_offsets.get_unchecked(k as usize));
                if tap_inside[t] {
                    let (chunks, rest) = n_offsets.as_chunks::<8>();
                    for chunk in chunks {
                        let v: [T; 8] =
                            std::array::from_fn(|i| *(ptr.wrapping_offset(chunk[i]) as *const T));
                        for v in v {
                            writer.write(v);
                        }
                    }
                    for offset in rest {
                        writer.write(*(ptr.wrapping_offset(*offset) as *const T));
                    }
                } else {
                    // An outside lane reads the zero below instead of its own address, so the
                    // row keeps the branchless gather.
                    let bit = 1u64 << t;
                    let zero = T::default();
                    let zero = &zero as *const T;
                    let at = |i: usize| {
                        if masks[i] & bit != 0 {
                            ptr.wrapping_offset(n_offsets[i]) as *const T
                        } else {
                            zero
                        }
                    };
                    let mut i = 0;
                    while i + 8 <= n_offsets.len() {
                        let v: [T; 8] = std::array::from_fn(|j| *at(i + j));
                        for v in v {
                            writer.write(v);
                        }
                        i += 8;
                    }
                    while i < n_offsets.len() {
                        writer.write(*at(i));
                        i += 1;
                    }
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

    fn format(&self) -> &dyn MMMInputFormat {
        &*self.im2col
    }

    fn exotic_fact(&self) -> &dyn ExoticFact {
        &*self.im2col
    }
    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        unimplemented!()
    }
    fn extract_at_mn_f32(&self, _mn: usize, _slice: &mut [f32]) -> TractResult<()> {
        unimplemented!()
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
        if mn_range.len() == r && mn_start.is_multiple_of(r) {
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
