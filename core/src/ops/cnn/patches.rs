use crate::internal::*;
use crate::ops::cnn::PaddingSpec;
use crate::ops::nn::{DataFormat, DataShape};
use ndarray::prelude::*;

use super::PatchAxis;

use std::fmt::Debug;
use std::ops::Range;

use tract_itertools::{izip, Itertools};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PatchSpec {
    pub input_shape: TVec<usize>,
    pub input_inner_stride: usize,
    pub output_inner_stride: usize,
    pub kernel_shape: TVec<usize>,
    pub strides: TVec<usize>,
    pub dilations: TVec<usize>,
    pub padding: PaddingSpec,
}

impl Debug for PatchSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input: {} kernel: {} strides: {} dil: {} pad: {:?}",
            self.input_shape.iter().join(","),
            self.kernel_shape.iter().join(","),
            self.strides.iter().join(","),
            self.dilations.iter().join(","),
            self.padding
        )
    }
}

impl PatchSpec {
    pub fn for_full_shape(
        data_format: DataFormat,
        input_full_shape: &[usize],
    ) -> TractResult<PatchSpec> {
        let shape = data_format.shape(input_full_shape.into())?;
        Ok(Self::for_data_shape(shape))
    }

    pub fn for_data_shape(data_shape: DataShape) -> PatchSpec {
        let input_shape: TVec<usize> = data_shape.hw_dims().into();
        PatchSpec {
            kernel_shape: tvec!(1; input_shape.len()),
            input_inner_stride: *data_shape.w_stride(),
            output_inner_stride: 1,
            strides: tvec!(1; input_shape.len()),
            dilations: tvec!(1; input_shape.len()),
            padding: PaddingSpec::Valid,
            input_shape,
        }
    }

    pub fn with_kernel_shape(self, kernel_shape: TVec<usize>) -> PatchSpec {
        PatchSpec { kernel_shape, ..self }
    }

    pub fn with_dilations(self, dilations: TVec<usize>) -> PatchSpec {
        PatchSpec { dilations, ..self }
    }

    pub fn with_strides(self, strides: TVec<usize>) -> PatchSpec {
        PatchSpec { strides, ..self }
    }

    pub fn with_padding(self, padding: PaddingSpec) -> PatchSpec {
        PatchSpec { padding, ..self }
    }

    pub fn with_output_inner_stride(self, output_inner_stride: usize) -> PatchSpec {
        PatchSpec { output_inner_stride, ..self }
    }

    pub fn into_patch(self) -> Patch {
        let dims = self.padding.compute(
            &self.input_shape,
            &self.kernel_shape,
            &self.dilations,
            &self.strides,
        );
        let output: TVec<usize> = dims.iter().map(|d| d.convoluted).collect();
        let pad_before: TVec<usize> = dims.iter().map(|d| d.pad_before).collect();
        let pad_after: TVec<usize> = dims.iter().map(|d| d.pad_after).collect();

        let data_field: Vec<isize> = ::ndarray::indices(&*self.kernel_shape)
            .into_iter()
            .flat_map(|coords| {
                #[allow(clippy::unnecessary_to_owned)] // I think this one is a clippy bug.
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * self.dilations[ix]) as isize - pad_before[ix] as isize)
            })
            .collect();
        let data_field = Array2::from_shape_vec(
            (self.kernel_shape.iter().cloned().product(), self.kernel_shape.len()),
            data_field,
        )
        .unwrap();
        let data_field_min_max: TVec<_> = data_field
            .columns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        fn strides(shape: &[usize], inner: usize) -> TVec<isize> {
            let mut strides: TVec<isize> = tvec![inner as isize];
            for dim in shape.iter().skip(1).rev() {
                let previous = *strides.last().unwrap();
                strides.push(*dim as isize * previous);
            }
            strides.reverse();
            strides
        }

        let input_storage_strides = strides(&self.input_shape, self.input_inner_stride);
        let output_storage_strides = strides(&output, self.output_inner_stride);

        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| izip!(coords, &input_storage_strides).map(|(a, b)| a * b).sum::<isize>())
            .collect();

        // regions[axis][range+mask]
        let regions: TVec<TVec<_>> = dims
            .iter()
            .enumerate()
            .map(|(ix, d)| {
                PatchAxis {
                    input_dim: self.input_shape[ix],
                    kernel_dim: self.kernel_shape[ix],
                    pad_before: d.pad_before,
                    pad_after: d.pad_after,
                    output_dim: d.convoluted,
                    stride: self.strides[ix],
                    dilation: self.dilations[ix],
                }
                .regions()
            })
            .collect::<TVec<_>>();

        let zone_strides = strides(&regions.iter().map(|d| d.len()).collect::<TVec<_>>(), 1);

        let zones: Vec<Zone> = regions
            .iter()
            .multi_cartesian_product()
            .map(|regions| Zone {
                input_zone_offset: 0,
                output_ranges: regions.iter().map(|reg| reg.range.clone()).collect(),
                output_shape: regions.iter().map(|reg| reg.range.end - reg.range.start).collect(),
                output_zone_offset: izip!(&regions, &output_storage_strides)
                    .map(|(reg, &stride)| reg.range.start as isize * stride)
                    .sum::<isize>(),
                valid: regions.iter().all(|reg| reg.mask.is_none()),
                values_offsets: izip!(
                    0..,
                    ndarray::indices(&*self.kernel_shape),
                    &standard_layout_data_field
                )
                .filter(|(_ix, coords, _offset)| {
                    izip!(coords.slice(), &regions)
                        .all(|(&x, axis)| !axis.mask.as_ref().map(|mask| mask[x]).unwrap_or(false))
                })
                .map(|(ix, _coords, &window_offset)| (ix, window_offset))
                .collect(),
            })
            .collect();

        let valid_zone = zones.iter().position(|z| z.valid);

        let mut valid_output_zone = tvec!();
        let mut invalid_output_zones = tvec!();
        for ix in 0..self.input_shape.len() {
            let min_max = data_field_min_max[ix];
            let min = (-min_max.0 as usize).divceil(self.strides[ix]);
            let max =
                (self.input_shape[ix].saturating_sub(min_max.1 as usize)).divceil(self.strides[ix]);
            if min != 0 {
                let mut invalid = valid_output_zone.clone();
                invalid.push(0..min);
                while invalid.len() < output.len() {
                    invalid.push(0..output[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            if max < output[ix] {
                let mut invalid = valid_output_zone.clone();
                invalid.push(max..output[ix]);
                while invalid.len() < output.len() {
                    invalid.push(0..output[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            valid_output_zone.push(min..max)
        }

        let op_strides_times_input_storage_strides =
            izip!(&self.strides, &input_storage_strides).map(|(a, b)| (*a as isize * b)).collect();

        Patch {
            spec: self,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            output_shape: output,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            input_storage_strides,
            output_storage_strides,
            op_strides_times_input_storage_strides,
            valid_output_zone,
            invalid_output_zones,
            zones,
            valid_zone_id: valid_zone,
            zone_strides,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Patch {
    pub spec: PatchSpec,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    pub output_shape: TVec<usize>,
    pub data_field: Array2<isize>,
    pub data_field_min_max: TVec<(isize, isize)>,
    pub standard_layout_data_field: Vec<isize>,
    pub op_strides_times_input_storage_strides: TVec<isize>,
    pub valid_output_zone: TVec<Range<usize>>,
    pub invalid_output_zones: TVec<TVec<Range<usize>>>,
    pub zones: Vec<Zone>,
    pub valid_zone_id: Option<usize>,
    pub zone_strides: TVec<isize>,
    pub input_storage_strides: TVec<isize>,
    pub output_storage_strides: TVec<isize>,
}

impl Debug for Patch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.spec)
    }
}

impl Patch {
    #[inline]
    pub fn rank(&self) -> usize {
        self.spec.input_shape.len()
    }

    unsafe fn is_valid(&self, coords: &[usize]) -> bool {
        for ix in 0..self.rank() {
            let c = *coords.get_unchecked(ix) as isize;
            let strides = *self.spec.strides.get_unchecked(ix) as isize;
            let pos = c * strides;
            let min_max = self.data_field_min_max.get_unchecked(ix);
            if pos + min_max.0 < 0
                || pos + min_max.1 >= *self.spec.input_shape.get_unchecked(ix) as isize
            {
                return false;
            }
        }
        true
    }

    pub fn valid_zone(&self) -> Option<&Zone> {
        self.valid_zone_id.map(|id| &self.zones[id])
    }

    #[inline]
    pub fn visit_output(&self, mut acceptor: impl FnMut(&Scanner)) {
        if self.zones.len() == 0 {
            return;
        }
        let mut scanner = Scanner::new(self);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }

    pub fn centers_offsets(&self) -> Vec<isize> {
        if self.zones.len() == 0 {
            return vec![];
        }
        let mut scanner = Scanner::new(self);
        let len = self.output_shape.iter().cloned().product();
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(scanner.input_center_offset);
            scanner.next()
        }
        v
    }

    pub fn at<'p>(&'p self, coords: &[usize]) -> PatchIterator<'p> {
        self.at_hint(coords, None)
    }

    pub fn at_hint<'p>(&'p self, coords: &[usize], hint: Option<bool>) -> PatchIterator<'p> {
        unsafe {
            assert_eq!(coords.len(), self.spec.kernel_shape.len());
            let mut center = 0;
            for i in 0..self.op_strides_times_input_storage_strides.len() {
                center += *self.op_strides_times_input_storage_strides.get_unchecked(i)
                    * *coords.get_unchecked(i) as isize;
            }
            let valid = hint.unwrap_or_else(|| !self.padded || self.is_valid(coords));
            if valid {
                PatchIterator::Fast(FastPatchIterator { patch: self, center, item: 0 })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center
                    .iter_mut()
                    .zip(self.spec.strides.iter())
                    .for_each(|(a, &b)| *a *= b);
                PatchIterator::Safe(SafePatchIterator {
                    patch: self,
                    item: 0,
                    input_patch_center,
                    center,
                })
            }
        }
    }

    pub fn global_offset_for(&self, coords: &[usize], patch_index: usize) -> usize {
        assert_eq!(coords.len(), self.spec.kernel_shape.len());
        let center = izip!(coords, &self.op_strides_times_input_storage_strides)
            .map(|(a, b)| *a as isize * *b)
            .sum::<isize>();
        (center + self.standard_layout_data_field[patch_index]) as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Zone {
    pub valid: bool,
    pub input_zone_offset: isize,
    pub output_zone_offset: isize,
    pub output_ranges: Box<[Range<usize>]>,
    pub output_shape: Box<[usize]>,
    /// (index in kernel, offset from center in image)
    pub values_offsets: Box<[(usize, isize)]>,
}

impl Zone {
    pub fn contains_output(&self, coords: &[usize]) -> bool {
        self.output_ranges.iter().zip(coords).all(|(range, &x)| x >= range.start && x < range.end)
    }

    #[inline]
    pub fn visit_output(&self, patch: &Patch, mut acceptor: impl FnMut(&ZoneScanner)) {
        let mut scanner = ZoneScanner::new(self, patch);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneScanner<'p> {
    pub patch: &'p Patch,
    pub zone: &'p Zone,
    pub output_offset: isize,
    pub output_coords: Box<[usize]>,
    pub input_center_offset: isize,
    pub inner_loop_axis: usize,
    pub inner_loop_len: usize,
    pub inner_loop_output_range: Range<usize>,
    pub inner_loop_output_stride: isize,
    pub inner_loop_input_full_stride: isize,
    pub done: bool,
}

impl<'p> ZoneScanner<'p> {
    pub fn new(zone: &'p Zone, patch: &'p Patch) -> ZoneScanner<'p> {
        let inner_loop_axis =
            zone.output_shape.iter().enumerate().max_by_key(|(_, dim)| *dim).unwrap().0;
        let inner_loop_output_range = zone.output_ranges[inner_loop_axis].clone();
        let inner_loop_output_stride = patch.output_storage_strides[inner_loop_axis];
        let inner_loop_input_full_stride =
            patch.op_strides_times_input_storage_strides[inner_loop_axis];
        let mut scan = ZoneScanner {
            patch,
            zone,
            output_offset: 0,
            input_center_offset: 0,
            inner_loop_axis,
            inner_loop_len: inner_loop_output_range.len(),
            inner_loop_output_range,
            inner_loop_output_stride,
            inner_loop_input_full_stride,
            output_coords: zone.output_ranges.iter().map(|r| r.start).collect(),
            done: false,
        };
        scan.refresh_dependent();
        scan
    }

    #[inline]
    pub fn valid_offsets_ker_in(&self) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }

    pub unsafe fn next_non_inner_axis(&mut self) {
        let rank = self.patch.rank();
        let inner_loop_axis = self.inner_loop_axis;
        for axis in (0..rank).rev() {
            if axis == inner_loop_axis {
                continue;
            }
            *self.output_coords.get_unchecked_mut(axis) += 1;
            if *self.output_coords.get_unchecked_mut(axis)
                < self.zone.output_ranges.get_unchecked(axis).end
            {
                self.refresh_dependent();
                return;
            }
            *self.output_coords.get_unchecked_mut(axis) =
                self.zone.output_ranges.get_unchecked(axis).start;
        }
        self.done = true;
    }

    pub unsafe fn reset(&mut self) {
        self.output_offset = 0;
        self.input_center_offset = 0;
        for ix in 0..self.output_coords.len() {
            *self.output_coords.get_unchecked_mut(ix) =
                self.zone.output_ranges.get_unchecked(ix).start;
        }
        self.done = false;
        self.refresh_dependent()
    }

    #[inline(never)]
    fn refresh_dependent(&mut self) {
        self.input_center_offset = self
            .patch
            .op_strides_times_input_storage_strides
            .iter()
            .zip(self.output_coords.iter())
            .map(|(a, b)| *a * *b as isize)
            .sum();
        self.output_offset = self
            .patch
            .output_storage_strides
            .iter()
            .zip(self.output_coords.iter())
            .map(|(a, b)| a * *b as isize)
            .sum();
    }

    #[inline]
    pub fn next(&mut self) {
        let inner_loop_axis = self.inner_loop_axis;
        unsafe {
            *self.output_coords.get_unchecked_mut(inner_loop_axis) += 1;
            if *self.output_coords.get_unchecked(inner_loop_axis) < self.inner_loop_output_range.end
            {
                self.input_center_offset += self.inner_loop_input_full_stride;
                self.output_offset += self.inner_loop_output_stride;
            } else {
                *self.output_coords.get_unchecked_mut(inner_loop_axis) =
                    self.inner_loop_output_range.start;
                self.next_non_inner_axis();
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scanner<'p> {
    pub patch: &'p Patch,
    pub zone_id: usize,
    pub zone_coords: TVec<usize>,
    pub zone: &'p Zone,
    pub output_offset: isize,
    pub output_coords: TVec<usize>,
    pub input_coords: TVec<usize>,
    pub input_center_offset: isize,
    done: bool,
}

impl<'p> Scanner<'p> {
    fn new(patch: &'p Patch) -> Scanner<'p> {
        let rank = patch.rank();
        let zone = &patch.zones[0];
        Scanner {
            patch,
            zone_coords: tvec!(0; rank),
            zone,
            zone_id: 0,
            output_offset: 0,
            input_center_offset: 0,
            input_coords: tvec!(0; rank),
            output_coords: tvec!(0; rank),
            done: false,
        }
    }

    #[inline]
    pub fn valid_count(&self) -> usize {
        self.zone.values_offsets.len()
    }

    #[inline]
    pub fn valid_offsets(&self) -> impl Iterator<Item = isize> + '_ {
        self.zone.values_offsets.iter().map(move |pair| pair.1 + self.input_center_offset)
    }

    #[inline]
    pub fn valid_offsets_ker_in(&self) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }

    #[inline]
    pub fn next(&mut self) {
        let rank = self.patch.rank();
        let inner_dim = rank - 1;
        unsafe {
            *self.output_coords.get_unchecked_mut(inner_dim) += 1;
            *self.input_coords.get_unchecked_mut(inner_dim) +=
                *self.patch.spec.strides.get_unchecked(inner_dim);
            self.output_offset += self.patch.spec.output_inner_stride as isize;
            self.input_center_offset +=
                self.patch.op_strides_times_input_storage_strides.get_unchecked(inner_dim);
            if *self.output_coords.get_unchecked(inner_dim)
                < self.zone.output_ranges.get_unchecked(inner_dim).end
            {
                return;
            }
            if self.output_coords.get_unchecked(inner_dim)
                < self.patch.output_shape.get_unchecked(inner_dim)
            {
                self.zone_id += 1;
                *self.zone_coords.get_unchecked_mut(inner_dim) += 1;
                self.zone = self.patch.zones.get_unchecked(self.zone_id);
            } else {
                for axis in (0..rank - 1).rev() {
                    *self.output_coords.get_unchecked_mut(axis + 1) = 0;
                    *self.input_coords.get_unchecked_mut(axis + 1) = 0;
                    *self.output_coords.get_unchecked_mut(axis) += 1;
                    *self.input_coords.get_unchecked_mut(axis) +=
                        self.patch.spec.strides.get_unchecked(axis);
                    *self.zone_coords.get_unchecked_mut(axis + 1) = 0;
                    if *self.output_coords.get_unchecked(axis)
                        == self.zone.output_ranges.get_unchecked(axis).end
                    {
                        *self.zone_coords.get_unchecked_mut(axis) += 1;
                    }
                    if *self.output_coords.get_unchecked(axis)
                        < *self.patch.output_shape.get_unchecked(axis)
                    {
                        break;
                    }
                }
                if self.output_coords.get_unchecked(0) == self.patch.output_shape.get_unchecked(0) {
                    self.done = true;
                    return;
                }
                self.zone_id = 0;
                self.input_center_offset = 0;
                for i in 0..rank {
                    self.zone_id += *self.zone_coords.get_unchecked(i)
                        * *self.patch.zone_strides.get_unchecked(i) as usize;
                    self.input_center_offset += *self.input_coords.get_unchecked(i) as isize
                        * *self.patch.input_storage_strides.get_unchecked(i);
                }
                self.zone = self.patch.zones.get_unchecked(self.zone_id);
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Debug)]
pub enum PatchIterator<'p> {
    Fast(FastPatchIterator<'p>),
    Safe(SafePatchIterator<'p>),
}

impl Iterator for PatchIterator<'_> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        match self {
            PatchIterator::Fast(ref mut it) => it.next(),
            PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

#[derive(Debug)]
pub struct FastPatchIterator<'p> {
    patch: &'p Patch,
    center: isize,
    item: usize,
}

impl Iterator for FastPatchIterator<'_> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        if self.item == self.patch.standard_layout_data_field.len() {
            return None;
        }
        unsafe {
            let position =
                self.center + self.patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(position))
        }
    }
}

#[derive(Debug)]
pub struct SafePatchIterator<'p> {
    patch: &'p Patch,
    item: usize,
    input_patch_center: TVec<usize>,
    center: isize,
}

impl Iterator for SafePatchIterator<'_> {
    type Item = Option<isize>;
    fn next(&mut self) -> Option<Option<isize>> {
        unsafe {
            if self.item == self.patch.standard_layout_data_field.len() {
                return None;
            }
            let input_shape = &self.patch.spec.input_shape;
            let img_offset = self.patch.data_field.as_ptr().add(self.item * input_shape.len());

            for ix in 0..input_shape.len() {
                let pos = *self.input_patch_center.get_unchecked(ix) as isize + *img_offset.add(ix);
                if pos < 0 || pos as usize >= *input_shape.get_unchecked(ix) {
                    self.item += 1;
                    return Some(None);
                }
            }
            let position =
                self.center + self.patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(position))
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::ops::nn::DataFormat::*;
    use proptest::prelude::*;
    use proptest::*;

    fn compute_output_spatial_dim(
        input: usize,
        dilation: usize,
        kdim: usize,
        pad_before: usize,
        bad_after: usize,
        stride: usize,
    ) -> usize {
        let patch = PatchSpec::for_full_shape(NCHW, &[1, 1, input])
            .unwrap()
            .with_dilations(tvec!(dilation))
            .with_kernel_shape(tvec!(kdim))
            .with_padding(PaddingSpec::ExplicitOnnxPool(tvec![pad_before], tvec![bad_after], true))
            .with_strides(tvec![stride])
            .into_patch();
        patch.output_shape[0]
    }

    #[test]
    fn basic() {
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 0, 0, 1), 3);
    }

    #[test]
    fn strides() {
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 0, 0, 2), 3);
    }

    #[test]
    fn padding() {
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 1, 1, 1), 5);
    }

    #[test]
    fn strides_and_padding() {
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 1, 1, 2), 4);
    }

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<isize> {
        let patch =
            PatchSpec::for_data_shape(NCHW.from_n_c_hw(1, 1, tvec![10; kdim.len()]).unwrap())
                .with_dilations(dilations.into())
                .with_kernel_shape(kdim.into())
                .with_strides(tvec![1; kdim.len()])
                .into_patch();
        patch.data_field
    }

    #[test]
    fn test_field() {
        assert_eq!(field(&[3], &[1]), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2]), arr2(&[[0], [2], [4]]));
        assert_eq!(field(&[2, 2], &[1, 1]), arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(field(&[2, 2], &[2, 1]), arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]]));
    }

    pub fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
        let len = shape.iter().product::<usize>();
        let shape = shape.to_vec();
        proptest::collection::vec(any::<i8>().prop_map(|i| i as f32), len..=len)
            .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap().into_tensor())
            .boxed()
    }

    #[derive(Debug)]
    struct Problem {
        patch: Patch,
        input: Tensor,
        data_format: DataFormat,
    }

    impl Arbitrary for Problem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Problem>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                prop_oneof!(Just(NCHW), Just(NHWC)),
                (1usize..3, 1usize..3),
                1usize..3,
                (1usize..3, 1usize..3),
                prop_oneof![
                    Just(PaddingSpec::Valid),
                    Just(PaddingSpec::SameLower),
                    Just(PaddingSpec::SameUpper)
                ],
                (1usize..4, 1usize..4),
            )
                .prop_flat_map(|p| {
                    let dil = p.1;
                    let ks = p.3;
                    let strides = p.5;
                    let min_size: (usize, usize) = (1 + (ks.0 - 1) * dil.0, 1 + (ks.1 - 1) * dil.1);
                    (
                        Just(p),
                        (min_size.0..min_size.0 + strides.0 * 3),
                        (min_size.1..min_size.1 + strides.1 * 3),
                    )
                })
                .prop_flat_map(|(p, h, w)| {
                    let input_shape = p.0.from_n_c_hw(1, p.2, [h, w]).unwrap();
                    let input = tensor(&input_shape.shape);
                    (Just(p), input)
                })
                .prop_map(|((fmt, dil, c, ks, pad, strides), input)| {
                    let output_inner_stride = if fmt.c_is_last() { c } else { 1 };
                    Problem {
                        patch: PatchSpec::for_full_shape(fmt, input.shape())
                            .unwrap()
                            .with_dilations(tvec!(dil.0, dil.1))
                            .with_kernel_shape(tvec!(ks.0, ks.1))
                            .with_padding(pad)
                            .with_strides(tvec![strides.0, strides.1])
                            .with_output_inner_stride(output_inner_stride)
                            .into_patch(),
                        input,
                        data_format: fmt,
                    }
                })
                .boxed()
        }
    }

    impl Problem {
        fn input_shape(&self) -> DataShape {
            self.data_format.shape(self.input.shape().into()).unwrap()
        }

        fn output_shape(&self) -> DataShape {
            self.data_format
                .from_n_c_hw(
                    self.input_shape().n().cloned().unwrap_or(1),
                    *self.input_shape().c(),
                    &*self.patch.output_shape,
                )
                .unwrap()
        }

        fn reference_sumpool(&self) -> Tensor {
            let input_shape = self.input_shape();
            let output_shape = self.output_shape();
            let mut output = Tensor::zero::<f32>(&output_shape.shape).unwrap();
            for geo_out in tract_ndarray::indices(output_shape.hw_dims()) {
                for geo_ker in tract_ndarray::indices(&*self.patch.spec.kernel_shape) {
                    let geo_in: TVec<isize> = izip!(
                        geo_out.slice(),
                        geo_ker.slice(),
                        &self.patch.spec.strides,
                        &self.patch.spec.dilations,
                        &self.patch.pad_before
                    )
                    .map(|(o, k, s, d, p)| (o * s + k * d) as isize - *p as isize)
                    .collect();
                    if izip!(&geo_in, input_shape.hw_dims())
                        .any(|(g, i)| *g >= *i as isize || *g < 0)
                    {
                        continue;
                    }
                    let geo_in: TVec<usize> = geo_in.into_iter().map(|x| x as usize).collect();
                    for c in 0..*output_shape.c() {
                        let ocoords = self.data_format.from_n_c_hw(0, c, geo_out.slice()).unwrap();
                        let icoords = self.data_format.from_n_c_hw(0, c, &geo_in).unwrap();
                        output.to_array_view_mut::<f32>().unwrap()[&*ocoords.shape] +=
                            self.input.to_array_view::<f32>().unwrap()[&*icoords.shape];
                    }
                }
            }
            output
        }

        fn check_visitor(&self) {
            let input_shape = self.input_shape();
            let output_shape = self.output_shape();
            let mut output = Tensor::zero::<f32>(&output_shape.shape).unwrap();
            self.patch.visit_output(|visitor| {
                for (_k, offset_in) in visitor.valid_offsets_ker_in() {
                    for c in 0..*output_shape.c() {
                        output.as_slice_mut::<f32>().unwrap()
                            [visitor.output_offset as usize + c * output_shape.c_stride()] +=
                            self.input.as_slice::<f32>().unwrap()
                                [offset_in as usize + c * input_shape.c_stride()];
                    }
                }
            });
            assert_eq!(output, self.reference_sumpool());
        }

        fn check_zone_visitor(&self) {
            let input_shape = self.input_shape();
            let output_shape = self.output_shape();
            let mut output = Tensor::zero::<f32>(&output_shape.shape).unwrap();
            for zone in &self.patch.zones {
                zone.visit_output(&self.patch, |visitor| {
                    for (_k, offset_in) in visitor.valid_offsets_ker_in() {
                        for c in 0..*output_shape.c() {
                            output.as_slice_mut::<f32>().unwrap()
                                [visitor.output_offset as usize + c * output_shape.c_stride()] +=
                                self.input.as_slice::<f32>().unwrap()
                                    [offset_in as usize + c * input_shape.c_stride()];
                        }
                    }
                });
            }
            assert_eq!(output, self.reference_sumpool());
        }

        fn check_zoning(&self) {
            fn in_zone(full_coords: &[usize], h_axis: usize, zone: &[Range<usize>]) -> bool {
                for a in 0..zone.len() {
                    if full_coords[h_axis + a] < zone[a].start
                        || full_coords[h_axis + a] >= zone[a].end
                    {
                        return false;
                    }
                }
                true
            }

            let valid_zone = &self.patch.valid_output_zone;
            let invalid_zones = &self.patch.invalid_output_zones;
            let output_full_shape = self.output_shape();
            let h_axis = self.input_shape().h_axis();
            for coords in ndarray::indices(&*output_full_shape.shape) {
                let inside_valid = in_zone(coords.slice(), h_axis, valid_zone);
                let invalid_count =
                    invalid_zones.iter().filter(|z| in_zone(coords.slice(), h_axis, z)).count();
                unsafe {
                    assert_eq!(
                        inside_valid,
                        self.patch.is_valid(&coords.slice()[self.input_shape().hw_axes()]),
                        "coords {:?}, valid_zone: {:?} inside_valid: {:?}",
                        coords.slice(),
                        valid_zone,
                        inside_valid
                    );
                }
                if inside_valid {
                    assert_eq!(invalid_count, 0);
                } else {
                    assert_eq!(
                        invalid_count,
                        1,
                        "coords {:?}, valid_zone: {:?} inside_valid: {:?} invalid_zones: {:?}",
                        coords.slice(),
                        valid_zone,
                        inside_valid,
                        invalid_zones
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_visitor(pb in any::<Problem>()) {
            pb.check_visitor();
        }

        #[test]
        fn test_zone_visitor(pb in any::<Problem>()) {
            pb.check_zone_visitor();
        }

        #[test]
        fn test_zoning(pb in any::<Problem>()) {
            pb.check_zoning();
        }
    }

    #[test]
    fn test_visitor_1() {
        let input_shape = NCHW.from_n_c_hw(1, 1, [2, 2]).unwrap();
        let input = Tensor::zero::<f32>(&input_shape.shape).unwrap();
        let patch = PatchSpec::for_data_shape(input_shape.clone())
            .with_kernel_shape(tvec![2, 1])
            .with_padding(PaddingSpec::SameLower)
            .with_strides(tvec![1, 2])
            .into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_visitor();
    }

    #[test]
    fn test_visitor_2() {
        let input_shape = NCHW.from_n_c_hw(1, 2, [1, 1]).unwrap();
        let input = tensor4(&[[[[0.]], [[1f32]]]]);
        assert_eq!(input.shape(), &*input_shape.shape);
        let patch =
            PatchSpec::for_data_shape(input_shape.clone()).with_output_inner_stride(2).into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_visitor();
    }

    #[test]
    fn test_visitor_3() {
        let input_shape = NHWC.from_n_c_hw(1, 2, [2, 1]).unwrap();
        let input = tensor4(&[[[[0., 0.]], [[1., 0f32]]]]);
        assert_eq!(input.shape(), &*input_shape.shape);
        let patch =
            PatchSpec::for_data_shape(input_shape.clone()).with_output_inner_stride(2).into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_visitor();
    }

    #[test]
    fn test_visitor_4() {
        let input_shape = NCHW.from_n_c_hw(1, 1, [1, 2]).unwrap();
        let input = tensor4(&[[[[0., 1f32]]]]);
        assert_eq!(input.shape(), &*input_shape.shape);
        let patch = PatchSpec::for_data_shape(input_shape.clone())
            .with_kernel_shape(tvec!(1, 2))
            .with_output_inner_stride(1)
            .with_padding(PaddingSpec::SameLower)
            .into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_visitor();
    }

    #[test]
    fn test_zone_visitor_1() {
        let input_shape = NCHW.from_n_c_hw(1, 1, [2, 1]).unwrap();
        let input = tensor4(&[[[[0.], [1f32]]]]);
        assert_eq!(input.shape(), &*input_shape.shape);
        let patch = PatchSpec::for_data_shape(input_shape.clone()).into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_zone_visitor();
    }

    #[test]
    fn test_zone_visitor_2() {
        let input_shape = NCHW.from_n_c_hw(1, 1, [1, 2]).unwrap();
        let input = tensor4(&[[[[0., 1f32]]]]);
        assert_eq!(input.shape(), &*input_shape.shape);
        let patch = PatchSpec::for_data_shape(input_shape.clone()).into_patch();
        Problem { patch, input, data_format: input_shape.fmt }.check_zone_visitor();
    }
}
