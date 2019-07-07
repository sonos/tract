use crate::internal::*;
use crate::ops::cnn::PaddingSpec;
use crate::ops::nn::{DataFormat, DataShape};
use ndarray::prelude::*;

use super::PatchAxis;

use std::ops::Range;

use itertools::zip;
use itertools::Itertools;

use unsafe_unwrap::UnsafeUnwrap;

#[derive(Debug, Clone, PartialEq)]
pub struct PatchSpec {
    pub input_shape: TVec<usize>,
    pub input_inner_stride: usize,
    pub output_inner_stride: usize,
    pub kernel_shape: TVec<usize>,
    pub strides: TVec<usize>,
    pub dilations: TVec<usize>,
    pub padding: PaddingSpec,
}

impl PatchSpec {
    pub fn for_full_shape(data_format: DataFormat, input_full_shape: &[usize]) -> PatchSpec {
        let shape = data_format.shape(input_full_shape.into());
        Self::for_data_shape(shape)
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
            &*self.input_shape,
            &*self.kernel_shape,
            &*self.dilations,
            &*self.strides,
        );
        let output: TVec<usize> = dims.iter().map(|d| d.output).collect();
        let pad_before: TVec<usize> = dims.iter().map(|d| d.pad_before).collect();
        let pad_after: TVec<usize> = dims.iter().map(|d| d.pad_after).collect();

        let data_field: Vec<isize> = ::ndarray::indices(&*self.kernel_shape)
            .into_iter()
            .flat_map(|coords| {
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
            .gencolumns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        fn strides(shape: &[usize], inner: usize) -> TVec<isize> {
            let mut strides: TVec<isize> = tvec![inner as isize];
            for dim in shape.into_iter().skip(1).rev() {
                let previous = *strides.last().unwrap();
                strides.push(*dim as isize * previous);
            }
            strides.reverse();
            strides
        }

        let input_layout_strides = strides(&*self.input_shape, self.input_inner_stride);
        let output_layout_strides = strides(&*output, self.output_inner_stride);

        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| zip(coords, &input_layout_strides).map(|(a, b)| a * b).sum::<isize>())
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
                    output_dim: d.output,
                    stride: self.strides[ix],
                    dilation: self.dilations[ix],
                }
                .regions()
            })
            .collect::<TVec<_>>();

        let op_strides_times_input_storage_strides =
            zip(&self.strides, &input_layout_strides).map(|(a, b)| (*a as isize * b)).collect();

        let zone_strides = strides(&regions.iter().map(|d| d.len()).collect::<TVec<_>>(), 1);
        let zones: Vec<Zone> = regions
            .iter()
            .multi_cartesian_product()
            .map(|regions| {
                let validity = ndarray::indices(&*self.kernel_shape)
                    .into_iter()
                    .map(|coords| {
                        zip(coords.slice(), &regions).all(|(&x, axis)| {
                            axis.mask.as_ref().map(|mask| !mask[x]).unwrap_or(true)
                        })
                    })
                    .collect::<Vec<bool>>();
                let valid = validity.iter().all(|x| *x);
                let output_shape: TVec<usize> =
                    regions.iter().map(|reg| reg.range.end - reg.range.start).collect();
                let invalid_point_offset_pairs = if !valid {
                    Some(
                        ndarray::indices(&*output_shape)
                            .into_iter()
                            .map(|coords| {
                                (
                                    itertools::izip!(
                                        coords.slice(),
                                        regions.iter(),
                                        &op_strides_times_input_storage_strides
                                    )
                                    .map(|(a, r, &b)| ((a + r.range.start) as isize * b))
                                    .sum::<isize>(),
                                    itertools::izip!(
                                        coords.slice(),
                                        regions.iter(),
                                        &output_layout_strides
                                    )
                                    .map(|(a, r, &b)| ((a + r.range.start) as isize * b))
                                    .sum::<isize>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                Zone {
                    invalid_point_offset_pairs,
                    output_ranges: regions.iter().map(|reg| reg.range.clone()).collect(),
                    output_shape,
                    output_zone_offset: zip(&regions, &output_layout_strides)
                        .map(|(reg, &stride)| reg.range.start as isize * stride)
                        .sum::<isize>(),
                    valid,
                    values_offsets: standard_layout_data_field
                        .iter()
                        .cloned()
                        .enumerate()
                        .filter(|(ix, _)| validity[*ix])
                        .collect(),
                    validity,
                }
            })
            .collect();

        let valid_zone = zones.iter().position(|z| z.valid);

        Patch {
            spec: self,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            output_shape: output,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            input_layout_strides,
            output_layout_strides,
            op_strides_times_input_storage_strides,
            zones,
            valid_zone,
            zone_strides,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    pub zones: Vec<Zone>,
    pub valid_zone: Option<usize>,
    pub zone_strides: TVec<isize>,
    pub input_layout_strides: TVec<isize>,
    pub output_layout_strides: TVec<isize>,
}

impl Patch {
    #[inline]
    pub fn rank(&self) -> usize {
        self.spec.input_shape.len()
    }

    #[inline]
    pub fn visit_output_in_order(&self, mut acceptor: impl FnMut(&InOrderScanner)) {
        let mut scanner = InOrderScanner::new(self);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }

    #[inline]
    pub fn visit_output_by_zone(&self, mut acceptor: impl FnMut(&ByZoneScanner)) {
        let mut scanner = ByZoneScanner::new(self, false);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }

    #[inline]
    pub fn visit_invalid(&self, mut acceptor: impl FnMut(&InvalidScanner)) {
        let mut scanner = InvalidScanner::new(self);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }

    pub fn centers_offsets(&self) -> Vec<isize> {
        let mut scanner = InOrderScanner::new(self);
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
                PatchIterator::Fast(FastPatchIterator { patch: &self, center, item: 0 })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center
                    .iter_mut()
                    .zip(self.spec.strides.iter())
                    .for_each(|(a, &b)| *a *= b as usize);
                PatchIterator::Safe(SafePatchIterator {
                    patch: self,
                    item: 0,
                    input_patch_center,
                    center,
                })
            }
        }
    }

    fn is_valid(&self, coords: &[usize]) -> bool {
        if let Some(zone) = self.valid_zone {
            let zone = &self.zones[zone];
            zone.output_ranges.iter().zip(coords).all(|(r, &x)| x >= r.start && x < r.end)
        } else {
            false
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    valid: bool,
    output_zone_offset: isize,
    output_ranges: TVec<Range<usize>>,
    output_shape: TVec<usize>,
    /// (index, raw offset)
    values_offsets: Vec<(usize, isize)>,
    invalid_point_offset_pairs: Option<Vec<(isize, isize)>>,
    validity: Vec<bool>,
}

impl Zone {
    pub fn contains_output(&self, coords: &[usize]) -> bool {
        self.output_ranges.iter().zip(coords).all(|(range, &x)| x >= range.start && x < range.end)
    }

    pub fn output_ranges(&self) -> &[Range<usize>] {
        &*self.output_ranges
    }

    pub fn input_center_offsets(&self, patch: &Patch) -> Vec<isize> {
        ndarray::indices(&*self.output_shape)
            .into_iter()
            .map(|coords| {
                itertools::izip!(
                    coords.slice(),
                    &self.output_ranges,
                    &patch.op_strides_times_input_storage_strides
                )
                .into_iter()
                .map(|(x, range, stride)| (x + range.start) as isize * stride)
                .sum::<isize>()
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InvalidScanner<'p> {
    patch: &'p Patch,
    zone: &'p Zone,
    zone_id: usize,
    point_id: usize,
    done: bool,
}

impl<'p> InvalidScanner<'p> {
    fn new(patch: &'p Patch) -> InvalidScanner<'p> {
        let zone_id = if let Some(inv) = patch.zones.iter().position(|z| !z.valid) {
            inv
        } else {
            return InvalidScanner {
                patch,
                zone: &patch.zones[0],
                zone_id: 0,
                point_id: 0,
                done: true,
            };
        };
        let scanner = InvalidScanner {
            patch,
            zone: &patch.zones[zone_id],
            zone_id,
            point_id: 0,
            done: false,
        };
        scanner
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    pub unsafe fn point_offsets(&self) -> (isize, isize) {
        *self.zone.invalid_point_offset_pairs.as_ref().unsafe_unwrap().get_unchecked(self.point_id)
    }

    #[inline]
    pub fn valid_count(&self) -> usize {
        self.zone.values_offsets.len()
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    pub fn valid_kernel_offsets_with_indexes(&self) -> &[(usize, isize)] {
        &self.zone.values_offsets
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    pub fn next(&mut self) {
        unsafe {
            self.point_id += 1;
            if self.point_id < self.zone.invalid_point_offset_pairs.as_ref().unsafe_unwrap().len() {
                return;
            }
            self.point_id = 0;
            self.zone_id += 1;
            while self.zone_id < self.patch.zones.len()
                && self.patch.zones.get_unchecked(self.zone_id).valid
            {
                self.zone_id += 1
            }
            if self.zone_id < self.patch.zones.len() {
                self.zone = self.patch.zones.get_unchecked(self.zone_id)
            } else {
                self.done = true
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ByZoneScanner<'p> {
    patch: &'p Patch,
    skip_valid: bool,
    zone: &'p Zone,
    zone_id: usize,
    output_offset: isize,
    output_coords: TVec<usize>,
    input_coords: TVec<usize>,
    input_center_offset: isize,
    done: bool,
}

impl<'p> ByZoneScanner<'p> {
    fn new(patch: &'p Patch, skip_valid: bool) -> ByZoneScanner<'p> {
        let rank = patch.rank();
        let zone_id = if skip_valid {
            if let Some(inv) = patch.zones.iter().position(|z| !z.valid) {
                inv
            } else {
                return ByZoneScanner {
                    patch,
                    skip_valid,
                    zone: &patch.zones[0],
                    zone_id: 0,
                    output_offset: 0,
                    input_center_offset: 0,
                    input_coords: tvec!(0; rank),
                    output_coords: tvec!(0; rank),
                    done: true,
                };
            }
        } else {
            0
        };
        let mut scanner = ByZoneScanner {
            patch,
            skip_valid,
            zone: &patch.zones[zone_id],
            zone_id,
            output_offset: 0,
            input_center_offset: 0,
            input_coords: tvec!(0; rank),
            output_coords: tvec!(0; rank),
            done: false,
        };
        unsafe {
            scanner.to_zone_start();
        }
        scanner
    }

    #[inline]
    pub fn output_offset(&self) -> isize {
        self.output_offset
    }

    #[inline]
    pub fn input_center_offset(&self) -> isize {
        self.input_center_offset
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
    pub unsafe fn is_nth_valid(&self, n: usize) -> bool {
        *self.zone.validity.get_unchecked(n)
    }

    #[inline]
    pub unsafe fn nth_offset_if_valid(&self, n: usize) -> Option<isize> {
        if self.is_nth_valid(n) {
            Some(self.input_center_offset + *self.patch.standard_layout_data_field.get_unchecked(n))
        } else {
            None
        }
    }

    /*
    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    pub fn valid_offsets_with_indexes(&self) -> impl Iterator<Item = (usize, isize)>  {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }
    */

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    pub fn valid_kernel_offsets_with_indexes(&self) -> &[(usize, isize)] {
        &self.zone.values_offsets
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
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
            for axis in (0..rank - 1).rev() {
                *self.output_coords.get_unchecked_mut(axis + 1) =
                    self.zone.output_ranges.get_unchecked(axis + 1).start;
                *self.input_coords.get_unchecked_mut(axis + 1) =
                    self.zone.output_ranges.get_unchecked(axis + 1).start
                        * self.patch.spec.strides.get_unchecked(axis + 1);

                *self.output_coords.get_unchecked_mut(axis) += 1;
                *self.input_coords.get_unchecked_mut(axis) +=
                    self.patch.spec.strides.get_unchecked(axis);
                if *self.output_coords.get_unchecked(axis)
                    < self.zone.output_ranges.get_unchecked(axis).end
                {
                    self.reset_raw_offsets();
                    return;
                }
            }
            self.next_zone();
        }
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    unsafe fn reset_raw_offsets(&mut self) {
        self.input_center_offset = 0;
        self.output_offset = 0;
        for ix in 0..self.patch.rank() {
            self.input_center_offset += *self.patch.input_layout_strides.get_unchecked(ix)
                * *self.input_coords.get_unchecked(ix) as isize;
            self.output_offset += self.patch.output_layout_strides.get_unchecked(ix)
                * *self.output_coords.get_unchecked(ix) as isize;
        }
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    unsafe fn next_zone(&mut self) {
        self.zone_id += 1;
        while self.skip_valid
            && self.patch.zones.get_unchecked(self.zone_id).valid
            && self.zone_id < self.patch.zones.len()
        {
            self.zone_id += 1;
        }
        if self.zone_id == self.patch.zones.len() {
            self.done = true;
            return;
        }
        self.zone = self.patch.zones.get_unchecked(self.zone_id);
        self.to_zone_start()
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    unsafe fn to_zone_start(&mut self) {
        for ix in 0..self.patch.rank() {
            *self.input_coords.get_unchecked_mut(ix) =
                self.zone.output_ranges.get_unchecked(ix).start
                    * self.patch.spec.strides.get_unchecked(ix);
            *self.output_coords.get_unchecked_mut(ix) =
                self.zone.output_ranges.get_unchecked(ix).start;
        }
        self.reset_raw_offsets()
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InOrderScanner<'p> {
    patch: &'p Patch,
    rank: usize,
    zone_id: usize,
    zone_coords: TVec<usize>,
    zone: &'p Zone,
    output_offset: isize,
    output_coords: TVec<usize>,
    input_coords: TVec<usize>,
    input_center_offset: isize,
    done: bool,
    inner_output_coord: usize,
    inner_input_coord: usize,
    inner_input_stride: isize,
    inner_output_stride: isize,
    inner_current_zone_limit: usize,
    input_center_offset_stride: isize,
}

macro_rules! inc { ($a:expr, $b:expr) => { $a = $a.wrapping_add($b) } }

impl<'p> InOrderScanner<'p> {
    fn new(patch: &'p Patch) -> InOrderScanner<'p> {
        let rank = patch.rank();
        let zone = &patch.zones[0];
        InOrderScanner {
            patch,
            rank,
            zone_coords: tvec!(0; rank),
            zone,
            zone_id: 0,
            output_offset: 0,
            input_center_offset: 0,
            input_coords: tvec!(0; rank),
            output_coords: tvec!(0; rank),
            done: false,
            inner_output_coord: 0,
            inner_input_coord: 0,
            inner_output_stride: patch.spec.output_inner_stride as isize,
            inner_input_stride: patch.spec.strides[rank - 1] as isize,
            inner_current_zone_limit: zone.output_ranges[rank - 1].end,
            input_center_offset_stride: patch.op_strides_times_input_storage_strides[rank-1],
        }
    }

    #[inline]
    pub fn output_offset(&self) -> isize {
        self.output_offset
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
    pub unsafe fn is_nth_valid(&self, n: usize) -> bool {
        *self.zone.validity.get_unchecked(n)
    }

    #[inline]
    pub unsafe fn nth_offset_if_valid(&self, n: usize) -> Option<isize> {
        if self.is_nth_valid(n) {
            Some(self.input_center_offset + *self.patch.standard_layout_data_field.get_unchecked(n))
        } else {
            None
        }
    }

    #[inline]
    pub fn valid_offsets_with_indexes(&self) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }


    #[inline]
    pub fn next(&mut self) {
        unsafe {
            inc!(self.inner_output_coord, 1);
            inc!(self.inner_input_coord, self.inner_input_stride as usize);
            inc!(self.output_offset, self.inner_output_stride);
            inc!(self.input_center_offset, self.input_center_offset_stride);
            if self.inner_output_coord >= self.inner_current_zone_limit {
                let inner_dim = self.rank.wrapping_sub(1);
                *self.output_coords.get_unchecked_mut(inner_dim) = self.inner_output_coord;
                *self.input_coords.get_unchecked_mut(inner_dim) = self.inner_input_coord;
                self.next_zone();
                self.inner_output_coord = *self.output_coords.get_unchecked(inner_dim);
                self.inner_input_coord = *self.input_coords.get_unchecked(inner_dim);
                self.inner_current_zone_limit = self.zone.output_ranges.get_unchecked(inner_dim).end;
            }
        }
    }

    fn next_zone(&mut self) {
        let rank = self.rank;
        let inner_dim = rank.wrapping_sub(1);
        unsafe {
            if self.output_coords.get_unchecked(inner_dim)
                < self.patch.output_shape.get_unchecked(inner_dim)
            {
                inc!(self.zone_id, 1);
                inc!(*self.zone_coords.get_unchecked_mut(inner_dim), 1);
                self.zone = self.patch.zones.get_unchecked(self.zone_id);
            } else {
                for ix in 0..(inner_dim as usize) {
                    let axis_p_1 = inner_dim.wrapping_sub(ix);
                    let axis = axis_p_1.wrapping_sub(1);
                    *self.output_coords.get_unchecked_mut(axis_p_1) = 0;
                    *self.input_coords.get_unchecked_mut(axis_p_1) = 0;
                    inc!(*self.output_coords.get_unchecked_mut(axis), 1);
                    inc!(*self.input_coords.get_unchecked_mut(axis), *self.patch.spec.strides.get_unchecked(axis));
                    *self.zone_coords.get_unchecked_mut(axis_p_1) = 0;
                    if *self.output_coords.get_unchecked(axis)
                        == self.zone.output_ranges.get_unchecked(axis).end
                    {
                        inc!(*self.zone_coords.get_unchecked_mut(axis), 1);
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
                    inc!(self.zone_id, self.zone_coords.get_unchecked(i)
                        .wrapping_mul(*self.patch.zone_strides.get_unchecked(i) as usize));
                    inc!(self.input_center_offset, self.input_coords.get_unchecked(i)
                        .wrapping_mul(*self.patch.input_layout_strides.get_unchecked(i) as usize) as isize);
                }
                self.zone = &self.patch.zones.get_unchecked(self.zone_id);
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::ops::nn::DataFormat::NCHW;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
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
            .with_dilations(tvec!(dilation))
            .with_kernel_shape(tvec!(kdim))
            .with_padding(PaddingSpec::Explicit(tvec![pad_before], tvec![bad_after]))
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
        let patch = PatchSpec::for_data_shape(NCHW.from_n_c_hw(1, 1, tvec![10; kdim.len()]))
            .with_dilations(dilations.into())
            .with_kernel_shape(kdim.into())
            .with_padding(PaddingSpec::Explicit(tvec![0; kdim.len()], tvec![0; kdim.len()]))
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

    pub fn patch_2d() -> BoxedStrategy<(DataShape, PatchSpec)> {
        (
            Just(DataFormat::NCHW),
            (1usize..3, 1usize..3),
            1usize..3,
            (1usize..3, 1usize..3),
            //prop_oneof![PaddingSpec::SameLower, PaddingSpec::Valid],
            Just(PaddingSpec::SameLower),
            (1usize..4, 1usize..4),
        )
            .prop_flat_map(|p| {
                let size = p.3;
                (Just(p), (size.0 + 5..=size.0 + 10, size.1 + 5..=size.1 + 10))
            })
            .prop_map(|((fmt, dil, c, ks, pad, strides), inp)| {
                (
                    fmt.from_n_c_hw(1, c, tvec!(inp.0, inp.1)),
                    PatchSpec::for_full_shape(fmt, &[1, c, inp.0, inp.1])
                        .with_dilations(tvec!(dil.0, dil.1))
                        .with_kernel_shape(tvec!(ks.0, ks.1))
                        .with_padding(pad)
                        .with_strides(tvec![strides.0, strides.1]),
                )
            })
            .boxed()
    }

    fn is_valid(patch: &Patch, coords: &[usize]) -> bool {
        for ix in 0..patch.rank() {
            let c = coords[ix] as isize;
            let strides = patch.spec.strides[ix] as isize;
            let pos = c * strides;
            let min_max = patch.data_field_min_max[ix];
            if pos + min_max.0 < 0 || pos + min_max.1 >= patch.spec.input_shape[ix] as isize {
                return false;
            }
        }
        true
    }

    fn visit_all_in_order(_input_shape: DataShape, p: PatchSpec) -> TestCaseResult {
        let p = p.into_patch();
        let mut output_offsets = ndarray::ArrayD::<i32>::from_elem(&*p.output_shape, -1);
        let mut output_coords = ndarray::ArrayD::<i32>::from_elem(&*p.output_shape, -1);
        let mut count = 0;
        p.visit_output_in_order(|w| {
            output_offsets.as_slice_mut().unwrap()[w.output_offset as usize] = count;
            let mut oc = w.output_coords.clone();
            oc[p.rank() - 1] = w.inner_output_coord;
            output_coords[&*oc] = count;
            count += 1;
        });
        prop_assert!(output_offsets.iter().enumerate().all(|(ix, x)| *x == ix as i32));
        prop_assert!(output_coords.iter().enumerate().all(|(ix, x)| *x == ix as i32));
        Ok(())
    }

    fn visit_all_by_zone(_input_shape: DataShape, p: PatchSpec) -> TestCaseResult {
        let p = p.into_patch();
        let mut output_offsets = ndarray::ArrayD::<i32>::from_elem(&*p.output_shape, -1);
        let mut output_coords = ndarray::ArrayD::<i32>::from_elem(&*p.output_shape, -1);
        let mut count = 0;
        p.visit_output_by_zone(|w| {
            output_offsets.as_slice_mut().unwrap()[w.output_offset as usize] = count;
            output_coords[&*w.output_coords] = count;
            count += 1;
        });
        prop_assert!(output_offsets.iter().all(|x| *x != -1));
        prop_assert!(output_coords.iter().all(|x| *x != -1));
        Ok(())
    }

    fn visit_invalid(_input_shape: DataShape, p: PatchSpec) -> TestCaseResult {
        let p = p.into_patch();
        let mut output = ndarray::ArrayD::<i32>::from_elem(&*p.output_shape, 0);
        p.visit_invalid(|w| unsafe {
            output.as_slice_mut().unwrap()[w.point_offsets().1 as usize] = 1;
        });
        for coords in ndarray::indices(&*p.output_shape) {
            prop_assert!((output[&coords] == 0) == is_valid(&p, coords.slice()));
        }
        Ok(())
    }

    fn zoning(input_shape: DataShape, p: PatchSpec) -> TestCaseResult {
        let p = p.into_patch();
        let valid_zone = p.zones.iter().find(|z| z.valid).unwrap();
        let invalid_zones = p.zones.iter().filter(|z| !z.valid).collect::<Vec<_>>();
        let output_full_shape = input_shape.fmt.from_n_c_hw(*input_shape.n(), 1, &*p.output_shape);
        for coords in ndarray::indices(&*output_full_shape.shape) {
            let geo = &coords.slice()[output_full_shape.hw_axes()];
            let inside_valid = valid_zone.contains_output(geo);
            let invalid_count = invalid_zones.iter().filter(|z| z.contains_output(geo)).count();
            prop_assert_eq!(
                inside_valid,
                is_valid(&p, &coords.slice()[input_shape.hw_axes()]),
                "coords {:?}, valid_zone: {:?} inside_valid: {:?}",
                coords.slice(),
                valid_zone,
                inside_valid
            );
            if inside_valid {
                prop_assert_eq!(invalid_count, 0);
            } else {
                prop_assert_eq!(
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
        Ok(())
    }

    proptest! {
        #[test]
        fn prop_visit_all_in_order((input_shape, p) in patch_2d()) {
            visit_all_in_order(input_shape, p)?
        }

        #[test]
        fn prop_visit_all_by_zone((input_shape, p) in patch_2d()) {
            visit_all_by_zone(input_shape, p)?
        }

        #[test]
        fn prop_visit_invalid((input_shape, p) in patch_2d()) {
            visit_invalid(input_shape, p)?
        }

        #[test]
        fn test_zoning((input_shape, p) in patch_2d()) {
            zoning(input_shape, p)?
        }
    }

    #[test]
    fn test_visit_all_in_order_1() {
        let ishape = DataFormat::NCHW.shape(tvec![1, 1, 2, 2]);
        let p = PatchSpec::for_data_shape(ishape.clone())
            .with_kernel_shape(tvec![2, 1])
            .with_padding(PaddingSpec::SameLower)
            .with_strides(tvec![1, 2]);
        visit_all_in_order(ishape, p).unwrap()
    }

    #[test]
    fn test_visit_all_by_zone_1() {
        let ishape = DataFormat::NCHW.shape(tvec![1, 1, 6, 7]);
        let p = PatchSpec::for_data_shape(ishape.clone())
            .with_kernel_shape(tvec![1, 2])
            .with_padding(PaddingSpec::SameLower)
            .with_strides(tvec![1, 1]);
        visit_all_by_zone(ishape, p).unwrap()
    }

    #[test]
    fn test_zoning_1() {
        let ishape = DataFormat::NCHW.shape(tvec![1, 1, 7, 6]);
        let p = PatchSpec::for_data_shape(ishape.clone())
            .with_kernel_shape(tvec![1, 1])
            .with_padding(PaddingSpec::SameLower)
            .with_strides(tvec![2, 1]);
        zoning(ishape, p).unwrap()
    }
}

#[derive(Debug)]
pub enum PatchIterator<'p> {
    Fast(FastPatchIterator<'p>),
    Safe(SafePatchIterator<'p>),
}

impl<'p> Iterator for PatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        match self {
            &mut PatchIterator::Fast(ref mut it) => it.next(),
            &mut PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

#[derive(Debug)]
pub struct FastPatchIterator<'p> {
    patch: &'p Patch,
    center: isize,
    item: usize,
}

impl<'p> Iterator for FastPatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
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

impl<'p> Iterator for SafePatchIterator<'p> {
    type Item = Option<isize>;
    #[cfg_attr(not(debug_assertions), no_panic::no_panic)]
    fn next(&mut self) -> Option<Option<isize>> {
        unsafe {
            if self.item == self.patch.standard_layout_data_field.len() {
                return None;
            }
            let input_shape = &self.patch.spec.input_shape;
            let img_offset =
                self.patch.data_field.as_ptr().offset((self.item * input_shape.len()) as isize);

            for ix in 0..input_shape.len() {
                let pos = *self.input_patch_center.get_unchecked(ix) as isize
                    + *img_offset.offset(ix as isize);
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
