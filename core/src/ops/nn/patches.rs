use super::{DataFormat, DataShape, PaddingSpec};
use crate::internal::*;
use ndarray::prelude::*;
#[cfg(not(debug_assertions))]
use no_panic::no_panic;

use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    pub dilations: TVec<usize>,
    pub kernel_spatial_shape: TVec<usize>,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    pub kernel_strides: TVec<usize>,
    pub input_shape: DataShape<usize, TVec<usize>>,
    pub output_spatial_shape: TVec<usize>,
    pub data_field: Array2<isize>,
    pub data_field_min_max: TVec<(isize, isize)>,
    pub standard_layout_data_field: Vec<isize>,
    pub valid_output_zone: TVec<Range<usize>>,
    pub invalid_output_zones: TVec<TVec<Range<usize>>>,
}

impl Patch {
    pub fn new(
        data_fmt: DataFormat,
        dilations: TVec<usize>,
        kernel_spatial_shape: TVec<usize>,
        padding: &PaddingSpec,
        kernel_strides: TVec<usize>,
        input_full_shape: TVec<usize>,
    ) -> Patch {
        use crate::ops::nn::padding::ComputedPaddedDim;
        let input_shape = data_fmt.shape(input_full_shape);
        let ComputedPaddedDim { pad_after, pad_before, output } = padding.compute(
            input_shape.hw_dims(),
            &kernel_spatial_shape,
            &*dilations,
            &*kernel_strides,
        );

        let data_field: Vec<isize> = ::ndarray::indices(&*kernel_spatial_shape)
            .into_iter()
            .flat_map(|coords| {
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * dilations[ix]) as isize - pad_before[ix] as isize)
            })
            .collect();
        let data_field = Array2::from_shape_vec(
            (kernel_spatial_shape.iter().cloned().product(), kernel_spatial_shape.len()),
            data_field,
        )
        .unwrap();
        let data_field_min_max:TVec<_> = data_field
            .gencolumns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        let mut input_layout_strides: Vec<usize> = vec![1];
        for dim in input_shape.shape.iter().skip(1).rev() {
            let previous = input_layout_strides.last().cloned().unwrap_or(1);
            input_layout_strides.push(dim * previous);
        }
        input_layout_strides.reverse();
        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| {
                coords
                    .iter()
                    .zip(input_layout_strides.iter().skip(input_shape.h_axis()))
                    .map(|(&a, &b)| (a as isize * b as isize))
                    .sum()
            })
            .collect();

        let mut valid_output_zone = tvec!();
        let mut invalid_output_zones = tvec!();
        for ix in 0..input_shape.hw_dims().len() {
            let min_max = data_field_min_max[ix];
            let min = (-min_max.0 as usize).div_ceil(kernel_strides[ix]) as usize;
            let max = (input_shape.hw_dims()[ix] - min_max.1 as usize)
                .div_ceil(kernel_strides[ix]) as usize;
            if min != 0 {
                let mut invalid = valid_output_zone.clone();
                invalid.push(0..min);
                invalid_output_zones.push(invalid);
            }
            if max < output[ix] {
                let mut invalid = valid_output_zone.clone();
                invalid.push(max..output[ix]);
                invalid_output_zones.push(invalid);
            }
            valid_output_zone.push(min..max)
        }

        Patch {
            dilations,
            kernel_spatial_shape,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            kernel_strides,
            input_shape,
            output_spatial_shape: output,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            valid_output_zone,
            invalid_output_zones,
        }
    }

    pub fn output_full_shape(&self, channels: usize) -> TVec<usize> {
        let mut v = self.input_shape.shape.clone();
        v[self.input_shape.c_axis()] = channels;
        v[self.input_shape.hw_axes()].copy_from_slice(&self.output_spatial_shape);
        v
    }

    pub fn wrap<'i, 'p, T: Copy + Datum>(
        &'p self,
        input: &'i ArrayViewD<'i, T>,
    ) -> PatchVisitor<'i, 'p, T> {
        let valid = !self.padded; //input.is_standard_layout() && !self.padded;
        let mut fast_strides: Vec<_> = input.strides().into();
        fast_strides[self.input_shape.hw_axes()]
            .iter_mut()
            .zip(self.kernel_strides.iter())
            .for_each(|(a, &b)| *a *= b as isize);
        PatchVisitor { patch: &self, input, valid, fast_strides }
    }

    unsafe fn is_valid(&self, coords: &[usize]) -> bool {
        let spatial_coords = coords.get_unchecked(self.input_shape.hw_axes());
        for ix in 0..self.input_shape.hw_dims().len() {
            let c = *spatial_coords.get_unchecked(ix) as isize;
            let strides = *self.kernel_strides.get_unchecked(ix) as isize;
            let pos = c * strides;
            let min_max = self.data_field_min_max.get_unchecked(ix);
            if pos + min_max.0 < 0
                || pos + min_max.1 >= *self.input_shape.hw_dims().get_unchecked(ix) as isize
            {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
pub struct PatchVisitor<'i, 'p, T: Copy + Datum> {
    patch: &'p Patch,
    input: &'i ArrayViewD<'i, T>,
    valid: bool,
    fast_strides: Vec<isize>, // kernel strides * storage strides
}

impl<'i, 'p, T: Copy + Datum> PatchVisitor<'i, 'p, T> {
    pub fn at<'v>(&'p self, coords: &[usize]) -> PatchIterator<'i, 'p, 'v, T>
    where
        'i: 'v,
        'p: 'v,
    {
        unsafe {
            let mut center = 0;
            for i in 0..self.fast_strides.len() {
                center += *self.fast_strides.get_unchecked(i) * *coords.get_unchecked(i) as isize;
            }
            if self.valid || self.patch.is_valid(coords) {
                PatchIterator::Fast(FastPatchIterator {
                    visitor: &self,
                    ptr: self.input.as_ptr(),
                    center,
                    item: 0,
                })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center[self.patch.input_shape.hw_axes()]
                    .iter_mut()
                    .zip(self.patch.kernel_strides.iter())
                    .for_each(|(a, &b)| *a *= b as usize);
                PatchIterator::Safe(SafePatchIterator {
                    visitor: self,
                    item: 0,
                    input_patch_center,
                    center,
                    ptr: self.input.as_ptr(),
                })
            }
        }
    }

    pub fn global_offset_for(&self, coords: &[usize], patch_index: usize) -> usize {
        let center = coords
            .iter()
            .zip(self.fast_strides.iter())
            .map(|(&a, &b)| b * a as isize)
            .sum::<isize>();
        (center + self.patch.standard_layout_data_field[patch_index]) as usize
    }
}

#[derive(Debug)]
pub enum PatchIterator<'i: 'v, 'p: 'v, 'v, T: Copy + Datum> {
    Fast(FastPatchIterator<'i, 'p, 'v, T>),
    Safe(SafePatchIterator<'i, 'p, 'v, T>),
}

impl<'i: 'v, 'p: 'v, 'v, T: Copy + Datum + PartialEq> Iterator for PatchIterator<'p, 'i, 'v, T> {
    type Item = Option<T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<T>> {
        match self {
            &mut PatchIterator::Fast(ref mut it) => it.next(),
            &mut PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

#[derive(Debug)]
pub struct FastPatchIterator<'i: 'v, 'p: 'v, 'v, T: Copy + Datum> {
    visitor: &'v PatchVisitor<'i, 'p, T>,
    ptr: *const T,
    center: isize,
    item: usize,
}

impl<'i: 'v, 'p: 'v, 'v, T: Copy + Datum + PartialEq> Iterator
    for FastPatchIterator<'i, 'p, 'v, T>
{
    type Item = Option<T>;
    #[inline(always)]
    #[cfg_attr(not(debug_assertions), no_panic)]
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.visitor.patch.standard_layout_data_field.len() {
            return None;
        }
        unsafe {
            let position = self.center
                + self.visitor.patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(*(self.ptr.offset(position))))
        }
    }
}

#[derive(Debug)]
pub struct SafePatchIterator<'i: 'v, 'p: 'v, 'v, T: Copy + Datum> {
    visitor: &'v PatchVisitor<'i, 'p, T>,
    item: usize,
    input_patch_center: TVec<usize>,
    ptr: *const T,
    center: isize,
}

impl<'i: 'v, 'p: 'v, 'v, T: Copy + Datum + PartialEq> Iterator
    for SafePatchIterator<'i, 'p, 'v, T>
{
    type Item = Option<T>;
    #[cfg_attr(not(debug_assertions), no_panic)]
    fn next(&mut self) -> Option<Option<T>> {
        unsafe {
            let patch = self.visitor.patch;
            if self.item == patch.standard_layout_data_field.len() {
                return None;
            }
            let input_shape = &patch.input_shape;
            let img_offset = patch
                .data_field
                .as_ptr()
                .offset((self.item * (input_shape.shape.len() - 2)) as isize);

            for ix in 0..(input_shape.shape.len() - 2) {
                let ax = input_shape.h_axis() + ix;
                let pos = *self.input_patch_center.get_unchecked(ax) as isize
                    + *img_offset.offset(ix as isize);
                if pos < 0 || pos as usize >= *input_shape.shape.get_unchecked(ax) {
                    self.item += 1;
                    return Some(None);
                }
            }
            let position = self.center + patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(*(self.ptr.offset(position))))
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::ops::nn::DataFormat::NCHW;
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
        let patch = Patch::new(
            DataFormat::NCHW,
            tvec![dilation],
            tvec![kdim],
            &PaddingSpec::Explicit(tvec![pad_before], tvec![bad_after]),
            tvec![stride],
            tvec![1, 1, input],
        );
        patch.output_spatial_shape[0]
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
        let patch = Patch::new(
            NCHW,
            dilations.into(),
            kdim.into(),
            &PaddingSpec::Explicit(tvec![0; kdim.len()], tvec![0; kdim.len()]),
            tvec![1; kdim.len()],
            tvec![10; kdim.len() + 2],
        );
        patch.data_field
    }

    #[test]
    fn test_field() {
        assert_eq!(field(&[3], &[1]), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2]), arr2(&[[0], [2], [4]]));
        assert_eq!(field(&[2, 2], &[1, 1]), arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(field(&[2, 2], &[2, 1]), arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]]));
    }

    pub fn patch_2d() -> BoxedStrategy<Patch> {
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
                Patch::new(
                    fmt,
                    tvec!(dil.0, dil.1),
                    tvec!(ks.0, ks.1),
                    &pad,
                    tvec![strides.0, strides.1],
                    tvec!(1, c, inp.0, inp.1),
                )
            })
            .boxed()
    }

    fn in_zone(coords: &[usize], h_axis: usize, zone: &[Range<usize>]) -> bool {
        for a in 0..zone.len() {
            if coords[h_axis+a] < zone[a].start || coords[h_axis+a] >= zone[a].end {
                return false
            }
        }
        true
    }

    proptest! {
        #[test]
        fn test_2d(p in patch_2d()) {
            let valid_zone = &p.valid_output_zone;
            let invalid_zones = &p.invalid_output_zones;
            let h_axis = p.input_shape.h_axis();
            for coords in ndarray::indices(&*p.output_full_shape(1)) {
                let inside_valid = in_zone(coords.slice(), h_axis, valid_zone);
                let invalid_count = invalid_zones.iter().filter(|z| in_zone(coords.slice(), h_axis, z)).count();
                unsafe {
                    prop_assert_eq!(inside_valid, p.is_valid(coords.slice()), "coords {:?}, valid_zone: {:?} inside_valid: {:?}", coords.slice(), valid_zone, inside_valid);
                }
                if inside_valid {
                    prop_assert_eq!(invalid_count, 0);
                } else {
                    prop_assert_eq!(invalid_count, 1, "coords {:?}, valid_zone: {:?} inside_valid: {:?} invalid_zones: {:?}", coords.slice(), valid_zone, inside_valid, invalid_zones);
                }
            };
            /*
            let op = FixedAvgPool::new(p, true);
            prop_assert_eq!(op.generic(&d.view().into_dyn()).unwrap(), op.two_d(&d.view()).unwrap().into_dyn())
            */
        }
    }
}
