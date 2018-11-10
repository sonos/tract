use super::{DataFormat, DataShape, PaddingSpec};
use ndarray::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct Patch {
    pub dilations: TVec<usize>,
    pub kernel_spatial_shape: TVec<usize>,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    pub kernel_strides: TVec<usize>,
    pub input_shape: DataShape<usize, TVec<usize>>,
    pub output_spatial_shape: TVec<usize>,
    pub data_field: Array2<usize>,
    pub standard_layout_data_field: Vec<isize>,
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
        use ops::nn::padding::ComputedPaddedDim;
        let input_shape = data_fmt.shape(input_full_shape);
        let ComputedPaddedDim {
            pad_after,
            pad_before,
            output,
        } = padding.compute(
            input_shape.hw_dims(),
            &kernel_spatial_shape,
            &*dilations,
            &*kernel_strides,
        );

        let data_field: Vec<usize> = ::ndarray::indices(&*kernel_spatial_shape)
            .into_iter()
            .flat_map(|coords| {
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * dilations[ix]).wrapping_sub(pad_before[ix]))
            }).collect();
        let data_field = Array2::from_shape_vec(
            (
                kernel_spatial_shape.iter().cloned().product(),
                kernel_spatial_shape.len(),
            ),
            data_field,
        ).unwrap();

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
            }).collect();

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
            standard_layout_data_field,
        }
    }

    pub fn output_full_shape(&self, channels: usize) -> TVec<usize> {
        let mut v = self.input_shape.shape.clone();
        v[self.input_shape.c_axis()] = channels;
        v[self.input_shape.hw_axes()].copy_from_slice(&self.output_spatial_shape);
        v
    }

    pub fn wrap<'i, 'p, T: Datum>(
        &'p self,
        input: &'i ArrayViewD<'i, T>,
    ) -> PatchVisitor<'i, 'p, T> {
        let valid = !self.padded; //input.is_standard_layout() && !self.padded;
        let mut fast_strides = input.strides().to_vec();
        fast_strides[self.input_shape.hw_axes()]
            .iter_mut()
            .zip(self.kernel_strides.iter())
            .for_each(|(a, &b)| *a *= b as isize);
        PatchVisitor {
            patch: &self,
            input,
            valid,
            fast_strides,
        }
    }
}

#[derive(Debug)]
pub struct PatchVisitor<'i, 'p, T: Datum> {
    patch: &'p Patch,
    input: &'i ArrayViewD<'i, T>,
    valid: bool,
    fast_strides: Vec<isize>, // kernel strides * storage strides
}

impl<'i, 'p, T: Datum> PatchVisitor<'i, 'p, T> {
    pub fn at<'v>(&'p self, coords: &[usize]) -> PatchIterator<'i, 'p, 'v, T>
    where
        'i: 'v,
        'p: 'v,
    {
        if self.valid {
            let center = coords
                .iter()
                .zip(self.fast_strides.iter())
                .map(|(&a, &b)| b * a as isize)
                .sum();
            PatchIterator::Fast(FastPatchIterator {
                visitor: &self,
                ptr: self.input.as_ptr(),
                center,
                item: 0,
            })
        } else {
            let mut input_patch_center = coords.to_vec();
            input_patch_center[self.patch.input_shape.hw_axes()]
                .iter_mut()
                .zip(self.patch.kernel_strides.iter())
                .for_each(|(a, &b)| *a *= b as usize);
            let input_patch_current = vec![0; coords.len()];
            PatchIterator::Safe(SafePatchIterator {
                visitor: self,
                item: 0,
                input_patch_center,
                input_patch_current,
            })
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

pub enum PatchIterator<'i: 'v, 'p: 'v, 'v, T: Datum> {
    Fast(FastPatchIterator<'i, 'p, 'v, T>),
    Safe(SafePatchIterator<'i, 'p, 'v, T>),
}

impl<'i: 'v, 'p: 'v, 'v, T: Datum + PartialEq> Iterator for PatchIterator<'p, 'i, 'v, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        match self {
            &mut PatchIterator::Fast(ref mut it) => it.next(),
            &mut PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

pub struct FastPatchIterator<'i: 'v, 'p: 'v, 'v, T: Datum> {
    visitor: &'v PatchVisitor<'i, 'p, T>,
    ptr: *const T,
    center: isize,
    item: usize,
}

impl<'i: 'v, 'p: 'v, 'v, T: Datum + PartialEq> Iterator for FastPatchIterator<'i, 'p, 'v, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.visitor.patch.standard_layout_data_field.len() {
            return None;
        }
        unsafe {
            let position = self.center + self
                .visitor
                .patch
                .standard_layout_data_field
                .get_unchecked(self.item);
            self.item += 1;
            Some(Some(*(self.ptr.offset(position))))
        }
    }
}

pub struct SafePatchIterator<'i: 'v, 'p: 'v, 'v, T: Datum> {
    visitor: &'v PatchVisitor<'i, 'p, T>,
    item: usize,
    input_patch_center: Vec<usize>,
    input_patch_current: Vec<usize>,
}

impl<'i: 'v, 'p: 'v, 'v, T: Datum + PartialEq> Iterator for SafePatchIterator<'i, 'p, 'v, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.visitor.patch.data_field.rows() {
            return None;
        }
        let img_offset = self.visitor.patch.data_field.row(self.item);
        self.item += 1;

        (&mut *self.input_patch_current).copy_from_slice(&self.input_patch_center);
        self.input_patch_current[self.visitor.patch.input_shape.hw_axes()]
            .iter_mut()
            .zip(img_offset.iter())
            .for_each(|(x, &i)| *x = x.wrapping_add(i));
        Some(self.visitor.input.get(&*self.input_patch_current).cloned())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ops::nn::DataFormat::NCHW;

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

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<usize> {
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
        assert_eq!(
            field(&[2, 2], &[1, 1]),
            arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]])
        );
        assert_eq!(
            field(&[2, 2], &[2, 1]),
            arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]])
        );
    }
}
