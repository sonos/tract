use ndarray::prelude::*;

use super::{ DataFormat, DataShape, PaddingSpec};
use tensor::Datum;

#[derive(Debug, Clone)]
pub struct Patch {
    pub dilations: Vec<usize>,
    pub kernel_spatial_shape: Vec<usize>,
    pub pad_before: Vec<usize>,
    pub pad_after: Vec<usize>,
    pub padded: bool,
    pub strides: Vec<usize>,
    pub input_shape: DataShape<usize, Vec<usize>>,
    pub output_spatial_shape: Vec<usize>,
    pub data_field: Array2<usize>,
    pub standard_layout_data_field: Vec<isize>,
}

impl Patch {
    pub fn new(
        data_fmt: DataFormat,
        dilations: Vec<usize>,
        kernel_spatial_shape: Vec<usize>,
        padding: &PaddingSpec,
        strides: Vec<usize>,
        input_full_shape: Vec<usize>,
    ) -> Patch {
        let input_shape = data_fmt.shape(input_full_shape);
        let (output_spatial_shape, pad_before, pad_after) = padding.compute(
            input_shape.hw_dims(),
            &kernel_spatial_shape,
            &*dilations,
            &*strides,
        );

        let data_field: Vec<usize> = ::ndarray::indices(&*kernel_spatial_shape).into_iter()
            .flat_map(|coords| {
                coords.slice().to_vec().into_iter().enumerate().map(|(ix,c)| (c * dilations[ix]).wrapping_sub(pad_before[ix]))
            }).collect();
        let data_field = Array2::from_shape_vec(
            (
                kernel_spatial_shape.iter().cloned().product(),
                kernel_spatial_shape.len(),
            ),
            data_field,
        ).unwrap();

        let mut input_layout_strides:Vec<usize> = vec!(1);
        for dim in input_shape.shape.iter().skip(1).rev() {
            let previous = input_layout_strides.last().cloned().unwrap_or(1);
            input_layout_strides.push(dim*previous);
        }
        input_layout_strides.reverse();
        let standard_layout_data_field:Vec<isize> = data_field.outer_iter().map(|coords| {
            coords.iter().zip(input_layout_strides.iter().skip(input_shape.h_axis())).map(|(&a,&b)| (a as isize*b as isize)).sum()
        }).collect();

        Patch {
            dilations,
            kernel_spatial_shape,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            strides,
            input_shape,
            output_spatial_shape,
            data_field,
            standard_layout_data_field,
        }
    }

    pub fn output_full_shape(&self, channels: usize) -> Vec<usize> {
        let mut v = self.input_shape.shape.clone();
        v[self.input_shape.c_axis()] = channels;
        v[self.input_shape.hw_axes()].copy_from_slice(&self.output_spatial_shape);
        v
    }

    pub fn patch_data_iter<'a, 'b, 'c, 'd: 'a, T: Datum>(
        &'b self,
        input: &'a ArrayViewD<'d, T>,
        coords: &'c [usize],
    ) -> FastIterator<'a, 'b, 'd, T> {
        let mut center_coords = coords.to_vec();
        izip!(center_coords[self.input_shape.hw_axes()].iter_mut(), self.strides.iter()).for_each(|(a,&b)| *a*=b);

        FastIterator {
            input:input.as_slice().unwrap(),
            field: &*self.standard_layout_data_field,
            center: input.strides().iter().zip(center_coords.iter()).map(|(&a, &b)| a*b as isize).sum(),
            item:0,
            phantom: ::std::marker::PhantomData
        }

        /*
        let offset = if input.is_standard_layout() && !self.padded {
            Some(input.strides().iter().zip(center_coords.iter()).map(|(&a, &b)| a*b as isize).sum())
        } else {
            None
        };
        PatchIterator {
            patch: &*self,
            item: 0,
            input,
            center_coords,
            full_coords: vec![0; coords.len()],
            valid_center_offset: offset
        }
        */
    }
}

pub struct PatchVisitor<'a, 'b, 'd: 'a, T: Datum> {
    input: &'a ArrayViewD<'d, T>,
    patch: Patch,
    center_coords: Vec<usize>,
    full_coords: Vec<usize>,
}

pub struct FastIterator<'a, 'b, 'd: 'a, T: Datum> {
    input: &'a[T],
    field: &'b[isize],
    center: isize,
    item: usize,
    phantom: ::std::marker::PhantomData<&'d ()>,
}

impl<'a, 'b, 'd, T: Datum+PartialEq> Iterator for FastIterator<'a, 'b, 'd, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.field.len() {
            return None;
        }
        unsafe {
            let position = self.center + self.field.get_unchecked(self.item);
            self.item += 1;
            return Some(Some(*self.input.get_unchecked(position as usize)))
        }
    }
}

pub struct SafeIterator<'a, 'b, 'd: 'a, T: Datum> {
    input: &'a ArrayViewD<'d, T>,
    patch: &'b Patch,
    item: usize,
    center_coords: Vec<usize>,
    full_coords: Vec<usize>,
    valid_center_offset: Option<isize>,
}

impl<'a, 'b, 'd, T: Datum+PartialEq> Iterator for SafeIterator<'a, 'b, 'd, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.patch.data_field.rows() {
            return None;
        }
        self.item += 1;
        if let Some(center_offset) = self.valid_center_offset {
            unsafe {
                let position = center_offset + self.patch.standard_layout_data_field.get_unchecked(self.item-1);
                return Some(Some(*self.input.as_slice().unwrap().get_unchecked(position as usize)))
            }
        };
        let img_offset = self.patch.data_field.row(self.item-1);

        (&mut *self.full_coords).copy_from_slice(&self.center_coords);
        self.full_coords[self.patch.input_shape.hw_axes()]
            .iter_mut()
            .zip(img_offset.iter())
            .for_each(|(x, &i)| *x = x.wrapping_add(i));
        Some(self.input.get(&*self.full_coords).cloned())
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
            vec![dilation],
            vec![kdim],
            &PaddingSpec::Explicit(vec![pad_before], vec![bad_after]),
            vec![stride],
            vec![1, 1, input],
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
            dilations.to_vec(),
            kdim.to_vec(),
            &PaddingSpec::Explicit(vec![0; kdim.len()], vec![0; kdim.len()]),
            vec![1; kdim.len()],
            vec![10; kdim.len() + 2],
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
