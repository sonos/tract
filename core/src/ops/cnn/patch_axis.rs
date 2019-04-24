use crate::internal::*;

use itertools::Itertools;
use std::ops::Range;

#[derive(Clone, Debug, new)]
pub struct PatchAxis {
    input_dim: usize,
    kernel_dim: usize,
    pad_before: usize,
    pad_after: usize,
    output_dim: usize,
    stride: usize,
    dilation: usize,
}

impl PatchAxis {
    fn valid_range(&self) -> Range<usize> {
        let min = self.pad_before.div_ceil(self.stride);
        let max = self.output_dim - self.pad_after.div_ceil(self.stride);
        min..max
    }

    fn invalid_at_left(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_before.saturating_sub(center_pos).div_ceil(self.dilation)
    }

    fn invalid_at_right(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_after.saturating_sub(self.input_dim - center_pos - 1).div_ceil(self.dilation)
    }

    fn make_invalid_zones(&self, range: Range<usize>) -> TVec<(Range<usize>, Option<TVec<bool>>)> {
        range
            .map(move |ix| (ix, (self.invalid_at_left(ix), self.invalid_at_right(ix))))
            .group_by(|&pair| pair.1)
            .into_iter()
            .map(move |(invalid, pairs)| {
                let (min, max) = pairs.map(|p| p.0).minmax().into_option().unwrap();
                let mut mask = tvec!(false; self.kernel_dim);
                for i in 0..invalid.0 {
                    mask[i] = true;
                }
                for i in 0..invalid.1 {
                    mask[self.kernel_dim - 1 - i] = true;
                }
                (min..max + 1, Some(mask))
            })
            .collect()
    }

    fn zones(&self) -> TVec<(Range<usize>, Option<TVec<bool>>)> {
        let mut zones = tvec!();
        let valid_range = self.valid_range();
        if valid_range.start > 0 {
            zones.extend(self.make_invalid_zones(0..valid_range.start));
        }
        if valid_range.start != valid_range.end {
            zones.push((valid_range.clone(), None));
        }
        if valid_range.end < self.output_dim {
            zones.extend(self.make_invalid_zones(valid_range.end..self.output_dim));
        }
        zones
    }
}

#[cfg(test)]
pub mod test {
    use super::super::DataFormat;
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
    use proptest::*;

    // • 0 1 2 3 4 • -> 3 -> (0) 1 2 3 (4)
    fn axis_5_3() -> PatchAxis {
        PatchAxis::new(5, 3, 1, 1, 5, 1, 1)
    }

    // • • 0 1 2 3 4 • -> 4 -> (0) (1) 2 3 (4)
    fn axis_5_4() -> PatchAxis {
        PatchAxis::new(5, 4, 2, 1, 5, 1, 1)
    }

    // • • 0 1 2 3 4 • • -> 4 -> (0) (1) 2 (3) (4)
    fn axis_5_5() -> PatchAxis {
        PatchAxis::new(5, 5, 2, 2, 5, 1, 1)
    }

    // • 0 1 2 3 4 • -> 3 -> (0) 2 (4)
    fn axis_5_3_s2() -> PatchAxis {
        PatchAxis::new(5, 3, 1, 1, 3, 2, 1)
    }

    // • • 0 1 2 3 4 • • -> 3x2 -> (0) (1) 2 (3) (4)
    fn axis_5_3_d2() -> PatchAxis {
        PatchAxis::new(5, 3, 2, 2, 5, 1, 2)
    }

    // 0 1 2 3 4 5 6 7 8 9 -> 2 -> 0 3 6
    fn axis_10_2_s3_valid() -> PatchAxis {
        PatchAxis::new(10, 2, 0, 0, 3, 1, 1)
    }

    #[test]
    fn axis_valid_ranges() {
        assert_eq!(axis_5_3().valid_range(), 1..4);
        assert_eq!(axis_5_4().valid_range(), 2..4);
        assert_eq!(axis_5_5().valid_range(), 2..3);
        assert_eq!(axis_5_3_s2().valid_range(), 1..2);
        assert_eq!(axis_5_3_d2().valid_range(), 2..3);
    }

    #[test]
    fn axis_invalid_at_left() {
        assert_eq!(axis_5_3().invalid_at_left(0), 1);
        assert_eq!(axis_5_3().invalid_at_left(1), 0);
        assert_eq!(axis_5_3().invalid_at_left(2), 0);

        assert_eq!(axis_5_4().invalid_at_left(0), 2);
        assert_eq!(axis_5_4().invalid_at_left(1), 1);
        assert_eq!(axis_5_4().invalid_at_left(2), 0);

        assert_eq!(axis_5_5().invalid_at_left(0), 2);
        assert_eq!(axis_5_5().invalid_at_left(1), 1);
        assert_eq!(axis_5_5().invalid_at_left(2), 0);

        assert_eq!(axis_5_3_d2().invalid_at_left(0), 1);
        assert_eq!(axis_5_3_d2().invalid_at_left(1), 1);
        assert_eq!(axis_5_3_d2().invalid_at_left(2), 0);
    }

    #[test]
    fn axis_invalid_at_right() {
        assert_eq!(axis_5_3().invalid_at_right(0), 0);
        assert_eq!(axis_5_3().invalid_at_right(3), 0);
        assert_eq!(axis_5_3().invalid_at_right(4), 1);

        assert_eq!(axis_5_4().invalid_at_right(0), 0);
        assert_eq!(axis_5_4().invalid_at_right(3), 0);
        assert_eq!(axis_5_4().invalid_at_right(4), 1);

        assert_eq!(axis_5_5().invalid_at_right(0), 0);
        assert_eq!(axis_5_5().invalid_at_right(3), 1);
        assert_eq!(axis_5_5().invalid_at_right(4), 2);
    }

    #[test]
    fn axis_5_3_zones() {
        let zones = axis_5_3().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..1, Some(tvec!(true, false, false))),
                (1..4, None),
                (4..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_s2_zones() {
        let zones = axis_5_3_s2().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..1, Some(tvec!(true, false, false))),
                (1..2, None),
                (2..3, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_d2_zones() {
        let zones = axis_5_3_d2().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..2, Some(tvec!(true, false, false))),
                (2..3, None),
                (3..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_10_2_s3_valid_zones() {
        let zones = axis_10_2_s3_valid().zones();
        assert_eq!(zones, tvec!((0..3, None),));
    }

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<isize> {
        let patch = PatchSpec {
            input_shape: tvec![10; kdim.len()],
            kernel_shape: kdim.into(),
            dilations: dilations.into(),
            strides: tvec![1; kdim.len()],
            padding: PaddingSpec::Explicit(tvec![0; kdim.len()], tvec![0; kdim.len()]),
            input_storage_stride: 1,
            output_storage_stride: 1,
        }
        .into_patch();
        patch.data_field
    }

    #[test]
    #[ignore]
    fn test_field() {
        assert_eq!(field(&[3], &[1]), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2]), arr2(&[[0], [2], [4]]));
        assert_eq!(field(&[2, 2], &[1, 1]), arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(field(&[2, 2], &[2, 1]), arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]]));
    }
}
