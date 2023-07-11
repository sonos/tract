use crate::internal::*;

use std::ops::Range;
use tract_itertools::Itertools;

#[derive(Clone, Debug, new, PartialEq, Eq)]
pub struct Region {
    pub range: Range<usize>,
    pub mask: Option<TVec<bool>>,
}

#[derive(Clone, Debug, new, PartialEq, Eq)]
pub struct PatchAxis {
    pub input_dim: usize,
    pub kernel_dim: usize,
    pub pad_before: usize,
    pub pad_after: usize,
    pub output_dim: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl PatchAxis {
    fn valid_range(&self) -> Option<Range<usize>> {
        let field = (self.kernel_dim - 1) * self.dilation + 1;
        if field > self.input_dim {
            return None;
        }
        let min = self.pad_before.divceil(self.stride);
        let max = (self.input_dim + self.pad_before).saturating_sub(field) / self.stride;
        if max >= min {
            Some(min..(max + 1))
        } else {
            None
        }
    }

    fn invalid_at_left(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_before.saturating_sub(center_pos).divceil(self.dilation).min(self.kernel_dim)
    }

    fn invalid_at_right(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        let last_valid = self.input_dim + self.pad_before;
        let valid = last_valid.saturating_sub(center_pos).divceil(self.dilation);
        self.kernel_dim.saturating_sub(valid)
    }

    fn make_invalid_regions(&self, range: Range<usize>) -> TVec<Region> {
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
                Region::new(min..max + 1, Some(mask))
            })
            .collect()
    }

    pub fn regions(&self) -> TVec<Region> {
        let mut regions = tvec!();
        if let Some(valid_range) = self.valid_range() {
            if valid_range.start > 0 {
                regions.extend(self.make_invalid_regions(0..valid_range.start));
            }
            if valid_range.start != valid_range.end {
                regions.push(Region::new(valid_range.clone(), None));
            }
            if valid_range.end < self.output_dim {
                regions.extend(self.make_invalid_regions(valid_range.end..self.output_dim));
            }
        } else {
            regions.extend(self.make_invalid_regions(0..self.output_dim));
        }
        regions
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

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
        PatchAxis::new(10, 2, 0, 0, 3, 3, 1)
    }

    #[test]
    fn axis_valid_ranges() {
        assert_eq!(axis_5_3().valid_range(), Some(1..4));
        assert_eq!(axis_5_4().valid_range(), Some(2..4));
        assert_eq!(axis_5_5().valid_range(), Some(2..3));
        assert_eq!(axis_5_3_s2().valid_range(), Some(1..2));
        assert_eq!(axis_5_3_d2().valid_range(), Some(2..3));
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
    fn axis_5_3_regions() {
        let regions = axis_5_3().regions();
        assert_eq!(
            regions,
            tvec!(
                Region::new(0..1, Some(tvec!(true, false, false))),
                Region::new(1..4, None),
                Region::new(4..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_s2_regions() {
        let regions = axis_5_3_s2().regions();
        assert_eq!(
            regions,
            tvec!(
                Region::new(0..1, Some(tvec!(true, false, false))),
                Region::new(1..2, None),
                Region::new(2..3, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_d2_regions() {
        let regions = axis_5_3_d2().regions();
        assert_eq!(
            regions,
            tvec!(
                Region::new(0..2, Some(tvec!(true, false, false))),
                Region::new(2..3, None),
                Region::new(3..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_10_2_s3_valid_regions() {
        let regions = axis_10_2_s3_valid().regions();
        assert_eq!(regions, tvec!(Region::new(0..3, None),));
    }

    #[test]
    fn axis_7_3_s2_regions() {
        // • 0 1 2 3 4 5 6 • -> 3 -> (0) 2 4 (6)
        let regions = PatchAxis::new(7, 3, 1, 1, 4, 2, 1).regions();
        assert_eq!(
            regions,
            tvec!(
                Region::new(0..1, Some(tvec!(true, false, false))),
                Region::new(1..3, None),
                Region::new(3..4, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_2_s2_regions() {
        // • 0 1 2 3 4 • -> 2 -> (0) 2 4
        let regions = PatchAxis::new(5, 2, 1, 1, 3, 2, 1).regions();
        assert_eq!(
            regions,
            tvec!(Region::new(0..1, Some(tvec!(true, false))), Region::new(1..3, None),)
        );
    }

    #[test]
    fn axis_28_3_very_padded_regions() {
        // • • 0 1 2 3 ... 26 27 • • -> 2 -> (-1) (0) (1) 2 3 4 ... 26 (27) (28) (29)
        let regions = PatchAxis::new(28, 3, 2, 2, 30, 1, 1).regions();
        assert_eq!(
            regions,
            tvec!(
                Region::new(0..1, Some(tvec!(true, true, false))),
                Region::new(1..2, Some(tvec!(true, false, false))),
                Region::new(2..28, None),
                Region::new(28..29, Some(tvec!(false, false, true))),
                Region::new(29..30, Some(tvec!(false, true, true))),
            )
        );
    }

    #[test]
    fn axis_7_1_s2_regions() {
        // 0 1 2 3 4 5 6 -> 1 -> 0 2 4 6
        let regions = PatchAxis::new(7, 1, 0, 0, 4, 2, 1).regions();
        assert_eq!(regions, tvec!(Region::new(0..4, None),));
    }

    #[test]
    fn axis_1_2_regions() {
        // 0 -> 2 -> (0)
        let regions = PatchAxis::new(1, 2, 0, 1, 1, 1, 1).regions();
        assert_eq!(regions, tvec!(Region::new(0..1, Some(tvec!(false, true))),));
    }

    #[test]
    fn axis_dnn_left_pad() {
        let regions = PatchAxis::new(1, 1, 2, 0, 3, 1, 1).regions();
        assert_eq!(regions, tvec!(Region::new(0..2, Some(tvec!(true))), Region::new(2..3, None)));
    }
}
