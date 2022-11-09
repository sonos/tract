use crate::prelude::*;
use ndarray::Dimension;

pub(crate) unsafe fn scatter_contig_data<T: Datum>(
    mut src: *const T,
    dst: *mut T,
    dst_len_and_strides: &[(usize, usize)],
) {
    match *dst_len_and_strides {
        [(len_a, stride_a)] => {
            for a in 0..len_a {
                *dst.add(a * stride_a) = (*src).clone();
                src = src.offset(1);
            }
        }
        [(len_a, stride_a), (len_b, stride_b)] => {
            for a in 0..len_a {
                for b in 0..len_b {
                    *dst.add(a * stride_a + b * stride_b) = (*src).clone();
                    src = src.offset(1);
                }
            }
        }
        [(len_a, stride_a), (len_b, stride_b), (len_c, stride_c)] => {
            for a in 0..len_a {
                for b in 0..len_b {
                    for c in 0..len_c {
                        *dst.add(a * stride_a + b * stride_b + c * stride_c) =
                            (*src).clone();
                        src = src.offset(1);
                    }
                }
            }
        }
        [(len_a, stride_a), (len_b, stride_b), (len_c, stride_c), (len_d, stride_d)] => {
            for a in 0..len_a {
                for b in 0..len_b {
                    for c in 0..len_c {
                        for d in 0..len_d {
                            *dst.add(a * stride_a + b * stride_b + c * stride_c + d * stride_d) = (*src).clone();
                            src = src.offset(1);
                        }
                    }
                }
            }
        }
        [(len_a, stride_a), (len_b, stride_b), (len_c, stride_c), (len_d, stride_d), (len_e, stride_e)] => {
            for a in 0..len_a {
                for b in 0..len_b {
                    for c in 0..len_c {
                        for d in 0..len_d {
                            for e in 0..len_e {
                                *dst.add(a * stride_a
                                        + b * stride_b
                                        + c * stride_c
                                        + d * stride_d
                                        + e * stride_e) = (*src).clone();
                                src = src.offset(1);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            let shape: TVec<usize> = dst_len_and_strides.iter().map(|pair| pair.0).collect();
            for coords in ndarray::indices(&*shape) {
                let offset = coords
                    .slice()
                    .iter()
                    .zip(dst_len_and_strides.iter())
                    .map(|(x, (_len, stride))| x * stride)
                    .sum::<usize>();
                *dst.add(offset) = (*src).clone();
                src = src.offset(1);
            }
        }
    }
}
