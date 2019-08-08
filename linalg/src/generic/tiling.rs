use crate::frame::tiling_kernel::LinearSpec::*;
use crate::frame::tiling_kernel::TileStorageSpec::*;
use crate::frame::tiling_kernel::*;

#[derive(Copy, Clone, Debug)]
pub struct STiling4x4;

impl TilingKer<f32> for STiling4x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "generic"
    }
    #[inline(always)]
    fn mr() -> usize {
        4
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &TileOpSpec<f32>) -> isize {
        unsafe {
            let mut ab = [[0.0f32; 4]; 4];
            match (*spec.a, *spec.b, *spec.linear) {
                (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let b = std::slice::from_raw_parts(b.offset(4 * i as isize), 4);
                        ab[0][0] += a[0] * b[0];
                        ab[0][1] += a[0] * b[1];
                        ab[0][2] += a[0] * b[2];
                        ab[0][3] += a[0] * b[3];
                        ab[1][0] += a[1] * b[0];
                        ab[1][1] += a[1] * b[1];
                        ab[1][2] += a[1] * b[2];
                        ab[1][3] += a[1] * b[3];
                        ab[2][0] += a[2] * b[0];
                        ab[2][1] += a[2] * b[1];
                        ab[2][2] += a[2] * b[2];
                        ab[2][3] += a[2] * b[3];
                        ab[3][0] += a[3] * b[0];
                        ab[3][1] += a[3] * b[1];
                        ab[3][2] += a[3] * b[2];
                        ab[3][3] += a[3] * b[3];
                    }
                }
                (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
                    let pb0 = *(col_ptrs.offset(0));
                    let pb1 = *(col_ptrs.offset(1));
                    let pb2 = *(col_ptrs.offset(2));
                    let pb3 = *(col_ptrs.offset(3));
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                        let offset = *row_byte_offsets.offset(i as isize) / 4;
                        let b0 = *(pb0.offset(offset));
                        let b1 = *(pb1.offset(offset));
                        let b2 = *(pb2.offset(offset));
                        let b3 = *(pb3.offset(offset));
                        ab[0][0] += a[0] * b0;
                        ab[0][1] += a[0] * b1;
                        ab[0][2] += a[0] * b2;
                        ab[0][3] += a[0] * b3;
                        ab[1][0] += a[1] * b0;
                        ab[1][1] += a[1] * b1;
                        ab[1][2] += a[1] * b2;
                        ab[1][3] += a[1] * b3;
                        ab[2][0] += a[2] * b0;
                        ab[2][1] += a[2] * b1;
                        ab[2][2] += a[2] * b2;
                        ab[2][3] += a[2] * b3;
                        ab[3][0] += a[3] * b0;
                        ab[3][1] += a[3] * b1;
                        ab[3][2] += a[3] * b2;
                        ab[3][3] += a[3] * b3;
                    }
                }
                _ => return 1,
            }
            let mut pnl = spec.non_linear;
            loop {
                if pnl.is_null() {
                    break;
                }
                match *pnl {
                    NonLinearSpec::Done => break,
                    NonLinearSpec::AddC => {
                        match *spec.c {
                            Strides { ptr: c, row_byte_stride, col_byte_stride } => {
                                let rsc = row_byte_stride as usize / 4;
                                let csc = col_byte_stride as usize / 4;
                                let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
                                ab[0][0] += c[0 * csc + 0 * rsc];
                                ab[0][1] += c[1 * csc + 0 * rsc];
                                ab[0][2] += c[2 * csc + 0 * rsc];
                                ab[0][3] += c[3 * csc + 0 * rsc];
                                ab[1][0] += c[0 * csc + 1 * rsc];
                                ab[1][1] += c[1 * csc + 1 * rsc];
                                ab[1][2] += c[2 * csc + 1 * rsc];
                                ab[1][3] += c[3 * csc + 1 * rsc];
                                ab[2][0] += c[0 * csc + 2 * rsc];
                                ab[2][1] += c[1 * csc + 2 * rsc];
                                ab[2][2] += c[2 * csc + 2 * rsc];
                                ab[2][3] += c[3 * csc + 2 * rsc];
                                ab[3][0] += c[0 * csc + 3 * rsc];
                                ab[3][1] += c[1 * csc + 3 * rsc];
                                ab[3][2] += c[2 * csc + 3 * rsc];
                                ab[3][3] += c[3 * csc + 3 * rsc];
                            }
                            _ => return 1
                        }
                    }
                    _ => return 1
                }
                pnl = pnl.add(1);
            }
            match *spec.c {
                Strides { ptr: c, row_byte_stride, col_byte_stride } => {
                    let rsc = row_byte_stride as usize / 4;
                    let csc = col_byte_stride as usize / 4;
                    let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
                    c[0 * csc + 0 * rsc] = ab[0][0];
                    c[1 * csc + 0 * rsc] = ab[0][1];
                    c[2 * csc + 0 * rsc] = ab[0][2];
                    c[3 * csc + 0 * rsc] = ab[0][3];
                    c[0 * csc + 1 * rsc] = ab[1][0];
                    c[1 * csc + 1 * rsc] = ab[1][1];
                    c[2 * csc + 1 * rsc] = ab[1][2];
                    c[3 * csc + 1 * rsc] = ab[1][3];
                    c[0 * csc + 2 * rsc] = ab[2][0];
                    c[1 * csc + 2 * rsc] = ab[2][1];
                    c[2 * csc + 2 * rsc] = ab[2][2];
                    c[3 * csc + 2 * rsc] = ab[2][3];
                    c[0 * csc + 3 * rsc] = ab[3][0];
                    c[1 * csc + 3 * rsc] = ab[3][1];
                    c[2 * csc + 3 * rsc] = ab[3][2];
                    c[3 * csc + 3 * rsc] = ab[3][3];
                }
                _ => return 1,
            }
        }
        return 0
    }
}

#[cfg(test)]
#[derive(Copy, Clone, Debug)]
pub struct STilingTest3x2;

#[cfg(test)]
impl TilingKer<f32> for STilingTest3x2 {
    #[inline(always)]
    fn name() -> &'static str {
        "generic-test-3x2"
    }
    #[inline(always)]
    fn mr() -> usize {
        3
    }
    #[inline(always)]
    fn nr() -> usize {
        2
    }
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &TileOpSpec<f32>) -> isize {
        unsafe {
            let mut ab = [[0.0f32; 2]; 3];
            match (*spec.a, *spec.b, *spec.linear) {
                (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(3 * i as isize), 3);
                        let b = std::slice::from_raw_parts(b.offset(2 * i as isize), 2);
                        ab[0][0] += a[0] * b[0];
                        ab[0][1] += a[0] * b[1];
                        ab[1][0] += a[1] * b[0];
                        ab[1][1] += a[1] * b[1];
                        ab[2][0] += a[2] * b[0];
                        ab[2][1] += a[2] * b[1];
                    }
                }
                (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
                    let pb0 = *(col_ptrs.offset(0));
                    let pb1 = *(col_ptrs.offset(1));
                    for i in 0..k {
                        let a = std::slice::from_raw_parts(a.offset(3 * i as isize), 3);
                        let offset = *row_byte_offsets.offset(i as isize) / 4;
                        let b0 = *(pb0.offset(offset));
                        let b1 = *(pb1.offset(offset));
                        ab[0][0] += a[0] * b0;
                        ab[0][1] += a[0] * b1;
                        ab[1][0] += a[1] * b0;
                        ab[1][1] += a[1] * b1;
                        ab[2][0] += a[2] * b0;
                        ab[2][1] += a[2] * b1;
                    }
                }
                _ => return 1,
            }
            let mut pnl = spec.non_linear;
            loop {
                if pnl.is_null() {
                    break;
                }
                match *pnl {
                    NonLinearSpec::Done => break,
                    NonLinearSpec::AddC => {
                        match *spec.c {
                            Strides { ptr: c, row_byte_stride, col_byte_stride } => {
                                let rsc = row_byte_stride as usize / 4;
                                let csc = col_byte_stride as usize / 4;
                                let c = std::slice::from_raw_parts_mut(c, 1 + 1 * csc + 2 * rsc);
                                ab[0][0] += c[0 * csc + 0 * rsc];
                                ab[0][1] += c[1 * csc + 0 * rsc];
                                ab[1][0] += c[0 * csc + 1 * rsc];
                                ab[1][1] += c[1 * csc + 1 * rsc];
                                ab[2][0] += c[0 * csc + 2 * rsc];
                                ab[2][1] += c[1 * csc + 2 * rsc];
                            }
                            _ => return 1
                        }
                    }
                    _ => return 1
                }
                pnl = pnl.add(1);
            }
            match *spec.c {
                Strides { ptr: c, row_byte_stride, col_byte_stride } => {
                    let rsc = row_byte_stride as usize / 4;
                    let csc = col_byte_stride as usize / 4;
                    let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
                    c[0 * csc + 0 * rsc] = ab[0][0];
                    c[1 * csc + 0 * rsc] = ab[0][1];
                    c[0 * csc + 1 * rsc] = ab[1][0];
                    c[1 * csc + 1 * rsc] = ab[1][1];
                    c[0 * csc + 2 * rsc] = ab[2][0];
                    c[1 * csc + 2 * rsc] = ab[2][1];
                }
                _ => return 1,
            }
        }
        return 0
    }
}

#[cfg(test)]
mod test_3_2 {
    tile_kernel_tests!(true, crate::generic::tiling::STilingTest3x2, f32);
    tile_frame_tests!(true, crate::generic::tiling::STilingTest3x2);
}

#[cfg(test)]
mod test {
    tile_kernel_tests!(true, crate::generic::STiling4x4, f32);
    tile_frame_tests!(true, crate::generic::STiling4x4);
}
