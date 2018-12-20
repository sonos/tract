use crate::frame;

#[derive(Clone, Debug)]
pub struct SMatMul4x4;

impl frame::matmul::MatMul<f32> for SMatMul4x4 {
    #[inline(always)]
    fn mr(&self) -> usize {
        4
    }
    #[inline(always)]
    fn nr(&self) -> usize {
        4
    }
    #[inline(never)]
    fn kernel(&self, k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize) {
        unsafe {
            let mut ab = [[0.0f32; 4]; 4];
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
            let c = std::slice::from_raw_parts_mut(c, 4 + 3 * rsc);
            c[0 + 0 * rsc] = ab[0][0];
            c[1 + 0 * rsc] = ab[0][1];
            c[2 + 0 * rsc] = ab[0][2];
            c[3 + 0 * rsc] = ab[0][3];
            c[0 + 1 * rsc] = ab[1][0];
            c[1 + 1 * rsc] = ab[1][1];
            c[2 + 1 * rsc] = ab[1][2];
            c[3 + 1 * rsc] = ab[1][3];
            c[0 + 2 * rsc] = ab[2][0];
            c[1 + 2 * rsc] = ab[2][1];
            c[2 + 2 * rsc] = ab[2][2];
            c[3 + 2 * rsc] = ab[2][3];
            c[0 + 3 * rsc] = ab[3][0];
            c[1 + 3 * rsc] = ab[3][1];
            c[2 + 3 * rsc] = ab[3][2];
            c[3 + 3 * rsc] = ab[3][3];
        }
    }
}

#[derive(Clone, Debug)]
pub struct DMatMul4x2;

impl frame::matmul::MatMul<f64> for DMatMul4x2 {
    #[inline(always)]
    fn mr(&self) -> usize {
        4
    }
    #[inline(always)]
    fn nr(&self) -> usize {
        2
    }
    #[inline(never)]
    fn kernel(&self, k: usize, a: *const f64, b: *const f64, c: *mut f64, rsc: usize) {
        unsafe {
            let mut ab = [[0.0f64; 2]; 4];
            for i in 0..k {
                let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                let b = std::slice::from_raw_parts(b.offset(4 * i as isize), 4);
                ab[0][0] += a[0] * b[0];
                ab[0][1] += a[0] * b[1];
                ab[1][0] += a[1] * b[0];
                ab[1][1] += a[1] * b[1];
                ab[2][0] += a[2] * b[0];
                ab[2][1] += a[2] * b[1];
                ab[3][0] += a[3] * b[0];
                ab[3][1] += a[3] * b[1];
            }
            let c = std::slice::from_raw_parts_mut(c, 2 + 3 * rsc);
            c[0 + 0 * rsc] = ab[0][0];
            c[1 + 0 * rsc] = ab[0][1];
            c[0 + 1 * rsc] = ab[1][0];
            c[1 + 1 * rsc] = ab[1][1];
            c[0 + 2 * rsc] = ab[2][0];
            c[1 + 2 * rsc] = ab[2][1];
            c[0 + 3 * rsc] = ab[3][0];
            c[1 + 3 * rsc] = ab[3][1];
        }
    }
}
