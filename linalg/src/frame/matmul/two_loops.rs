use super::MatMul;
use std::ops::{Add, Mul};
use num_traits::Zero;

#[inline(always)]
pub fn pack_panel_a<T:Copy>(
    pa: *mut T,
    a: *const T,
    rsa: isize,
    csa: isize,
    mr: usize,
    rows: usize,
    k: usize,
) {
    for i in 0..k {
        for j in 0..rows {
            unsafe {
                *pa.offset((i * mr + j) as isize) = *a.offset(i as isize * csa + j as isize * rsa)
            }
        }
    }
}

#[inline(always)]
pub fn pack_panel_b<T:Copy>(
    pb: *mut T,
    b: *const T,
    rsb: isize,
    csb: isize,
    nr: usize,
    cols: usize,
    k: usize,
) {
    for i in 0..k {
        for j in 0..cols {
            unsafe {
                *pb.offset((i * nr + j) as isize) = *b.offset(j as isize * csb + i as isize * rsb)
            }
        }
    }
}

pub fn pack_b<T: Copy>(pb: *mut T, b: *const T, rsb: isize, csb: isize, nr: usize, k: usize, n: usize) {
}

pub fn pack_a<T:Copy>(pa: *mut T, a: *const T, rsa: isize, csa: isize, mr: usize, m: usize, k: usize) {
    unsafe {
        for p in 0..(m / mr) {
            pack_panel_a(
                pa.offset((p * mr * k) as isize),
                a.offset((p * mr) as isize * rsa),
                rsa,
                csa,
                mr,
                mr,
                k,
            )
        }
        if m % mr != 0 {
            pack_panel_a(
                pa.offset((m / mr * mr * k) as isize),
                a.offset((m / mr * mr) as isize * rsa),
                rsa,
                csa,
                mr,
                m % mr,
                k,
            )
        }
    }
}

pub fn two_loops<K: MatMul<T>, T: Copy + Zero + Add + Mul>(
    m: usize,
    k: usize,
    n: usize,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    c: *mut T,
    rsc: isize,
    csc: isize,
) {
}

pub fn two_loops_prepacked<K: MatMul<T>, T: Copy + Mul + Add + Zero>(
    m: usize,
    k: usize,
    n: usize,
    pa: *const T,
    pb: *const T,
    c: *mut T,
    rsc: isize,
    csc: isize,
) {
}
