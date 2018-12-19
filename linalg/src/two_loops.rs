use super::Kernel;

#[inline(always)]
pub fn pack_panel_a(
    pa: *mut f32,
    a: *const f32,
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
pub fn pack_panel_b(
    pb: *mut f32,
    b: *const f32,
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

pub fn pack_b(pb: *mut f32, b: *const f32, rsb: isize, csb: isize, nr: usize, k: usize, n: usize) {
    unsafe {
        for p in 0..(n / nr) {
            pack_panel_b(
                pb.offset((p * nr * k) as isize),
                b.offset((p * nr) as isize * csb),
                rsb,
                csb,
                nr,
                nr,
                k,
            )
        }
        if n % nr != 0 {
            pack_panel_b(
                pb.offset((n / nr * nr * k) as isize),
                b.offset((n / nr * nr) as isize * csb),
                rsb,
                csb,
                nr,
                n % nr,
                k,
            )
        }
    }
}

pub fn pack_a(pa: *mut f32, a: *const f32, rsa: isize, csa: isize, mr: usize, m: usize, k: usize) {
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

pub fn two_loops<K: Kernel>(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    let mr = K::mr();
    let nr = K::nr();
    let mut pa = vec![0.0; mr * k];
    let mut pb = vec![0.0; K::packed_b_len(k, n)];
    pack_b(pb.as_mut_ptr(), b, rsb, csb, nr, k, n);
    let mut tmpc = vec![0.0; mr * nr];
    unsafe {
        for ia in 0..m / mr {
            pack_panel_a(
                pa.as_mut_ptr(),
                a.offset((mr * ia) as isize * rsa),
                rsa,
                csa,
                mr,
                mr,
                k,
            );
            for ib in 0..n / nr {
                K::kernel(
                    k,
                    pa.as_ptr(),
                    pb.as_ptr().offset((ib * k * nr) as isize),
                    c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                    rsc as usize, // FIXME
                );
            }
            if n % nr != 0 {
                K::kernel(
                    k,
                    pa.as_ptr(),
                    pb.as_ptr().offset((n / nr * k * nr) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..mr {
                    for x in 0..(n % nr) {
                        *c.offset(
                            (mr * ia + y) as isize * rsc + (x + n / nr * nr) as isize * csc,
                        ) = tmpc[y * nr + x];
                    }
                }
            }
        }
        if m % mr != 0 {
            let row = m - m % mr;
            pack_panel_a(
                pa.as_mut_ptr(),
                a.offset(row as isize * rsa),
                rsa,
                csa,
                mr,
                m % mr,
                k,
            );
            for ib in 0..n / nr {
                K::kernel(
                    k,
                    pa.as_ptr(),
                    pb.as_ptr().offset((ib * nr * k) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..(m % mr) {
                    for x in 0..nr {
                        *c.offset((y + row) as isize * rsc + (x + ib * nr) as isize * csc) =
                            tmpc[y * nr + x];
                    }
                }
            }
            if n % nr != 0 {
                K::kernel(
                    k,
                    pa.as_ptr(),
                    pb.as_ptr().offset((n / nr * nr * k) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..(m % mr) {
                    for x in 0..(n % nr) {
                        *c.offset((y + row) as isize * rsc + (x + n / nr * nr) as isize * csc) =
                            tmpc[y * nr + x];
                    }
                }
            }
        }
    }
}

pub fn two_loops_prepacked<K: Kernel>(
    m: usize,
    k: usize,
    n: usize,
    pa: *const f32,
    pb: *const f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    let mr = K::mr();
    let nr = K::nr();
    let mut tmpc = vec![0.0; mr * nr];
    unsafe {
        for ia in 0..m / mr {
            for ib in 0..n / nr {
                K::kernel(
                    k,
                    pa.offset((ia * k * mr) as isize),
                    pb.offset((ib * k * nr) as isize),
                    c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                    rsc as usize, // FIXME
                );
            }
            if n % nr != 0 {
                K::kernel(
                    k,
                    pa.offset((ia * k * mr) as isize),
                    pb.offset((n / nr * k * nr) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..mr {
                    for x in 0..(n % nr) {
                        *c.offset(
                            (mr * ia + y) as isize * rsc + (x + n / nr * nr) as isize * csc,
                        ) = tmpc[y * nr + x];
                    }
                }
            }
        }
        if m % mr != 0 {
            for ib in 0..n / nr {
                K::kernel(
                    k,
                    pa.offset((m / mr * mr * k) as isize),
                    pb.offset((ib * nr * k) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..(m % mr) {
                    for x in 0..nr {
                        *c.offset(
                            (y + m / mr * mr) as isize * rsc + (x + ib * nr) as isize * csc,
                        ) = tmpc[y * nr + x];
                    }
                }
            }
            if n % nr != 0 {
                K::kernel(
                    k,
                    pa.offset((m / mr * mr * k) as isize),
                    pb.offset((n / nr * nr * k) as isize),
                    tmpc.as_mut_ptr(),
                    nr,
                );
                for y in 0..(m % mr) {
                    for x in 0..(n % nr) {
                        *c.offset(
                            (y + m / mr * mr) as isize * rsc + (x + n / nr * nr) as isize * csc,
                        ) = tmpc[y * nr + x];
                    }
                }
            }
        }
    }
}
