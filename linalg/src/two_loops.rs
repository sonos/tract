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

pub fn packed_b_panels(nr: usize, n: usize) -> usize {
    (n + nr - 1) / nr
}

pub fn pack_b(pb: *mut f32, b: *const f32, rsb: isize, csb: isize, nr: usize, n: usize, k: usize) {
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
    let mut pbs = (0..n / nr)
        .map(|i| {
            let mut pb = vec![0.0; nr * k];
            unsafe {
                pack_panel_b(
                    pb.as_mut_ptr(),
                    b.offset((nr * i) as isize * csb),
                    rsb,
                    csb,
                    nr,
                    nr,
                    k,
                );
            }
            pb
        })
        .collect::<Vec<_>>();
    if n % nr != 0 {
        let mut pb = vec![0.0; nr * k];
        unsafe {
            pack_panel_b(
                pb.as_mut_ptr(),
                b.offset((n / nr * nr) as isize * csb),
                rsb,
                csb,
                nr,
                n % nr,
                k,
            );
        }
        pbs.push(pb);
    }
    let mut tmpc = vec![0.0; mr * nr];
    for ia in 0..m / mr {
        unsafe {
            pack_panel_a(
                pa.as_mut_ptr(),
                a.offset((mr * ia) as isize * rsa),
                rsa,
                csa,
                mr,
                mr,
                k,
            );
        }
        for ib in 0..n / nr {
            unsafe {
                K::kernel(
                    k,
                    pa.as_ptr(),
                    pbs[ib].as_ptr(),
                    c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                    rsc as usize, // FIXME
                );
            }
        }
        if n % nr != 0 {
            K::kernel(
                k,
                pa.as_ptr(),
                pbs.last().unwrap().as_ptr(),
                tmpc.as_mut_ptr(),
                nr,
            );
            for y in 0..mr {
                for x in 0..(n % nr) {
                    unsafe {
                        *c.offset(
                            (mr * ia + y) as isize * rsc + (x + n / nr * nr) as isize * csc,
                        ) = tmpc[y * nr + x];
                    }
                }
            }
        }
    }
    if m % mr != 0 {
        let row = m - m % mr;
        unsafe {
            pack_panel_a(
                pa.as_mut_ptr(),
                a.offset(row as isize * rsa),
                rsa,
                csa,
                mr,
                m % mr,
                k,
            );
        }
        for ib in 0..n / nr {
            K::kernel(k, pa.as_ptr(), pbs[ib].as_ptr(), tmpc.as_mut_ptr(), nr);
            for y in 0..(m % mr) {
                for x in 0..nr {
                    unsafe {
                        *c.offset((y + row) as isize * rsc + (x + ib * nr) as isize * csc) =
                            tmpc[y * nr + x];
                    }
                }
            }
        }
        if n % nr != 0 {
            K::kernel(
                k,
                pa.as_ptr(),
                pbs.last().unwrap().as_ptr(),
                tmpc.as_mut_ptr(),
                nr,
            );
            for y in 0..(m % mr) {
                for x in 0..(n % nr) {
                    unsafe {
                        *c.offset((y + row) as isize * rsc + (x + n / nr * nr) as isize * csc) =
                            tmpc[y * nr + x];
                    }
                }
            }
        }
    }
}
