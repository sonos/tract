//! Element-wise `buf op scalar` over f32, RVV 1.0.
//!
//! The loop is strip-mined on `vsetvli`, so it is vector-length agnostic and
//! carries no VLEN predicate, unlike the matmul kernels: `vl` is whatever the
//! hart grants and the tail falls out of the loop condition. `nr` is therefore
//! only the frame's chunking granularity, not a tile width.

macro_rules! rvv_by_scalar {
    ($func: ident, $op: expr) => {
        by_scalar_impl_wrap!(
            f32,
            $func,
            4,
            4,
            f32,
            #[inline(never)]
            fn run(buf: &mut [f32], s: f32) {
                assert!(!buf.is_empty());
                let len = buf.len();
                let ptr = buf.as_mut_ptr();
                // SAFETY: `len` elements are in bounds from `ptr` by
                // construction, and the loop advances both together.
                unsafe {
                    std::arch::asm!(
                        concat!("
                        .option push
                        .option arch, +v
                        2:
                        vsetvli t0, {len}, e32, m8, ta, ma
                        vle32.v v8, ({ptr})
                        ", $op, "
                        vse32.v v8, ({ptr})
                        slli t1, t0, 2
                        add {ptr}, {ptr}, t1
                        sub {len}, {len}, t0
                        bnez {len}, 2b
                        .option pop
                        "),
                        len = inout(reg) len => _,
                        ptr = inout(reg) ptr => _,
                        s = in(freg) s,
                        out("t0") _,
                        out("t1") _,
                        options(nostack),
                    );
                }
            }
        );
    };
}

rvv_by_scalar!(rvv_mul_by_scalar_f32, "vfmul.vf v8, v8, {s}");
rvv_by_scalar!(rvv_add_by_scalar_f32, "vfadd.vf v8, v8, {s}");
rvv_by_scalar!(rvv_sub_by_scalar_f32, "vfsub.vf v8, v8, {s}");
rvv_by_scalar!(rvv_subf_by_scalar_f32, "vfrsub.vf v8, v8, {s}");
rvv_by_scalar!(rvv_min_by_scalar_f32, "vfmin.vf v8, v8, {s}");
rvv_by_scalar!(rvv_max_by_scalar_f32, "vfmax.vf v8, v8, {s}");

#[cfg(test)]
mod test {
    use super::*;
    use crate::riscv64::has_rvv;

    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_mul_by_scalar_f32, |a, b| a * b);
    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_add_by_scalar_f32, |a, b| a + b);
    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_sub_by_scalar_f32, |a, b| a - b);
    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_subf_by_scalar_f32, |a, b| b - a);
    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_min_by_scalar_f32, |a, b| a.min(b));
    crate::by_scalar_frame_tests!(has_rvv(), f32, rvv_max_by_scalar_f32, |a, b| a.max(b));
}
