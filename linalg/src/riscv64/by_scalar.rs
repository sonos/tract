//! Element-wise `buf op scalar` over f32, RVV 1.0.
//!
//! The loop is strip-mined on `vsetvli`, so it is vector-length agnostic and
//! declares only [`Isa::RiscV64V`], unlike the matmul kernels: `vl` is whatever
//! the hart grants and the tail falls out of the loop condition. `nr` is
//! therefore only the frame's chunking granularity, not a tile width.

macro_rules! rvv_by_scalar {
    ($func: ident, $vop: expr, $op: ident $(, param($param: ident))?) => {
        routine_by_scalar_rust!(riscv64;
            f32,
            $func,
            4,
            4,
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
                        ", $vop, "
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
            },
            op($op),
            bin
            $(, param($param))?,
            isa(RiscV64V)
        );
    };
}

rvv_by_scalar!(rvv_mul_by_scalar_f32, "vfmul.vf v8, v8, {s}", Mul, param(MulByScalar));
rvv_by_scalar!(rvv_add_by_scalar_f32, "vfadd.vf v8, v8, {s}", Add);
rvv_by_scalar!(rvv_sub_by_scalar_f32, "vfsub.vf v8, v8, {s}", Sub);
rvv_by_scalar!(rvv_subf_by_scalar_f32, "vfrsub.vf v8, v8, {s}", SubF);
rvv_by_scalar!(rvv_min_by_scalar_f32, "vfmin.vf v8, v8, {s}", Min);
rvv_by_scalar!(rvv_max_by_scalar_f32, "vfmax.vf v8, v8, {s}", Max);
