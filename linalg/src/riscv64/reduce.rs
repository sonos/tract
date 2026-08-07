//! Whole-slice f32 reductions, RVV 1.0.
//!
//! The running result lives in element 0 of v1 and is fed back as the
//! reduction's scalar operand each round, so strip-mining needs no separate
//! accumulator vector and no final horizontal step. Reduction operands are
//! LMUL=1 whatever vtype says, which is why v1 stays clear of the v8 group.
//!
//! `vfredusum` is the unordered sum: it may reassociate, as the reference
//! kernels on other targets also do.

macro_rules! rvv_reduce {
    ($func: ident, $neutral: expr, $red: expr, $op: ident) => {
        routine_reduce_rust!(riscv64;
            f32,
            $func,
            4,
            4,
            #[inline(never)]
            fn run(buf: &[f32], _: ()) -> f32 {
                assert!(!buf.is_empty());
                let len = buf.len();
                let ptr = buf.as_ptr();
                let out: f32;
                // SAFETY: `len` elements are in bounds from `ptr`, and the
                // loop advances both together.
                unsafe {
                    std::arch::asm!(
                        concat!("
                        .option push
                        .option arch, +v
                        vsetivli t0, 1, e32, m1, ta, ma
                        vfmv.s.f v1, {neutral}
                        2:
                        vsetvli t0, {len}, e32, m8, ta, ma
                        vle32.v v8, ({ptr})
                        ", $red, "
                        slli t1, t0, 2
                        add {ptr}, {ptr}, t1
                        sub {len}, {len}, t0
                        bnez {len}, 2b
                        vfmv.f.s {out}, v1
                        .option pop
                        "),
                        len = inout(reg) len => _,
                        ptr = inout(reg) ptr => _,
                        neutral = in(freg) $neutral,
                        out = lateout(freg) out,
                        out("t0") _,
                        out("t1") _,
                        options(nostack),
                    );
                }
                out
            },
            op($op),
            isa(RiscV64V)
        );
    };
}

rvv_reduce!(rvv_max_f32, f32::MIN, "vfredmax.vs v1, v8, v1", Max);
rvv_reduce!(rvv_min_f32, f32::MAX, "vfredmin.vs v1, v8, v1", Min);
rvv_reduce!(rvv_sum_f32, 0f32, "vfredusum.vs v1, v8, v1", Sum);
