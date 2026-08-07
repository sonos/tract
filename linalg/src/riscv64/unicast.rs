//! Element-wise `a op b` over two f32 slices, RVV 1.0.
//!
//! Strip-mined on `vsetvli` like [`super::by_scalar`], so vector-length
//! agnostic and needing nothing beyond the vector unit itself.

macro_rules! rvv_unicast {
    ($func: ident, $vop: expr, $op: ident) => {
        routine_unicast_rust!(riscv64;
            f32,
            $func,
            4,
            4,
            #[inline(never)]
            fn run(a: &mut [f32], b: &[f32]) {
                assert!(a.len() == b.len());
                assert!(!a.is_empty());
                let len = a.len();
                let a_ptr = a.as_mut_ptr();
                let b_ptr = b.as_ptr();
                // SAFETY: both slices hold `len` elements and the loop
                // advances all three cursors in step.
                unsafe {
                    std::arch::asm!(
                        concat!("
                        .option push
                        .option arch, +v
                        2:
                        vsetvli t0, {len}, e32, m8, ta, ma
                        vle32.v v8, ({a})
                        vle32.v v16, ({b})
                        ", $vop, "
                        vse32.v v8, ({a})
                        slli t1, t0, 2
                        add {a}, {a}, t1
                        add {b}, {b}, t1
                        sub {len}, {len}, t0
                        bnez {len}, 2b
                        .option pop
                        "),
                        len = inout(reg) len => _,
                        a = inout(reg) a_ptr => _,
                        b = inout(reg) b_ptr => _,
                        out("t0") _,
                        out("t1") _,
                        options(nostack),
                    );
                }
            },
            op($op),
            isa(RiscV64V)
        );
    };
}

rvv_unicast!(rvv_unicast_mul_f32, "vfmul.vv v8, v8, v16", Mul);
rvv_unicast!(rvv_unicast_add_f32, "vfadd.vv v8, v8, v16", Add);
rvv_unicast!(rvv_unicast_sub_f32, "vfsub.vv v8, v8, v16", Sub);
rvv_unicast!(rvv_unicast_subf_f32, "vfsub.vv v8, v16, v8", SubF);
rvv_unicast!(rvv_unicast_min_f32, "vfmin.vv v8, v8, v16", Min);
rvv_unicast!(rvv_unicast_max_f32, "vfmax.vv v8, v8, v16", Max);
