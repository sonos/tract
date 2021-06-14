#![feature(asm)]
#![allow(dead_code, non_upper_case_globals, unused_macros, non_snake_case, unused_assignments)]

use criterion::{criterion_group, criterion_main, Criterion};

macro_rules! r2 { ($($stat:stmt)*) => { $( $stat )* $( $stat )* } }
macro_rules! r4 { ($($stat:stmt)*) => { r2!(r2!($($stat)*)) }}
macro_rules! r8 { ($($stat:stmt)*) => { r4!(r2!($($stat)*)) }}
macro_rules! r16 { ($($stat:stmt)*) => { r4!(r4!($($stat)*)) }}
macro_rules! r32 { ($($stat:stmt)*) => { r8!(r4!($($stat)*)) }}
macro_rules! r64 { ($($stat:stmt)*) => { r8!(r8!($($stat)*)) }}

const _F32: [f32; 64] = [12.; 64];
const F32: *const f32 = _F32.as_ptr();

pub fn ld_64F32(c: &mut Criterion) {
    let mut group = c.benchmark_group("ld64f32");
    group.throughput(criterion::Throughput::Elements(64));
    group.bench_function("noop", |b| {
        b.iter(|| unsafe {
            r64!(asm!("nop"));
        })
    });
    group.bench_function("orr_with_dep", |b| {
        b.iter(|| unsafe {
            r64!(asm!("orr x0, x0, x0"));
        })
    });
    group.bench_function("fmla", |b| {
        b.iter(|| unsafe {
            r4!(asm!("
                      fmla v0.4s, v0.4s, v0.4s
                      fmla v1.4s, v1.4s, v1.4s
                      fmla v2.4s, v2.4s, v2.4s
                      fmla v3.4s, v3.4s, v3.4s
                      fmla v4.4s, v4.4s, v4.4s
                      fmla v5.4s, v5.4s, v5.4s
                      fmla v6.4s, v6.4s, v6.4s
                      fmla v7.4s, v7.4s, v7.4s
                      fmla v8.4s, v8.4s, v8.4s
                      fmla v9.4s, v9.4s, v9.4s
                      fmla v10.4s,v10.4s,v10.4s
                      fmla v11.4s,v11.4s,v11.4s
                      fmla v12.4s,v12.4s,v12.4s
                      fmla v13.4s,v13.4s,v13.4s
                      fmla v14.4s,v14.4s,v14.4s
                      fmla v15.4s,v15.4s,v15.4s
                ",
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("fmla_with_dep", |b| {
        b.iter(|| unsafe {
            r64!(asm!("fmla v0.4s, v0.4s, v0.4s", out("v0") _));
        })
    });
    group.bench_function("w_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr w20, [{0}]
                        ldr w21, [{0}]
                        ldr w22, [{0}]
                        ldr w23, [{0}]
                        ldr w24, [{0}]
                        ldr w25, [{0}]
                        ldr w26, [{0}]
                        ldr w27, [{0}]
                      ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            ));
        })
    });
    group.bench_function("x_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr x20, [{0}]
                        ldr x21, [{0}]
                        ldr x22, [{0}]
                        ldr x23, [{0}]
                        ldr x24, [{0}]
                        ldr x25, [{0}]
                        ldr x26, [{0}]
                        ldr x27, [{0}]
                      ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            ));
        })
    });
    group.bench_function("d_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr d0, [{0}]
                        ldr d1, [{0}]
                        ldr d2, [{0}]
                        ldr d3, [{0}]
                        ldr d4, [{0}]
                        ldr d5, [{0}]
                        ldr d6, [{0}]
                        ldr d7, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("s_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ld1 {{v0.s}}[0], [{0}]
                        ld1 {{v1.s}}[0], [{0}]
                        ld1 {{v2.s}}[0], [{0}]
                        ld1 {{v3.s}}[0], [{0}]
                        ld1 {{v4.s}}[0], [{0}]
                        ld1 {{v5.s}}[0], [{0}]
                        ld1 {{v6.s}}[0], [{0}]
                        ld1 {{v7.s}}[0], [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("v_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ld1 {{v0.4s}}, [{0}]
                        ld1 {{v1.4s}}, [{0}]
                        ld1 {{v2.4s}}, [{0}]
                        ld1 {{v3.4s}}, [{0}]
                        ld1 {{v4.4s}}, [{0}]
                        ld1 {{v5.4s}}, [{0}]
                        ld1 {{v6.4s}}, [{0}]
                        ld1 {{v7.4s}}, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("v2_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ld1 {{v0.4s, v1.4s}}, [{0}]
                        ld1 {{v2.4s, v3.4s}}, [{0}]
                        ld1 {{v4.4s, v5.4s}}, [{0}]
                        ld1 {{v6.4s, v7.4s}}, [{0}]
                        ld1 {{v8.4s, v9.4s}}, [{0}]
                        ld1 {{v10.4s, v11.4s}}, [{0}]
                        ld1 {{v12.4s, v13.4s}}, [{0}]
                        ld1 {{v14.4s, v15.4s}}, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("v3_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r16!(asm!("
                        ld1 {{v0.4s, v1.4s, v2.4s}}, [{0}]
                        ld1 {{v3.4s, v4.4s, v5.4s}}, [{0}]
                        ld1 {{v6.4s, v7.4s, v8.4s}}, [{0}]
                        ld1 {{v9.4s, v10.4s, v11.4s}}, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("v4_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r16!(asm!("
                        ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{0}]
                        ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{0}]
                        ld1 {{v8.4s, v9.4s, v10.4s, v11.4s}}, [{0}]
                        ld1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("ins_32b", |b| {
        b.iter(|| unsafe {
            r8!(asm!("
                ins v8.s[0], w20
                ins v9.s[0], w20
                ins v10.s[0], w20
                ins v11.s[0], w20
                ins v12.s[0], w20
                ins v13.s[0], w20
                ins v14.s[0], w20
                ins v15.s[0], w20
                ",
            out("x20") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("ins_32b_same", |b| {
        b.iter(|| unsafe {
            r4!(asm!("
        ins         v0.s[0], w9
        ins         v1.s[0], w13
            ins         v4.s[0], w20
            ins         v5.s[0], w24
        ins         v0.s[1], w10
        ins         v1.s[1], w14
            ins         v4.s[1], w21
            ins         v5.s[1], w25
        ins         v0.s[2], w11
        ins         v1.s[2], w15
            ins         v4.s[2], w22
            ins         v5.s[2], w26
        ins         v0.s[3], w12
        ins         v1.s[3], w4
            ins         v4.s[3], w23
            ins         v5.s[3], w27
                ",
            out("v0") _, out("v1") _,
            out("v4") _, out("v5") _,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            ));
        })
    });
    group.bench_function("ins_64b", |b| {
        b.iter(|| unsafe {
            r8!(asm!("
                ins v8.d[0], x20
                ins v9.d[0], x20
                ins v10.d[0], x20
                ins v11.d[0], x20
                ins v12.d[0], x20
                ins v13.d[0], x20
                ins v14.d[0], x20
                ins v15.d[0], x20
                ",
            out("x20") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("fmla_with_w_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr w20, [{0}]
                        fmla v0.4s, v0.4s, v0.4s
                        ldr w21, [{0}]
                        fmla v1.4s, v1.4s, v1.4s
                        ldr w22, [{0}]
                        fmla v2.4s, v2.4s, v2.4s
                        ldr w23, [{0}]
                        fmla v3.4s, v3.4s, v3.4s
                        ldr w24, [{0}]
                        fmla v4.4s, v4.4s, v4.4s
                        ldr w25, [{0}]
                        fmla v5.4s, v5.4s, v5.4s
                        ldr w26, [{0}]
                        fmla v6.4s, v6.4s, v6.4s
                        ldr w27, [{0}]
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("fmla_with_w_load_inc", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr w20, [{0}], #4
                        fmla v0.4s, v0.4s, v0.4s
                        ldr w21, [{0}], #4
                        fmla v1.4s, v1.4s, v1.4s
                        ldr w22, [{0}], #4
                        fmla v2.4s, v2.4s, v2.4s
                        ldr w23, [{0}], #4
                        fmla v3.4s, v3.4s, v3.4s
                        ldr w24, [{0}], #4
                        fmla v4.4s, v4.4s, v4.4s
                        ldr w25, [{0}], #4
                        fmla v5.4s, v5.4s, v5.4s
                        ldr w26, [{0}], #4
                        fmla v6.4s, v6.4s, v6.4s
                        ldr w27, [{0}], #4
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("fmla_with_w_load_inc_alt", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            let mut q = F32;
            r8!(asm!("
                        ldr w20, [{0}], #4
                        fmla v0.4s, v0.4s, v0.4s
                        ldr w21, [{1}], #4
                        fmla v1.4s, v1.4s, v1.4s
                        ldr w22, [{0}], #4
                        fmla v2.4s, v2.4s, v2.4s
                        ldr w23, [{1}], #4
                        fmla v3.4s, v3.4s, v3.4s
                        ldr w24, [{0}], #4
                        fmla v4.4s, v4.4s, v4.4s
                        ldr w25, [{1}], #4
                        fmla v5.4s, v5.4s, v5.4s
                        ldr w26, [{0}], #4
                        fmla v6.4s, v6.4s, v6.4s
                        ldr w27, [{1}], #4
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p, inout(reg) q,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("fmla_with_w_load_offset", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr w20, [{0}]
                        fmla v0.4s, v0.4s, v0.4s
                        ldr w21, [{0}, #4]
                        fmla v1.4s, v1.4s, v1.4s
                        ldr w22, [{0}, #8]
                        fmla v2.4s, v2.4s, v2.4s
                        ldr w23, [{0}, #12]
                        fmla v3.4s, v3.4s, v3.4s
                        ldr w24, [{0}, #16]
                        fmla v4.4s, v4.4s, v4.4s
                        ldr w25, [{0}, #20]
                        fmla v5.4s, v5.4s, v5.4s
                        ldr w26, [{0}, #24]
                        fmla v6.4s, v6.4s, v6.4s
                        ldr w27, [{0}, #28]
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("fmla_with_x_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                fmla v0.4s, v0.4s, v0.4s
                ldr x20, [{0}]
                fmla v1.4s, v1.4s, v1.4s
                ldr x21, [{0}]
                fmla v2.4s, v2.4s, v2.4s
                ldr x22, [{0}]
                fmla v3.4s, v3.4s, v3.4s
                ldr x23, [{0}]
                fmla v4.4s, v4.4s, v4.4s
                ldr x24, [{0}]
                fmla v5.4s, v5.4s, v5.4s
                ldr x25, [{0}]
                fmla v6.4s, v6.4s, v6.4s
                ldr x26, [{0}]
                fmla v7.4s, v7.4s, v7.4s
                ldr x27, [{0}]
                ",
            inout(reg) p,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            ));
        })
    });
    group.bench_function("fmla_with_s_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr s16, [{0}]
                        fmla v0.4s, v0.4s, v0.4s
                        ldr s17, [{0}]
                        fmla v1.4s, v1.4s, v1.4s
                        ldr s18, [{0}]
                        fmla v2.4s, v2.4s, v2.4s
                        ldr s19, [{0}]
                        fmla v3.4s, v3.4s, v3.4s
                        ldr s20, [{0}]
                        fmla v4.4s, v4.4s, v4.4s
                        ldr s21, [{0}]
                        fmla v5.4s, v5.4s, v5.4s
                        ldr s22, [{0}]
                        fmla v6.4s, v6.4s, v6.4s
                        ldr s23, [{0}]
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("fmla_with_d_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                        ldr d16, [{0}]
                        fmla v0.4s, v0.4s, v0.4s
                        ldr d17, [{0}]
                        fmla v1.4s, v1.4s, v1.4s
                        ldr d18, [{0}]
                        fmla v2.4s, v2.4s, v2.4s
                        ldr d19, [{0}]
                        fmla v3.4s, v3.4s, v3.4s
                        ldr d20, [{0}]
                        fmla v4.4s, v4.4s, v4.4s
                        ldr d21, [{0}]
                        fmla v5.4s, v5.4s, v5.4s
                        ldr d22, [{0}]
                        fmla v6.4s, v6.4s, v6.4s
                        ldr d23, [{0}]
                        fmla v7.4s, v7.4s, v7.4s
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            ));
        })
    });
    group.bench_function("fmla_with_v_load", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                fmla v0.4s, v0.4s, v0.4s
                ld1 {{ v9.4s }}, [{0}]
                fmla v1.4s, v1.4s, v1.4s
                ld1 {{ v10.4s }}, [{0}]
                fmla v2.4s, v2.4s, v2.4s
                ld1 {{ v11.4s }}, [{0}]
                fmla v3.4s, v3.4s, v3.4s
                ld1 {{ v12.4s }}, [{0}]
                fmla v4.4s, v4.4s, v4.4s
                ld1 {{ v13.4s }}, [{0}]
                fmla v5.4s, v5.4s, v5.4s
                ld1 {{ v14.4s }}, [{0}]
                fmla v6.4s, v6.4s, v6.4s
                ld1 {{ v15.4s }}, [{0}]
                fmla v7.4s, v7.4s, v7.4s
                ld1 {{ v16.4s }}, [{0}]
                      ",
            inout(reg) p,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("fmla_with_ins_32b", |b| {
        b.iter(|| unsafe {
            r8!(asm!("
                fmla v0.4s, v0.4s, v0.4s
                ins v8.s[0], w20
                fmla v1.4s, v1.4s, v1.4s
                ins v9.s[0], w20
                fmla v2.4s, v2.4s, v2.4s
                ins v10.s[0], w20
                fmla v3.4s, v3.4s, v3.4s
                ins v11.s[0], w20
                fmla v4.4s, v4.4s, v4.4s
                ins v12.s[0], w20
                fmla v5.4s, v5.4s, v5.4s
                ins v13.s[0], w20
                fmla v6.4s, v6.4s, v6.4s
                ins v14.s[0], w20
                fmla v7.4s, v7.4s, v7.4s
                ins v15.s[0], w20
                ",
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("x20") _,
            ));
        })
    });
    group.bench_function("fmla_with_ins_64b", |b| {
        b.iter(|| unsafe {
            r8!(asm!("
                fmla v0.4s, v0.4s, v0.4s
                ins v8.d[0], x20
                fmla v1.4s, v1.4s, v1.4s
                ins v9.d[0], x20
                fmla v2.4s, v2.4s, v2.4s
                ins v10.d[0], x20
                fmla v3.4s, v3.4s, v3.4s
                ins v11.d[0], x20
                fmla v4.4s, v4.4s, v4.4s
                ins v12.d[0], x20
                fmla v5.4s, v5.4s, v5.4s
                ins v13.d[0], x20
                fmla v6.4s, v6.4s, v6.4s
                ins v14.d[0], x20
                fmla v7.4s, v7.4s, v7.4s
                ins v15.d[0], x20
                ",
            out("x20") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("fmla_with_ins_64b_cross_parity", |b| {
        b.iter(|| unsafe {
            r8!(asm!("
                fmla v0.4s, v0.4s, v0.4s
                ins v9.d[0], x20
                fmla v1.4s, v1.4s, v1.4s
                ins v10.d[0], x20
                fmla v2.4s, v2.4s, v2.4s
                ins v11.d[0], x20
                fmla v3.4s, v6.4s, v3.4s
                ins v12.d[0], x20
                fmla v4.4s, v4.4s, v4.4s
                ins v13.d[0], x20
                fmla v5.4s, v5.4s, v5.4s
                ins v14.d[0], x20
                fmla v6.4s, v6.4s, v6.4s
                ins v15.d[0], x20
                fmla v7.4s, v7.4s, v7.4s
                ins v8.d[0], x20
                ",
            out("x20") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("ins_32b_with_load_s", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                ldr s0, [{0}]
                ins v8.d[0], x20
                ldr s1, [{0}]
                ins v9.d[0], x20
                ldr s2, [{0}]
                ins v10.d[0], x20
                ldr s3, [{0}]
                ins v11.d[0], x20
                ldr s4, [{0}]
                ins v12.d[0], x20
                ldr s5, [{0}]
                ins v13.d[0], x20
                ldr s6, [{0}]
                ins v14.d[0], x20
                ldr s7, [{0}]
                ins v15.d[0], x20
                ",
            inout(reg) p,
            out("x20") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
    group.bench_function("ins_32b_with_load_s_cross_parity", |b| {
        b.iter(|| unsafe {
            let mut p = F32;
            r8!(asm!("
                ldr s0, [{0}]
                ins v9.d[0], x20
                ldr s1, [{0}]
                ins v10.d[0], x20
                ldr s2, [{0}]
                ins v11.d[0], x20
                ldr s3, [{0}]
                ins v12.d[0], x20
                ldr s4, [{0}]
                ins v13.d[0], x20
                ldr s5, [{0}]
                ins v14.d[0], x20
                ldr s6, [{0}]
                ins v15.d[0], x20
                ldr s7, [{0}]
                ins v8.d[0], x20
                ",
            inout(reg) p,
            out("x20") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            ));
        })
    });
}

fn packed_packed_8x8_loop1(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_packed_8x8_loop1");
    group.throughput(criterion::Throughput::Elements(8 * 8 * 4));

    macro_rules! bench_4_loop_1 {
        ($id:ident) => {
            group.bench_function(stringify!($id), |b| {
                b.iter(|| unsafe {
                    let mut p = F32;
                    let mut q = F32;
                    r4!(asm!(include_str!(concat!("../arm64/arm64simd/arm64simd_mmm_f32_8x8/packed_packed_loop1/", stringify!($id), ".tmpli")),
                    inout("x1") p, inout("x2") q,
                    out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                    out("x8") _, out("x9") _, out("x10") _, out("x11") _,
                    out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                    out("x20") _, out("x21") _, out("x22") _,
                    out("x23") _, out("x24") _, out("x25") _, out("x26") _,
                    out("v0") _, out("v1") _, out("v2") _,
                    out("v4") _, out("v5") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                    out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                    ));
                })
            });
        }
    }

    bench_4_loop_1!(naive);
    bench_4_loop_1!(ldr_x_no_preload);
    bench_4_loop_1!(ldr_w_no_preload);
    bench_4_loop_1!(ldr_w_preload);
}

fn packed_packed_12x8_loop1(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_packed_12x8_loop1");
    group.throughput(criterion::Throughput::Elements(12 * 8 * 4));

    macro_rules! bench_4_loop_1 {
        ($id:ident) => {
            group.bench_function(stringify!($id), |b| {
                b.iter(|| unsafe {
                    let mut p = F32;
                    let mut q = F32;
                    r4!(asm!(include_str!(concat!("../arm64/arm64simd/arm64simd_mmm_f32_12x8/packed_packed_loop1/", stringify!($id), ".tmpli")),
                    inout("x1") p, inout("x2") q,
                    out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                    out("x8") _, out("x9") _, out("x10") _, out("x11") _,
                    out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                    out("x20") _, out("x21") _, out("x22") _,
                    out("x23") _, out("x24") _, out("x25") _, out("x26") _,
                    out("v0") _, out("v1") _, out("v2") _,
                    out("v4") _, out("v5") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                    out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                    ));
                })
            });
        }
    }

    bench_4_loop_1!(naive);
    bench_4_loop_1!(ldr_w_no_preload);
    bench_4_loop_1!(ldr_w_preload);
}

fn packed_packed_16x4_loop1(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_packed_16x4_loop1");
    group.throughput(criterion::Throughput::Elements(16 * 4 * 4));

    macro_rules! bench_4_loop_1 {
        ($id:ident) => {
            group.bench_function(stringify!($id), |b| {
                b.iter(|| unsafe {
                    let mut p = F32;
                    let mut q = F32;
                    r4!(asm!(include_str!(concat!("../arm64/arm64simd/arm64simd_mmm_f32_16x4/packed_packed_loop1/", stringify!($id), ".tmpli")),
                    inout("x1") p, inout("x2") q,
                    out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                    out("x8") _, out("x9") _, out("x10") _, out("x11") _,
                    out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                    out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                    out("x24") _, out("x25") _, out("x26") _,
                    out("v0") _, out("v1") _, out("v2") _, out("v4") _, out("v5") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                    out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                    ));
                })
            });
        }
    }

    bench_4_loop_1!(naive);
    bench_4_loop_1!(cortex_a53);
}

criterion_group!(
    benches,
    ld_64F32,
    packed_packed_12x8_loop1,
    packed_packed_8x8_loop1,
    packed_packed_16x4_loop1
);
criterion_main!(benches);
