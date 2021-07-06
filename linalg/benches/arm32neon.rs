#![feature(asm)]
#![allow(dead_code, non_upper_case_globals, unused_macros, non_snake_case, unused_assignments)]

use std::time::Instant;

macro_rules! r2 { ($($stat:stmt)*) => { $( $stat )* $( $stat )* } }
macro_rules! r4 { ($($stat:stmt)*) => { r2!(r2!($($stat)*)) }}
macro_rules! r8 { ($($stat:stmt)*) => { r4!(r2!($($stat)*)) }}
macro_rules! r16 { ($($stat:stmt)*) => { r4!(r4!($($stat)*)) }}
macro_rules! r32 { ($($stat:stmt)*) => { r8!(r4!($($stat)*)) }}
macro_rules! r64 { ($($stat:stmt)*) => { r8!(r8!($($stat)*)) }}
macro_rules! r128 { ($($stat:stmt)*) => { r8!(r16!($($stat)*)) }}
macro_rules! r1024 { ($($stat:stmt)*) => { r8!(r128!($($stat)*)) }}
macro_rules! r4096 { ($($stat:stmt)*) => { r4!(r1024!($($stat)*)) }}

const _F32: [f32; 1024] = [12.; 1024];
const F32: *const f32 = _F32.as_ptr();

/*
fn ruin_cache() {
let _a = (0..1000000).collect::<Vec<i32>>();
}
*/

macro_rules! b {
    ($f: block, $inner_loop: expr, $measures: expr) => {{
        let mut values = Vec::with_capacity($measures);
        for _ in 0..$measures {
            //       ruin_cache();
            let start = Instant::now();
            for _ in 0..$inner_loop {
                unsafe { $f };
            }
            values.push(start.elapsed());
        }
        values.sort();
        values[$measures / 2].as_nanos() as f64 / 1e9 / $inner_loop as f64
    }};
}

fn main() {
    let cycle = b!(
        {
            r1024!(asm!("orr r0, r0, r0", out("r0") _));
        },
        1000,
        1000
    ) / 1024.;
    let indep_fmla = b!(
        {
            r8!(asm!("
                vmla.f32 q0, q0, q0
                vmla.f32 q1, q1, q1
                vmla.f32 q2, q2, q2
                vmla.f32 q3, q3, q3
                vmla.f32 q4, q4, q4
                vmla.f32 q5, q5, q5
                vmla.f32 q6, q6, q6
                vmla.f32 q7, q7, q7
                 ", out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _));
        },
        1000,
        1000
    ) / 64.;
    eprintln!("rcp tp: indep fmla: {}", indep_fmla / cycle);
    let dep_accu_fmla = b!(
        {
            r16!(asm!("
                vmla.f32 q15, q0, q0
                vmla.f32 q15, q1, q1
                vmla.f32 q15, q2, q2
                vmla.f32 q15, q3, q3
                vmla.f32 q15, q4, q4
                vmla.f32 q15, q5, q5
                vmla.f32 q15, q6, q6
                vmla.f32 q15, q7, q7
                vmla.f32 q15, q8, q8
                vmla.f32 q15, q9, q9
                vmla.f32 q15, q10, q10
                vmla.f32 q15, q11, q11
                vmla.f32 q15, q12, q12
                vmla.f32 q15, q13, q13
                vmla.f32 q15, q14, q14
                 ", out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
                 out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _));
        },
        1000,
        1000
    ) / 16.
        / 15.;
    eprintln!("rcp tp: accu-dep fmla: {}", dep_accu_fmla / cycle);
    let load_s_using_vld1_64 = b!(
        {
            let mut p = F32;
            r16!(asm!("
                vld1.64         {{d0-d3}}, [{0}]!
                vld1.64         {{d4-d7}}, [{0}]!
                vld1.64         {{d8-d11}}, [{0}]!
                vld1.64         {{d12-d15}}, [{0}]!
                vld1.64         {{d16-d19}}, [{0}]!
                vld1.64         {{d20-d23}}, [{0}]!
                vld1.64         {{d24-d27}}, [{0}]!
                vld1.64         {{d28-d31}}, [{0}]!
                 ", 
                 inout(reg) p,
                 out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
                 out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _));
        },
        1000,
        1000
    ) / 16.
        / 64.; // each line load 8 s
    eprintln!("rcp tp: load s using vld1_64 ia {}", load_s_using_vld1_64 / cycle);
    let load_s_using_vldm_q = b!(
        {
            let mut p = F32;
            r16!(asm!("
                vldm            {0}!, {{q0-q3}}
                vldm            {0}!, {{q4-q7}}
                vldm            {0}!, {{q8-q11}}
                vldm            {0}!, {{q12-q15}}
                 ", 
                 inout(reg) p,
                 out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
                 out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _));
        },
        1000,
        1000
    ) / 16.
        / 64.;
    eprintln!("rcp tp: load s using vldmia q: {}", load_s_using_vldm_q / cycle);
    let load = b!(
        {
            let mut p = F32;
            r16!(asm!("
                vldr.64  d0, [{0}]
                vldr.64  d1, [{0}, #8]
                vldr.64  d2, [{0}, #16]
                vldr.64  d3, [{0}, #24]
                vldr.64  d4, [{0}, #32]
                vldr.64  d5, [{0}, #40]
                vldr.64  d6, [{0}, #48]
                vldr.64  d7, [{0}, #56]
                vldr.64  d8, [{0}, #64]
                vldr.64  d9, [{0}, #72]
                vldr.64  d10, [{0}, #80]
                vldr.64  d11, [{0}, #88]
                vldr.64  d12, [{0}, #96]
                vldr.64  d13, [{0}, #104]
                vldr.64  d14, [{0}, #112]
                vldr.64  d15, [{0}, #120]
                vldr.64  d16, [{0}, #128]
                vldr.64  d17, [{0}, #136]
                vldr.64  d18, [{0}, #144]
                vldr.64  d19, [{0}, #152]
                vldr.64  d20, [{0}, #160]
                vldr.64  d21, [{0}, #168]
                vldr.64  d22, [{0}, #176]
                vldr.64  d23, [{0}, #184]
                vldr.64  d24, [{0}, #192]
                vldr.64  d25, [{0}, #200]
                vldr.64  d26, [{0}, #208]
                vldr.64  d27, [{0}, #216]
                vldr.64  d28, [{0}, #224]
                vldr.64  d29, [{0}, #232]
                vldr.64  d30, [{0}, #240]
                vldr.64  d31, [{0}, #248]
                add {0}, #256
                 ", 
                 inout(reg) p,
                 out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
                 out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _));
        },
        1000,
        1000
    ) / 16.
        / 64.;
    eprintln!("rcp tp: load s using vldr d + imm: {}", load / cycle);
}
