#![allow(dead_code, non_upper_case_globals, unused_macros, non_snake_case, unused_assignments)]

use std::arch::asm;

mod nano;

#[repr(C, align(8))]
struct Floats([f32; 4096]);
const _F32: Floats = Floats([12.; 4096]);
const F32: *const f32 = (&_F32) as *const Floats as *const f32;

lazy_static::lazy_static! {
    static ref TICK: f64 = unsafe { b8192!(asm!("orr x20, x20, x20", out("x20") _)) };
}

pub unsafe fn armv8(filter: Option<&str>) {
    macro_rules! s32 {
        ($label: literal, $n: expr, $stmt:block) => {
            if $label.contains(filter.unwrap_or("")) {
                println!("{:40} {:.2}", $label, b32!($stmt) / $n as f64 / *TICK);
            }
        };
    }

    macro_rules! s128 {
        ($label: literal, $n: expr, $stmt:block) => {
            if $label.contains(filter.unwrap_or("")) {
                println!("{:40} {:.2}", $label, b128!($stmt) / $n as f64 / *TICK);
            }
        };
    }

    macro_rules! s1024 {
        ($label: literal, $n: expr, $stmt:block) => {
            if $label.contains(filter.unwrap_or("")) {
                println!("{:40} {:.2}", $label, b1024!($stmt) / $n as f64 / *TICK);
            }
        };
    }

    macro_rules! s8192 {
        ($label: literal, $n: expr, $stmt:block) => {
            if $label.contains(filter.unwrap_or("")) {
                println!("{:40} {:.2}", $label, b8192!($stmt) / $n as f64 / *TICK);
            }
        };
    }

    s128!("nop", 1, { asm!("nop") });
    s128!("vands", 4, {
        asm!("  and v0.16b, v1.16b, v1.16b
                and v2.16b, v3.16b, v3.16b
                and v4.16b, v5.16b, v5.16b
                and v6.16b, v7.16b, v7.16b ",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        )
    });
    s128!("fmax", 4, {
        asm!("  fmax v0.4s, v1.4s, v1.4s
                fmax v2.4s, v3.4s, v3.4s
                fmax v4.4s, v5.4s, v5.4s
                fmax v6.4s, v7.4s, v7.4s ",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        )
    });
    s128!("fmax_with_dep", 1, { asm!("fmax v0.4s, v0.4s, v0.4s", out("v0") _) });
    s128!("fmla", 16, {
        asm!(" fmla v0.4s, v0.4s, v0.4s
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
               fmla v15.4s,v15.4s,v15.4s ",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        )
    });

    s128!("fmla_with_dep", 1, { asm!("fmla v0.4s, v0.4s, v0.4s", out("v0") _) });
    s32!("w_load", 64, {
        let mut p = F32;
        r8!(asm!("ldr w20, [{0}]
                   ldr w21, [{0}]
                   ldr w22, [{0}]
                   ldr w23, [{0}]
                   ldr w24, [{0}]
                   ldr w25, [{0}]
                   ldr w26, [{0}]
                   ldr w27, [{0}]",
        inout(reg) p,
        out("x20") _, out("x21") _, out("x22") _, out("x23") _,
        out("x24") _, out("x25") _, out("x26") _, out("x27") _,
        ));
    });
    s32!("x_load", 64, {
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
    });
    s32!("d_load", 64, {
        let mut p = F32;
        r8!(asm!("
       ldr d20, [{0}]
       ldr d21, [{0}]
       ldr d22, [{0}]
       ldr d23, [{0}]
       ldr d24, [{0}]
       ldr d25, [{0}]
       ldr d26, [{0}]
       ldr d27, [{0}]
       ",
        inout(reg) p,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        ));
    });
    s32!("s_load", 64, {
        let mut p = F32;
        r8!(asm!("
       ld1 {{v20.s}}[0], [{0}]
       ld1 {{v21.s}}[0], [{0}]
       ld1 {{v22.s}}[0], [{0}]
       ld1 {{v23.s}}[0], [{0}]
       ld1 {{v24.s}}[0], [{0}]
       ld1 {{v25.s}}[0], [{0}]
       ld1 {{v26.s}}[0], [{0}]
       ld1 {{v27.s}}[0], [{0}]
       ",
        inout(reg) p,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        ));
    });
    s32!("d_load_as_v", 64, {
        let mut p = F32;
        r8!(asm!("
       ld1 {{v20.d}}[0], [{0}]
       ld1 {{v21.d}}[0], [{0}]
       ld1 {{v22.d}}[0], [{0}]
       ld1 {{v23.d}}[0], [{0}]
       ld1 {{v24.d}}[0], [{0}]
       ld1 {{v25.d}}[0], [{0}]
       ld1 {{v26.d}}[0], [{0}]
       ld1 {{v27.d}}[0], [{0}]
       ",
        inout(reg) p,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        ));
    });
    s32!("v_load", 64, {
        let mut p = F32;
        r8!(asm!("
       ld1 {{v20.4s}}, [{0}]
       ld1 {{v21.4s}}, [{0}]
       ld1 {{v22.4s}}, [{0}]
       ld1 {{v23.4s}}, [{0}]
       ld1 {{v24.4s}}, [{0}]
       ld1 {{v25.4s}}, [{0}]
       ld1 {{v26.4s}}, [{0}]
       ld1 {{v27.4s}}, [{0}]
       ",
        inout(reg) p,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        ));
    });
    s32!("v2_load", 64, {
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
    });
    s32!("v3_load", 32, {
        let mut p = F32;
        r8!(asm!("
           ld1 {{v0.4s, v1.4s, v2.4s}}, [{0}]
           ld1 {{v3.4s, v4.4s, v5.4s}}, [{0}]
           ld1 {{v6.4s, v7.4s, v8.4s}}, [{0}]
           ld1 {{v9.4s, v10.4s, v11.4s}}, [{0}]
       ",
        inout(reg) p,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        ));
    });
    s32!("v4_load", 32, {
        let mut p = F32;
        r8!(asm!("
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
    });
    s32!("ins_32b", 64, {
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
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        ));
    });
    s32!("ins_32b_same_lane", 128, {
        r8!(asm!("
           ins         v0.s[0], w20
           ins         v1.s[0], w20
           ins         v4.s[0], w20
           ins         v5.s[0], w20
           ins         v0.s[1], w20
           ins         v1.s[1], w20
           ins         v4.s[1], w20
           ins         v5.s[1], w20
           ins         v0.s[2], w20
           ins         v1.s[2], w20
           ins         v4.s[2], w20
           ins         v5.s[2], w20
           ins         v0.s[3], w20
           ins         v1.s[3], w20
           ins         v4.s[3], w20
           ins         v5.s[3], w20
       ",
        out("v0") _, out("v1") _, out("v4") _, out("v5") _,
        ));
    });
    s32!("ins_64b", 64, {
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
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        ));
    });
    s32!("ins_64b_same_v", 64, {
        r8!(asm!("
                     ins v8.d[0], x20
                     ins v8.d[1], x20
                     ins v8.d[0], x20
                     ins v8.d[1], x20
                     ins v8.d[0], x20
                     ins v8.d[1], x20
                     ins v8.d[0], x20
                     ins v8.d[1], x20
                     ",
        out("v8") _,
        ));
    });
    s32!("ins_64b_from_v", 64, {
        r8!(asm!("
                     ins v8.d[0], v9.d[0]
                     ins v8.d[1], v9.d[0]
                     ins v8.d[0], v9.d[1]
                     ins v8.d[1], v9.d[1]
                     ins v8.d[0], v9.d[0]
                     ins v8.d[1], v9.d[0]
                     ins v8.d[0], v9.d[1]
                     ins v8.d[1], v9.d[1]
                     ",
        out("v8") _,
        ));
    });
    s32!("fmla_with_prfm", 64, {
        let mut p = F32;
        r8!(asm!("
           prfm pldl1keep, [{0}, #256]
           fmla v0.4s, v0.4s, v0.4s
           prfm pldl1keep, [{0}, #320]
           fmla v1.4s, v1.4s, v1.4s
           prfm pldl1keep, [{0}, #384]
           fmla v2.4s, v2.4s, v2.4s
           prfm pldl1keep, [{0}, #448]
           fmla v3.4s, v3.4s, v3.4s
           prfm pldl1keep, [{0}, #512]
           fmla v4.4s, v4.4s, v4.4s
           prfm pldl1keep, [{0}, #576]
           fmla v5.4s, v5.4s, v5.4s
           prfm pldl1keep, [{0}, #640]
           fmla v6.4s, v6.4s, v6.4s
           prfm pldl1keep, [{0}, #704]
           fmla v7.4s, v7.4s, v7.4s
           prfm pldl1keep, [{0}, #768]
           ",
        inout(reg) p,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        ));
    });
    s32!("fmla_with_w_load", 64, {
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
    });
    s32!("fmla_with_w_load_inc", 64, {
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
    });
    s32!("fmla_with_w_load_inc_alt", 64, {
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
    });
    s32!("fmla_with_w_load_offset", 64, {
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
    });
    s32!("fmla_with_x_load", 64, {
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
    });
    s32!("fmla_with_s_load", 64, {
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
    });
    s32!("fmla_with_d_load", 64, {
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
    });
    s32!("fmla_with_d_load_as_v", 64, {
        let mut p = F32;
        r8!(asm!("
                     fmla v0.4s, v0.4s, v0.4s
                     ld1 {{ v9.d }}[0], [{0}]
                     fmla v1.4s, v1.4s, v1.4s
                     ld1 {{ v10.d }}[0], [{0}]
                     fmla v2.4s, v2.4s, v2.4s
                     ld1 {{ v11.d }}[0], [{0}]
                     fmla v3.4s, v3.4s, v3.4s
                     ld1 {{ v12.d }}[0], [{0}]
                     fmla v4.4s, v4.4s, v4.4s
                     ld1 {{ v13.d }}[0], [{0}]
                     fmla v5.4s, v5.4s, v5.4s
                     ld1 {{ v14.d }}[0], [{0}]
                     fmla v6.4s, v6.4s, v6.4s
                     ld1 {{ v15.d }}[0], [{0}]
                     fmla v7.4s, v7.4s, v7.4s
                     ld1 {{ v16.d }}[0], [{0}]
                     ",
        inout(reg) p,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        ));
    });
    s32!("fmla_with_v_load", 64, {
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
    });
    s32!("fmla_with_ins_32b", 64, {
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
    });
    s32!("fmla_with_ins_64b", 64, {
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
    });
    s32!("fmla_with_ins_64b_cross_parity", 64, {
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
    });
    s32!("ins_32b_with_load_s", 64, {
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
    });
    s32!("ins_32b_with_load_s_cross_parity", 64, {
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
    });
}

fn has_asimdhp() -> bool {
    std::fs::read_to_string("/proc/cpuinfo").unwrap().contains("asimdhp")
}

#[target_feature(enable = "fp16")]
pub unsafe fn asimdhp(filter: Option<&str>) {
    macro_rules! s32 {
        ($label: literal, $n: expr, $stmt:block) => {
            if $label.contains(filter.unwrap_or("")) {
                println!("{:40} {:.2}", $label, b32!($stmt) / $n as f64 / *TICK);
            }
        };
    }

    s32!("fmlahp", 16, {
        asm!(" fmla v0.8h, v0.8h, v0.8h
               fmla v1.8h, v1.8h, v1.8h
               fmla v2.8h, v2.8h, v2.8h
               fmla v3.8h, v3.8h, v3.8h
               fmla v4.8h, v4.8h, v4.8h
               fmla v5.8h, v5.8h, v5.8h
               fmla v6.8h, v6.8h, v6.8h
               fmla v7.8h, v7.8h, v7.8h
               fmla v8.8h, v8.8h, v8.8h
               fmla v9.8h, v9.8h, v9.8h
               fmla v10.8h,v10.8h,v10.8h
               fmla v11.8h,v11.8h,v11.8h
               fmla v12.8h,v12.8h,v12.8h
               fmla v13.8h,v13.8h,v13.8h
               fmla v14.8h,v14.8h,v14.8h
               fmla v15.8h,v15.8h,v15.8h ",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        )
    });

    s32!("fcvt", 16, {
        asm!(" fcvtn v0.4h,  v0.4s
               fcvtn v1.4h,  v1.4s 
               fcvtn v2.4h,  v2.4s 
               fcvtn v3.4h,  v3.4s 
               fcvtn v4.4h,  v4.4s 
               fcvtn v5.4h,  v5.4s 
               fcvtn v6.4h,  v6.4s 
               fcvtn v7.4h,  v7.4s 
               fcvtn v8.4h,  v8.4s 
               fcvtn v9.4h,  v9.4s 
               fcvtn v10.4h, v10.4s
               fcvtn v11.4h, v11.4s
               fcvtn v12.4h, v12.4s
               fcvtn v13.4h, v13.4s
               fcvtn v14.4h, v14.4s
               fcvtn v15.4h, v15.4s",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        )
    });

    s32!("fcvt2", 16, {
        asm!(" fcvtn2 v0.8h,  v0.4s
               fcvtn2 v1.8h,  v1.4s 
               fcvtn2 v2.8h,  v2.4s 
               fcvtn2 v3.8h,  v3.4s 
               fcvtn2 v4.8h,  v4.4s 
               fcvtn2 v5.8h,  v5.4s 
               fcvtn2 v6.8h,  v6.4s 
               fcvtn2 v7.8h,  v7.4s 
               fcvtn2 v8.8h,  v8.4s 
               fcvtn2 v9.8h,  v9.4s 
               fcvtn2 v10.8h, v10.4s
               fcvtn2 v11.8h, v11.4s
               fcvtn2 v12.8h, v12.4s
               fcvtn2 v13.8h, v13.4s
               fcvtn2 v14.8h, v14.4s
               fcvtn2 v15.8h, v15.4s",
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        )
    });

    s32!("fmlahp_with_dep", 1, { asm!("fmla v0.8h, v0.8h, v0.8h", out("v0") _) });
    s32!("fcvtn_with_dep", 1, { asm!("fcvtn v0.4h, v0.4s", out("v0") _) });
    s32!("fcvtn2_with_dep", 1, { asm!("fcvtn2 v0.8h, v0.4s", out("v0") _) });
}

macro_rules! ksimd {
    ($filter: expr, $vector_size: expr, $geo: literal, $n: expr, $path: literal) => {
        kloop!($filter, $vector_size, $geo, $n, "arm64simd", $path)
    }
}

macro_rules! kfp16 {
    ($filter: expr, $vector_size: expr, $geo: literal, $n: expr, $path: literal) => {
        kloop!($filter, $vector_size, $geo, $n, "arm64fp16", $path)
    }
}

macro_rules! kloop {
    ($filter: expr, $vector_size: expr, $geo: literal, $n: expr, $dir: literal, $path: literal) => {
        let label = $path.split("/").last().unwrap().split_once(".").unwrap().0;
        let full_label = format!("{:8} {:40}", $geo, label);
        if full_label.contains($filter.unwrap_or("")) {
            let time = b2!({
                let mut p = F32;
                let mut q = F32;
                r4!(asm!(include_str!(concat!("../arm64/", $dir, "/", $path)),
                inout("x1") p, inout("x2") q, out("x3") _,
                out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                out("x8") _, out("x9") _, out("x10") _, out("x11") _,
                out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                out("x24") _, out("x25") _, out("x26") _, out("x27") _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                ));
            }) / 4.;
            println!("{} {:3.0}% ({:0.2}/{} cy)", full_label, $n as f64 / $vector_size as f64 / time * 100. * *TICK, time / *TICK, $n as f64 / $vector_size as f64);
        }
    }
}

unsafe fn f32_8x8(f: Option<&str>) {
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/naive.tmpli");
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/broken_chains.tmpli");
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/ldr_x_no_preload.tmpli");
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/ldr_x_preload.tmpli");
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/ldr_w_no_preload.tmpli");
    ksimd!(f, 4, "8x8x1xf32", 64, "arm64simd_mmm_f32_8x8/packed_packed_loop1/ldr_w_preload.tmpli");
    ksimd!(f, 4, "8x8x2xf32", 128, "arm64simd_mmm_f32_8x8/packed_packed_loop2/broken_chains.tmpli");
    ksimd!(f, 4, "8x8x2xf32", 128, "arm64simd_mmm_f32_8x8/packed_packed_loop2/cortex_a55.tmpli");
}

unsafe fn f32_12x8(f: Option<&str>) {
    ksimd!(f, 4, "12x8x1xf32", 96, "arm64simd_mmm_f32_12x8/packed_packed_loop1/naive.tmpli");
    ksimd!(f, 4, "12x8x1xf32", 96, "arm64simd_mmm_f32_12x8/packed_packed_loop1/ldr_w_no_preload.tmpli");
    ksimd!(f, 4, "12x8x1xf32", 96, "arm64simd_mmm_f32_12x8/packed_packed_loop1/ldr_w_preload.tmpli");
    ksimd!(f, 4, "12x8x1xf32", 96, "arm64simd_mmm_f32_12x8/packed_packed_loop1/ldr_x_preload.tmpli");
    ksimd!(f, 4, "12x8x2xf32", 192, "arm64simd_mmm_f32_12x8/packed_packed_loop2/cortex_a55.tmpli");
}

unsafe fn f32_16x4(f: Option<&str>) {
    ksimd!(f, 4, "16x4x1xf32", 64, "arm64simd_mmm_f32_16x4/packed_packed_loop1/naive.tmpli");
    ksimd!(f, 4, "16x4x1xf32", 64, "arm64simd_mmm_f32_16x4/packed_packed_loop1/cortex_a53.tmpli");
    ksimd!(f, 4, "16x4x2xf32", 128, "arm64simd_mmm_f32_16x4/packed_packed_loop2/cortex_a55.tmpli");
}

unsafe fn f32_24x4(f: Option<&str>) {
    ksimd!(f, 4, "24x4x1xf32", 96, "arm64simd_mmm_f32_24x4/packed_packed_loop1/naive.tmpli");
    ksimd!(f, 4, "24x4x1xf32", 96, "arm64simd_mmm_f32_24x4/packed_packed_loop1/cortex_a53.tmpli");
    ksimd!(f, 4, "24x4x1xf32", 96, "arm64simd_mmm_f32_24x4/packed_packed_loop1/cortex_a55.tmpli");
}

unsafe fn f32_64x1(f: Option<&str>) {
    ksimd!(f, 4, "64x1x1xf32", 64, "arm64simd_mmm_f32_64x1/loop1/naive.tmpli");
    ksimd!(f, 4, "64x1x1xf32", 64, "arm64simd_mmm_f32_64x1/loop1/cortex_a53.tmpli");
    ksimd!(f, 4, "64x1x2xf32", 128, "arm64simd_mmm_f32_64x1/loop2/naive.tmpli");
    ksimd!(f, 4, "64x1x2xf32", 128, "arm64simd_mmm_f32_64x1/loop2/cortex_a55.tmpli");
}

// RUSTFLAGS="-C target-feature=+fp16" cargo +nightly dinghy -d khadas-paris bench --bench arm64simd
#[target_feature(enable = "fp16")]
unsafe fn f16_16x8(f: Option<&str>) {
    kfp16!(f, 8, "16x8x1xf16", 128, "arm64fp16_mmm_f16_16x8/loop1/naive.tmpli");
    kfp16!(f, 8, "16x8x2xf16", 256, "arm64fp16_mmm_f16_16x8/loop2/cortex_a55.tmpli");
    kfp16!(f, 8, "32x4x1xf16", 128, "arm64fp16_mmm_f16_32x4/loop1/naive.tmpli");
    kfp16!(f, 8, "32x4x2xf16", 256, "arm64fp16_mmm_f16_32x4/loop2/cortex_a55.tmpli");
}

fn main() {
    println!("freq {:.2}GHz\n", 1e-9 / *TICK);

    let filter = std::env::args().skip(1).filter(|a| a != "--bench").next();
    unsafe {
        armv8(filter.as_deref());
        if has_asimdhp() {
            asimdhp(filter.as_deref());
        }
        f32_8x8(filter.as_deref());
        f32_12x8(filter.as_deref());
        f32_16x4(filter.as_deref());
        f32_24x4(filter.as_deref());
        f32_64x1(filter.as_deref());
        f16_16x8(filter.as_deref());
    }
}
