use nu_ansi_term::Color::*;
use tract_core::prelude::*;
use tract_core::tract_data::itertools::Itertools;
use tract_libcli::terminal::si_prefix;
use tract_linalg::hwbench::bandwidth::{l1_bandwidth_seq, main_memory_bandwith_seq};
use tract_linalg::hwbench::runner::run_bench;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

pub(crate) fn handle() -> TractResult<()> {
    println!("# Cores");
    println!("cpus: {}", num_cpus::get());
    println!("physical cpus: {}", num_cpus::get_physical());
    println!("\n# Cache");
    let mut threads = (1..=num_cpus::get()).collect_vec();
    for extra in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0] {
        let value = (num_cpus::get() * (extra * 4.) as usize) / 4;
        if !threads.contains(&value) {
            threads.push(value);
        }
    }
    // for &t in &threads {
    //     let m = l1_bandwidth_seq(t);
    //     println!(
    //         "{t:2}-thread L1 : {} — {}",
    //         si_prefix(m, "B/s"),
    //         si_prefix(m / t as f64, "B/s/thread"),
    //     );
    // }

    // println!("\n# Main memory");
    // for &t in &threads {
    //     let measured = main_memory_bandwith_seq(t);
    //     println!(
    //         "{t:2}-thread L∞ : {} — {}",
    //         si_prefix(measured, "B/s"),
    //         si_prefix(measured / t as f64, "B/s/thread")
    //     );
    // }
    println!("");

    mmm(f32::datum_type(), 512, 512, 512)?;
    mmm(f32::datum_type(), 512, 512, 1)?;

    Ok(())
}

fn mmm(dt: DatumType, m: usize, k: usize, n: usize) -> TractResult<()> {
    let a = Tensor::zero_dt(dt, &[m, k])?;
    let b = Tensor::zero_dt(dt, &[k, n])?;
    let mut c = Tensor::zero_dt(dt, &[m, n])?;
    println!("# Matmul {m}x{k}x{n}x{dt:?}\n");
    for mmm in tract_linalg::ops().mmm_impls() {
        unsafe {
            for (pix, (packed_a, packed_b)) in mmm.packings().iter().enumerate() {
                if packed_a.precursor().as_dt() != Some(dt)
                    || packed_b.precursor().as_dt() != Some(dt)
                {
                    continue;
                }
                print!(
                    "* {:25} {:30}",
                    format!("{mmm:?} ({pix})"),
                    format!("{packed_a} x {packed_b}")
                );
                let pa = packed_a.prepare_one(&a, 1, 0)?;
                let pb = packed_b.prepare_one(&b, 0, 1)?;
                let pc = mmm.c_view(Some(0), Some(1)).wrap(&c.view_mut());
                let time = run_bench(|loops| {
                    let mut scratch = mmm.allocate_scratch_space();
                    for _ in 0..loops {
                        mmm.run_with_scratch_space(
                            m,
                            n,
                            scratch.as_mut(),
                            &[
                                FusedSpec::AddMatMul {
                                    a: AsInputValue::Borrowed(&*pa),
                                    b: AsInputValue::Borrowed(&*pb),
                                    packing: 0,
                                },
                                FusedSpec::Store(pc),
                            ],
                        )
                        .unwrap();
                    }
                });
                let flops = (m * k * n) as f64 / time;
                let color = if flops.log10() > 9.0 {
                    Green
                } else if flops.log10() > 6.0 {
                    Yellow
                } else {
                    LightRed
                };
                println!(" {}", color.paint(si_prefix(flops, "flop/s")));
            }
        }
    }
    println!("");

    Ok(())
}
