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
    println!();

    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        println!("# Excerpt from /proc/cpuinfo");
        for line in cpuinfo.lines() {
            if line.is_empty() {
                break;
            }
            if ["model name", "cache size", "bogomips", "BogoMIPS", "Features", "CPU", "flags"]
                .iter()
                .any(|needle| line.starts_with(needle))
            {
                println!(" * {line}");
            }
        }
        println!();

        if let Some(flags) = cpuinfo
            .lines()
            .find(|line| line.starts_with("flags") || line.starts_with("Features"))
            .and_then(|l| l.split_once(":"))
            .map(|pair| pair.1)
        {
            print!("# Relevant CPU flags/features: ");
            for flag in flags.split_whitespace() {
                if ["fpu", "sse", "avx", "f16", "fma", "fp", "asimd", "neon", "vfp"]
                    .iter()
                    .any(|needle| flag.starts_with(needle))
                {
                    print!("{flag} ")
                };
            }
            println!("\n");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!(
            "# Aarch64 subfamily detected by tract-linalg: {:?}\n",
            tract_linalg::arm64::Kind::choose()
        );
    }

    println!("# Cache");
    let mut threads = (1..=num_cpus::get()).collect_vec();
    for extra in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0] {
        let value = (num_cpus::get() * (extra * 4.) as usize) / 4;
        if !threads.contains(&value) {
            threads.push(value);
        }
    }
    for &t in &threads {
        let m = l1_bandwidth_seq(t);
        println!(
            "{t:2}-thread L1 : {} — {}",
            si_prefix(m, "B/s"),
            si_prefix(m / t as f64, "B/s/thread"),
        );
    }

    println!("\n# Main memory");
    for &t in &threads {
        let measured = main_memory_bandwith_seq(t);
        println!(
            "{t:2}-thread L∞ : {} — {}",
            si_prefix(measured, "B/s"),
            si_prefix(measured / t as f64, "B/s/thread")
        );
    }
    println!();

    let big = if cfg!(target_arch = "arm") { 128 } else { 512 };
    mmm(f32::datum_type(), big, big, big)?;
    mmm(f32::datum_type(), big, big, 1)?;
    mmm(f16::datum_type(), big, big, big)?;
    mmm(f16::datum_type(), big, big, 1)?;

    Ok(())
}

fn mmm(dt: DatumType, m: usize, k: usize, n: usize) -> TractResult<()> {
    let a = Tensor::zero_dt(dt, &[m, k])?;
    let b = Tensor::zero_dt(dt, &[k, n])?;
    let mut c = Tensor::zero_dt(dt, &[m, n])?;
    let selection = tract_linalg::ops().mmm(dt, Some(m), Some(k), Some(n));
    println!("# Matmul {m}x{k}x{n}x{dt:?}\n");
    let mmms = tract_linalg::ops().mmm_impls();
    unsafe {
        mmms.iter()
            .flat_map(|mmm| {
                mmm.packings().iter().enumerate().map(move |(pix, (pa, pb))| (mmm, pix, pa, pb))
            })
            .filter(|(_mmm, _pix, pa, pb)| {
                pa.precursor().as_dt() == Some(dt) && pb.precursor().as_dt() == Some(dt)
            })
            .map(|(mmm, pix, pa, pb)| {
                if atty::is(atty::Stream::Stderr) {
                    eprint!("Benching {} ({pix})", mmm.name());
                }
                let a = pa.prepare_one(&a, 1, 0).unwrap();
                let b = pb.prepare_one(&b, 0, 1).unwrap();
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
                                    a: AsInputValue::Borrowed(&*a),
                                    b: AsInputValue::Borrowed(&*b),
                                    packing: 0,
                                },
                                FusedSpec::Store(pc),
                            ],
                        )
                        .unwrap();
                    }
                });
                if atty::is(atty::Stream::Stderr) {
                    eprint!("\x1B[2K\r"); // clear current line + CR
                }
                let flops = (m * k * n) as f64 / time;
                (mmm, pix, pa, pb, flops)
            })
            .sorted_by_key(|(_mmm, _pix, _pa, _pb, flops)| -(*flops as i64))
            .for_each(|(mmm, pix, pa, pb, flops)| {
                print!("{:>35} {:30}", format!("{mmm:?} ({pix})"), format!("{pa} • {pb}"));
                let color = if flops.log10() > 9.0 {
                    Green
                } else if flops.log10() > 6.0 {
                    Yellow
                } else {
                    LightRed
                };
                println!(
                    " {} {}",
                    color.paint(si_prefix(flops, "flop/s")),
                    if pix == 0 && Some(mmm) == selection.as_ref() { "<--" } else { "" }
                );
            });
    }
    println!();

    Ok(())
}
