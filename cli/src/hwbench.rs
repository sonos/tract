use tract_core::tract_data::itertools::Itertools;
use tract_libcli::terminal::si_prefix;
use tract_linalg::hwbench::bandwidth::{l1_bandwidth_seq, main_memory_bandwith_seq};

pub(crate) fn handle() {
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
    println!("");
}
