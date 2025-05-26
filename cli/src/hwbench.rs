use tract_libcli::terminal::si_prefix;
use tract_linalg::hwbench::bandwidth::{l1_bandwidth_seq, main_memory_bandwith_seq};

pub(crate) fn handle() {
    println!("# Cores");
    println!("cpus: {}", num_cpus::get());
    println!("physical cpus: {}", num_cpus::get_physical());
    println!("\n# Cache");
    for threads in [1, 2, 3, 4, 8, 16] {
        println!("{threads:2}-thread L∞  : {}", si_prefix(l1_bandwidth_seq(threads), "B/s"));
    }

    println!("\n# Main memory");
    for threads in [1, 2, 3, 4, 8, 16] {
        println!(
            "{threads:2}-thread L∞  : {}",
            si_prefix(main_memory_bandwith_seq(threads), "B/s")
        );
    }
    println!("");
}
