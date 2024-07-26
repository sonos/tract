use tract_data::prelude::Blob;

#[path = "../../benches/nano.rs"]
mod nano;

fn load_a_slice(slice: &[u8]) {
    unsafe {
        let mut ptr = slice.as_ptr();
        let end = ptr.add(slice.len());
        while ptr < end {
            std::arch::asm!("
                vmovaps ymm0, [rsi]
                vmovaps ymm1, [rsi + 256]
                vmovaps ymm2, [rsi + 512]
                vmovaps ymm3, [rsi + 768]
                    ", inout("rsi") ptr,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            out("ymm3") _,
            );
            ptr = ptr.add(1024);
        }
    }
}

fn main() {
    let buffer = unsafe { Blob::new_for_size_and_align(1024 * 1024 * 128, 256) };
    for size in (0..)
        .flat_map(|po2| (0..8).map(move |f| (8192 + 1024 * f) * (1 << po2)))
        .take_while(|&s| s < buffer.len())
    {
        let b = nano::run_bench(|| load_a_slice(&buffer[0..size]));
        let bw = size as f64 / b;
        // println!("{:12} B : {:4.0} GB/s", size, (bw / (1024. * 1024. * 1024.)) as usize);
        println!("{} {}", size, (bw / (1024. * 1024. * 1024.)) as usize);
    }
}
