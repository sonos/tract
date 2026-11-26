//! Runs the two hand models natively, for timing and for comparing with another runtime.
//!
//!   bezier-piano-native [--wgpu] <hand_landmarker.task>
//!       times both models on a blank input
//!   bezier-piano-native [--wgpu] <hand_landmarker.task> detect|landmarks <input.f32> <output.f32>
//!       runs one model on raw little-endian f32 RGB and writes its outputs the same way

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use bezier_piano_web::Backend;
use bezier_piano_web::DETECTOR_SIZE;
use bezier_piano_web::Hands;
use bezier_piano_web::LANDMARKS_SIZE;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("Reading {path}"))?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn run(hands: &Hands, model: &str, rgb: &[f32]) -> Result<Vec<f32>> {
    pollster::block_on(async {
        if model == "detect" { hands.detect(rgb).await } else { hands.landmarks(rgb).await }
    })
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = if args.first().is_some_and(|a| a == "--wgpu") {
        args.remove(0);
        Backend::Wgpu
    } else {
        Backend::Cpu
    };
    let task =
        args.first().context("Usage: bezier-piano-native [--wgpu] <hand_landmarker.task> ...")?;
    let hands = Hands::from_task(&std::fs::read(task)?, backend)?;
    match args.get(1).map(String::as_str) {
        None => {
            for (model, side) in [("detect", DETECTOR_SIZE), ("landmarks", LANDMARKS_SIZE)] {
                let rgb = vec![0.5f32; side * side * 3];
                run(&hands, model, &rgb)?;
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    run(&hands, model, &rgb)?;
                }
                let ms = start.elapsed().as_secs_f64() * 1000. / 20.;
                println!("{model} ({backend:?}): {ms:.2} ms");
            }
        }
        Some(model @ ("detect" | "landmarks")) => {
            let [input, output] = [args.get(2), args.get(3)].map(|a| a.context("Missing path"));
            let out = run(&hands, model, &read_f32(input?)?)?;
            std::fs::write(output?, out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        Some(other) => bail!("Unknown command {other}"),
    }
    Ok(())
}
