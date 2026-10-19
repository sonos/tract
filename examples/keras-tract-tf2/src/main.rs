use anyhow::Result;
use ndarray::ArrayD;
use tract::prelude::*;

tract::impl_ndarray_interop!();

fn main() -> Result<()> {
    let model = tract::onnx()?.load("example.onnx")?.into_model()?.into_runnable()?;

    // load input and expected output from npz file into ndarray arrays
    let io = std::fs::File::open("io.npz")?;
    let mut npz = tract_data::npz::NpzReader::new(io)?;
    let input: ArrayD<f32> = npz.get("input.npy")?.into_plain_array()?;
    let expected: ArrayD<f32> = npz.get("output.npy")?.into_plain_array()?;

    let found = model.run([input.tract()?])?;
    let found = found[0].as_slice::<f32>()?;
    let expected = expected.as_slice().unwrap();
    assert_eq!(found.len(), expected.len());
    for (a, b) in found.iter().zip(expected) {
        assert!((a - b).abs() < 1e-4, "found {a} vs expected {b}");
    }
    Ok(())
}
