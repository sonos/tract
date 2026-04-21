use anyhow::Result;
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use tract::prelude::*;

tract::impl_ndarray_interop!();

fn main() -> Result<()> {
    let model = tract::onnx()?.load("example.onnx")?.into_model()?.into_runnable()?;

    // load input and expected output from npz file into ndarray arrays
    let io = std::fs::File::open("io.npz")?;
    let mut npz = ndarray_npy::NpzReader::new(io)?;
    let input: ArrayD<f32> =
        npz.by_name::<OwnedRepr<f32>, IxDyn>("input.npy").unwrap().into_owned();
    let expected: ArrayD<f32> =
        npz.by_name::<OwnedRepr<f32>, IxDyn>("output.npy").unwrap().into_owned();

    let found = model.run([input.tract()?])?;
    let found = found[0].as_slice::<f32>()?;
    let expected = expected.as_slice().unwrap();
    assert_eq!(found.len(), expected.len());
    for (a, b) in found.iter().zip(expected) {
        assert!((a - b).abs() < 1e-4, "found {a} vs expected {b}");
    }
    Ok(())
}
