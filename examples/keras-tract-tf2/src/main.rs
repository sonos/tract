use tract_core::ndarray::{IxDyn, OwnedRepr};
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
    // load the model
    .model_for_path("example.onnx")?
    // optimize graph
    .into_optimized()?
    // make the model runnable and fix its inputs and outputs
    .into_runnable()?;

    // load input and expected output from npz file into tract tensors
    let io = std::fs::File::open("io.npz").unwrap();
    let mut npz = ndarray_npy::NpzReader::new(io)?;
    let input = npz.by_name::<OwnedRepr<f32>, IxDyn>("input.npy").unwrap().into_tensor();
    let expected = npz.by_name::<OwnedRepr<f32>, IxDyn>("output.npy").unwrap().into_tensor();

    // Input the generated data into the model
    let found = model.run(tvec![input.into()]).unwrap().remove(0);
    assert!(found.close_enough(&expected, true).is_ok());
    Ok(())
}
