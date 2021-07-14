use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
    // load the model
    .model_for_path("bert_for_classification.onnx")?    // Model : DistilBERT; HuggingFace transformers
    // specify input type and shape : imported model works with 1x8 input tensors
    .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 8)))?
    // optimize the model
    .into_optimized()?
    // make the model runnable and fix its inputs and outputs
    .into_runnable()?;

    // Inputting dummy tensor : ones(1x8)
    let elems: i64 = 1;
    let dummy_input: Tensor = tract_ndarray::Array2::from_shape_fn((1, 8), |(_,_)| {elems}).into();
    println!("Input : {:?}", dummy_input);

    // run the model on the input
    let result = model.run(tvec!(dummy_input))?;

    // PyTorch output = ONNX output = [[[ 0.01977395 -0.02752648 ]]]
    // Printing Tract output :
    println!("Output : {:?}", result[0]);
    Ok(())
}
