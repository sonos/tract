use ndarray::s;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use tokenizers::tokenizer::{Result, Tokenizer};
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let model_dir = PathBuf::from_str("./albert")?;
    let tokenizer = Tokenizer::from_file(Path::join(&model_dir, "tokenizer.json"))?;

    let text = "Paris is the [MASK] of France.";

    let tokenizer_output = tokenizer.encode(text, true)?;
    let input_ids = tokenizer_output.get_ids();
    let attention_mask = tokenizer_output.get_attention_mask();
    let token_type_ids = tokenizer_output.get_type_ids();
    let length = input_ids.len();
    let mask_pos =
        input_ids.iter().position(|&x| x == tokenizer.token_to_id("[MASK]").unwrap()).unwrap();

    let model = tract_onnx::onnx()
        .model_for_path(Path::join(&model_dir, "model.onnx"))?
        .into_optimized()?
        .into_runnable()?;

    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();

    let outputs =
        model.run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into()))?;
    let logits = outputs[0].to_array_view::<f32>()?;
    let logits = logits.slice(s![0, mask_pos, ..]);
    let word_id = logits.iter().zip(0..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap()).unwrap().1;
    let word = tokenizer.id_to_token(word_id);
    println!("Result: {word:?}");

    Ok(())
}
