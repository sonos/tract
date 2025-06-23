use std::str::FromStr;

use float_ord::FloatOrd;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_transformers::WithTractTransformers;

fn main() -> TractResult<()> {
    let nn_path = "OhanaTract-llama3_2_1B_f32/model/model.nnef.tgz";

    eprintln!("Loading tokenizers");
    let tokenizer_path = "OhanaTract-llama3_2_1B_f32/tokenizer/tokenizer.json";
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
    eprintln!("Loaded tokenizers");

    eprintln!("Loading NN");
    let nnef = tract_nnef::nnef().with_tract_transformers();
    let mut nn = nnef.model_for_path(&nn_path)?.into_decluttered()?;

    let transform = nnef.get_transform("transformers-detect-all").unwrap().unwrap();
    nn.transform(&*transform)?;
    #[cfg(any(target_os = "darwin", target_os = "ios"))]
    tract_metal::MetalTransform::from_str("")?.transform(&mut nn).unwrap();
    nn.optimize()?;
    let nn = nn.into_runnable()?;
    eprintln!("Loaded NN");

    let prompt = "Paul McCartney is";
    let tokenized = tokenizer.encode(prompt, false).unwrap();

    let mut state = SimpleState::new(&nn)?;
    dbg!(&state.model().inputs);
    dbg!(state.model().input_fact(0))?;

    let mut seq = tokenized.get_ids().iter().map(|id| *id as i64).collect_vec();

    let mut processed = 0; // this is P

    loop {
        let mut input = tensor1(&seq[processed..]);
        input.insert_axis(0)?;
        dbg!(&input);
        let output = state.run(tvec![input.into_tvalue()])?;
        dbg!("ran");
        let next_tok = output[0]
            .as_slice::<f32>()?
            .iter()
            .enumerate()
            .max_by_key(|(_ix, v)| FloatOrd(**v))
            .unwrap()
            .0;
        dbg!(next_tok);
        processed = seq.len();
        seq.push(next_tok as i64);
        if seq.len() == 128 {
            break;
        }
    }

    let text = tokenizer.decode(&seq.iter().map(|s| *s as u32).collect_vec(), true).unwrap();

    dbg!(text);
    Ok(())
}
