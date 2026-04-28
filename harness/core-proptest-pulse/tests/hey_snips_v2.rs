//! Smoke-test PulseV2 against the hey_snips WaveNet model.
//!
//! Loads the cached .pb, optimises to a TypedModel, and asks PulseV2 to
//! pulsify with pulse=8. No reference comparison yet — first goal is just
//! "does v2 walk this graph at all?"

use std::path::PathBuf;

use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;
use tract_tensorflow::prelude::*;

fn cached_hey_snips() -> Option<PathBuf> {
    for candidate in [
        std::env::var("MODELS").ok().map(|m| PathBuf::from(m).join("hey_snips_v4_model17.pb")),
        Some(PathBuf::from(".cached/hey_snips_v4_model17.pb")),
        Some(PathBuf::from("../../.cached/hey_snips_v4_model17.pb")),
    ]
    .into_iter()
    .flatten()
    {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[test]
#[ignore = "hangs at skip-Add: PulseV2Slice still emits variable-pulse output, \
            doesn't unify with the constant-P conv side; needs Slice rewrite \
            for fixed-pulse semantics"]
fn hey_snips_pulsify_v2() -> TractResult<()> {
    let Some(path) = cached_hey_snips() else {
        eprintln!("hey_snips_v4_model17.pb not cached — skipping");
        return Ok(());
    };

    let mut model = tensorflow().model_for_path(&path)?;
    let s = model.symbols.sym("S");
    model.set_input_fact(0, f32::fact([s.to_dim(), 20.to_dim()]).into())?;
    // Pulsification operates on the typed model. Don't call `into_optimized()`
    // here — that decomposes Conv into Im2col + OptMatMul and inserts AxisOp
    // wrappers that pulse-v2's RegionTransform inventory doesn't handle.
    let model = model.into_typed()?.into_decluttered()?;

    match PulseV2Model::new(&model, s.clone()) {
        Ok(_) => {
            println!("PulseV2 walked hey_snips successfully");
            Ok(())
        }
        Err(e) => {
            panic!("PulseV2 failed on hey_snips: {:#}", e);
        }
    }
}
