use tract_core::internal::*;
use tract_libcli::tensor::RunParams;
#[cfg(feature = "transformers")]
use tract_transformers::figure_out_causal_llm_b_s_p;

use crate::params::Parameters;

pub fn run_params_from_subcommand(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
) -> TractResult<RunParams> {
    let mut tv = params.tensors_values.clone();

    if let Some(bundle) = sub_matches.get_many::<String>("input-from-npz") {
        for input in bundle {
            let input = input.as_str();
            for tensor in Parameters::parse_npz(input, true, false)? {
                tv.add(tensor);
            }
        }
    }

    if let Some(dir) = sub_matches.get_one::<String>("input-from-nnef") {
        for tensor in Parameters::parse_nnef_tensors(dir, true, false)? {
            tv.add(tensor);
        }
    }

    // We also support the global arg variants for backward compatibility
    #[allow(unused_mut)]
    let mut allow_random_input: bool =
        params.allow_random_input || sub_matches.get_flag("allow-random-input");
    let allow_float_casts: bool =
        params.allow_float_casts || sub_matches.get_flag("allow-float-casts");

    let mut symbols = SymbolValues::default();

    #[cfg(feature = "transformers")]
    if let Some(pp) = sub_matches.get_one::<String>("pp") {
        let value: i64 =
            pp.parse().with_context(|| format!("Can not parse symbol value in --pp {pp}"))?;
        let Some(typed_model) = params.tract_model.downcast_ref::<TypedModel>() else {
            bail!("PP mode can only be used with a TypedModel");
        };
        let (b, s, p) = figure_out_causal_llm_b_s_p(typed_model)?;
        if let Some(b) = b {
            symbols.set(&b, 1);
        }

        ensure!(s.is_some() && p.is_some(), "Could not find LLM symbols in model");
        symbols.set(&p.unwrap(), 0);
        symbols.set(&s.unwrap(), value);
        allow_random_input = true
    }

    #[cfg(feature = "transformers")]
    if let Some(tg) = sub_matches.get_one::<String>("tg") {
        let value: i64 =
            tg.parse().with_context(|| format!("Can not parse symbol value in --tg {tg}"))?;
        let Some(typed_model) = params.tract_model.downcast_ref::<TypedModel>() else {
            bail!("TG mode can only be used with a TypedModel");
        };
        let (b, s, p) = figure_out_causal_llm_b_s_p(typed_model)?;
        if let Some(b) = b {
            symbols.set(&b, 1);
        }

        ensure!(s.is_some() && p.is_some(), "Could not find LLM symbols in model");
        symbols.set(&p.unwrap(), value - 1);
        symbols.set(&s.unwrap(), 1);
        allow_random_input = true
    }

    if let Some(set) = sub_matches.get_many::<String>("set") {
        // Right-hand side is a TDim expression (e.g. `--set T=2*S`).  Parse
        // against the model's symbol scope, then reduce to `i64` with the
        // symbols already set so far (CLI argument order matters when
        // expressions reference other symbols).
        let symbol_scope = params.tract_model.symbols();
        for set in set {
            let set = set.as_str();
            let (sym, value) = set.split_once('=').context("--set expects S=value form")?;
            let dim = tract_core::internal::parse_tdim(symbol_scope, value)
                .with_context(|| format!("--set: parsing TDim expression for {sym}={value}"))?;
            let value: i64 = dim.eval_to_i64(&symbols).with_context(|| {
                format!("--set {sym}={value}: resolving with current symbol values {symbols:?}")
            })?;
            let sym = params.tract_model.get_or_intern_symbol(sym);
            symbols.set(&sym, value);
        }
    }

    let prompt_chunk_size = sub_matches
        .get_one::<String>("prompt-chunk-size")
        .and_then(|chunk_size| chunk_size.parse().ok());
    let drop_partial_pulse = sub_matches.get_flag("drop-partial-pulse");
    Ok(RunParams {
        tensors_values: tv,
        allow_random_input,
        allow_float_casts,
        symbols,
        prompt_chunk_size,
        drop_partial_pulse,
    })
}
