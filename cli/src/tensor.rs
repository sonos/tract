use tract_core::internal::*;
use tract_libcli::tensor::RunParams;

use crate::params::Parameters;

pub fn run_params_from_subcommand(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
) -> TractResult<RunParams> {
    let mut tv = params.tensors_values.clone();

    if let Some(bundle) = sub_matches.values_of("input-from-npz") {
        for input in bundle {
            for tensor in Parameters::parse_npz(input, true, false)? {
                tv.add(tensor);
            }
        }
    }

    if let Some(dir) = sub_matches.value_of("input-from-nnef") {
        for tensor in Parameters::parse_nnef_tensors(dir, true, false)? {
            tv.add(tensor);
        }
    }

    // We also support the global arg variants for backward compatibility
    let mut allow_random_input: bool =
        params.allow_random_input || sub_matches.is_present("allow-random-input");
    let allow_float_casts: bool =
        params.allow_float_casts || sub_matches.is_present("allow-float-casts");

    let mut symbols = SymbolValues::default();

    if let Some(pp) = sub_matches.value_of("pp") {
        let value: i64 =
            pp.parse().with_context(|| format!("Can not parse symbol value in --pp {pp}"))?;
        let Some(typed_model) = params.tract_model.downcast_ref::<TypedModel>() else {
            bail!("PP mode can only be used with a TypedModel");
        };
        let (b, s, p) = crate::llm::figure_out_b_s_p(typed_model)?;
        if let Some(b) = b {
            symbols.set(&b, 1);
        }
        symbols.set(&p, 0);
        symbols.set(&s, value);
        allow_random_input = true
    }

    if let Some(tg) = sub_matches.value_of("tg") {
        let value: i64 =
            tg.parse().with_context(|| format!("Can not parse symbol value in --tg {tg}"))?;
        let Some(typed_model) = params.tract_model.downcast_ref::<TypedModel>() else {
            bail!("TG mode can only be used with a TypedModel");
        };
        let (b, s, p) = crate::llm::figure_out_b_s_p(typed_model)?;
        if let Some(b) = b {
            symbols.set(&b, 1);
        }
        symbols.set(&p, value - 1);
        symbols.set(&s, 1);
        allow_random_input = true
    }

    if let Some(set) = sub_matches.values_of("set") {
        for set in set {
            let (sym, value) = set.split_once('=').context("--set expect S=12 form")?;
            let sym = params.tract_model.get_or_intern_symbol(sym);
            let value: i64 = value
                .parse()
                .with_context(|| format!("Can not parse symbol value in set {set}"))?;
            symbols.set(&sym, value);
        }
    }

    Ok(RunParams { tensors_values: tv, allow_random_input, allow_float_casts, symbols })
}
