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
    let allow_random_input: bool =
        params.allow_random_input || sub_matches.is_present("allow-random-input");
    let allow_float_casts: bool =
        params.allow_float_casts || sub_matches.is_present("allow-float-casts");

    Ok(RunParams { tensors_values: tv, allow_random_input, allow_float_casts })
}
