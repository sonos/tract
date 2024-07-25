use tract_core::internal::*;


pub fn plan_options_from_subcommand(
    sub_matches: &clap::ArgMatches,
) -> TractResult<PlanOptions> {
    let skip_order_opt_ram: bool = sub_matches.is_present("skip-order-opt-ram");

    let mut options = PlanOptions::default();
    options.skip_order_opt_ram = skip_order_opt_ram;
    Ok(options)
}
