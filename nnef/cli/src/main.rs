use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use structopt::StructOpt;
use tract_nnef::internal::DocDumper;

fn main() {
    // Collecting user arguments
    let cli_args = CliArgs::from_args();

    // Setting up log level
    let level = match cli_args.verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    std::env::set_var("RUST_LOG", level);
    env_logger::Builder::from_env(env_logger::Env::default()).init();

    if let Err(e) = cli_args.run() {
        log::error!("{e:?}");
        std::process::exit(1)
    }
}

/// Struct used to define NNEF documentation CLI arguments.
#[derive(Debug, StructOpt)]
#[structopt(
    name = "tract NNEF doc command line",
    about = "Command line to generate NNEF documentaion"
)]
pub struct CliArgs {
    #[structopt(short = "v", parse(from_occurrences))]
    pub verbosity: usize,
    /// Path to write to the directory where to write the NNEF documentations
    #[structopt(long = "doc-dir")]
    pub docs_path: PathBuf,
}

impl CliArgs {
    pub fn run(&self) -> Result<()> {
        let registries = vec![
            ("tract-core.nnef", tract_nnef::ops::tract_core()),
            ("tract-resource.nnef", tract_nnef::ops::tract_resource()),
            ("tract-pulse.nnef", tract_pulse::tract_nnef_registry()),
            ("tract-onnx.nnef", tract_onnx_opl::onnx_opl_registry()),
        ];

        for (filename, registry) in registries {
            let path = self.docs_path.join(filename);
            DocDumper::registry_to_path(self.docs_path.join(filename), &registry).with_context(
                || {
                    anyhow!(
                        "Error while dumping NNEF documentation for {:?} registry at path {:?}",
                        registry.id,
                        path
                    )
                },
            )?;
        }
        Ok(())
    }
}
