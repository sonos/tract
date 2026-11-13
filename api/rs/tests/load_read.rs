#![cfg(feature = "unstable-load-read")]
use tract::prelude::*;

#[test]
fn load_read_streams_tar_and_tar_gz() -> anyhow::Result<()> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../harness/nnef-test-cases/deploy-pool");
    let nnef = tract::nnef()?;
    let model = nnef.load(path)?;
    let dir = tempfile::tempdir()?;
    let tar = dir.path().join("model.tar");
    nnef.write_model_to_tar(&tar, &model)?;
    let tar_gz = dir.path().join("model.tar.gz");
    nnef.write_model_to_tar_gz(&tar_gz, &model)?;
    for archive in [tar, tar_gz] {
        let reloaded = nnef.load_read(std::fs::File::open(archive)?)?;
        assert_eq!(reloaded.input_fact(0)?.to_string(), model.input_fact(0)?.to_string());
        assert_eq!(reloaded.output_fact(0)?.to_string(), model.output_fact(0)?.to_string());
    }
    Ok(())
}
