use anyhow::Context;
use anyhow::Result;
use fs2::FileExt;
use s3::creds::Credentials;
use s3::Bucket;
use s3::Region;
use s3::serde_types::Object;
use serde::{Deserialize, Deserializer};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Deserialize, Debug)]
struct Config {
    #[serde(default = "default_workdir")]
    workdir: PathBuf,
    #[serde(default = "default_region", deserialize_with = "deser_region")]
    region: Region,
    #[serde(default = "default_bucket")]
    s3_bucket: String,
    #[serde(default = "default_tasks")]
    s3_tasks: String,
    platform: String,
}

fn default_workdir() -> PathBuf {
    dirs::home_dir().expect("HOME does not exist").join("tract-minion")
}

fn default_region() -> Region {
    Region::UsEast1
}

fn default_bucket() -> String {
    "tract-ci-builds".to_string()
}

fn default_tasks() -> String {
    "tasks/".to_string()
}

fn deser_region<'de, D>(d: D) -> Result<Region, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let s: &str = Deserialize::deserialize(d)?;
    s.parse().map_err(D::Error::custom)
}

fn config_path() -> PathBuf {
    std::env::var("TRACT_MINION_CONFIG").map(|s| PathBuf::from(s)).unwrap_or_else(|_| {
        dirs::home_dir().context("HOME does not exist").unwrap().join(".minion.toml")
    })
}

fn read_config(path: impl AsRef<Path>) -> Result<Config> {
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Opening config {:?}", path.as_ref()))?;
    let config = toml::from_str(&text)
        .with_context(|| format!("Parsing configuration file {:?}", path.as_ref()))?;
    Ok(config)
}

fn consider_task(config: &Config, bucket: &Bucket, task: &Object) -> Result<()> {
    eprintln!("{}", task.key);
    Ok(())
}

fn run(config: &Config) -> Result<()> {
    log::info!("Running...");
    std::thread::sleep(Duration::from_secs(10));
    let credentials = Credentials::default().unwrap();
    let bucket = Bucket::new(&config.s3_bucket, config.region.clone(), credentials)?;
    let tasks_prefix = std::path::PathBuf::from(&config.s3_tasks).join(&config.platform).join("");
    let objects_parts =
        bucket.list_blocking(tasks_prefix.to_str().unwrap().to_string(), Some("/".to_string()))?;
    for parts in objects_parts {
        for item in parts.0.contents {
            consider_task(config, &bucket, &item)?;
        }
        //        anyhow::bail!("Non-200 code got from S3 on listing tasks");
    }
    Ok(())
}

fn main_loop() -> Result<()> {
    let cf_path = config_path();
    log::info!("Reading config from {:?}", cf_path);
    let config = read_config(cf_path)?;
    log::debug!("{:?}", config);
    let lock = config.workdir.join("lock");
    log::info!("Locking {:?}", lock);
    std::fs::create_dir_all(&config.workdir)?;
    let lock =
        std::fs::File::create(&lock).with_context(|| format!("Creating lock file {:?}", lock))?;
    loop {
        if let Ok(_) = lock.try_lock_exclusive() {
            log::info!("Lock obtained, starting for good.");
            if let Err(e) = run(&config) {
                eprintln!("{:?}", e);
            }
        } else {
            log::info!("Already locked, bailing out...");
            std::thread::sleep(Duration::from_secs(1));
        };
    }
}

fn main() {
    env_logger::init_from_env("TRACT_MINION_LOG");
    if let Err(e) = main_loop() {
        eprintln!("{:?}", e);
        std::process::exit(1);
    }
}
