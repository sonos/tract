use anyhow::Context;
use anyhow::Result;
use flate2::read::GzEncoder;
use fs2::FileExt;
use s3::creds::Credentials;
use s3::serde_types::Object;
use s3::Bucket;
use s3::Region;
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Deserialize, Debug)]
struct Config {
    id: String,
    #[serde(default = "default_workdir")]
    workdir: PathBuf,
    #[serde(default = "default_region", deserialize_with = "deser_region")]
    region: Region,
    aws_credentials: Option<AwsCredentials>,
    #[serde(default = "default_bucket")]
    s3_bucket: String,
    #[serde(default = "default_tasks")]
    s3_tasks: String,
    #[serde(default = "default_logs")]
    s3_logs: String,
    platform: String,
    graphite: Option<Graphite>,
    #[serde(default = "default_idle_sleep_secs")]
    idle_sleep_secs: usize,
}

#[derive(Deserialize, Debug)]
struct Graphite {
    host: String,
    port: u16,
    prefix: String,
}

#[derive(Deserialize)]
struct AwsCredentials {
    access_key: String,
    secret_key: String,
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AwsCredentials {{ access_key: {}, secret_key: <...> }}", self.access_key)
    }
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
    "tasks".to_string()
}

fn default_logs() -> String {
    "logs".to_string()
}

fn default_idle_sleep_secs() -> usize {
    5 * 60
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

fn do_task(config: &Config, bucket: &Bucket, task: &Object, task_name: &str) -> Result<()> {
    log::info!("Retrieving task {}", task.key);
    let task_dir = config.workdir.join("current");
    if task_dir.exists() {
        std::fs::remove_dir_all(&task_dir)?;
    }
    std::fs::create_dir_all(&task_dir)?;
    let (reader, mut writer) = pipe::pipe_buffered();
    let task_dir_2 = task_dir.clone();
    let thr = std::thread::spawn(move || {
        let uncompressed = flate2::read::GzDecoder::new(reader);
        tar::Archive::new(uncompressed).unpack(task_dir_2)
    });
    bucket.get_object_stream_blocking(&task.key, &mut writer)?;
    if thr.join().is_err() {
        anyhow::bail!("Failed to untar")
    }
    let vars_file = task_dir.join(task_name).join("vars");
    log::info!("Reading vars...");
    let mut vars: HashMap<String, String> = HashMap::default();
    for line in std::fs::read_to_string(vars_file)?.lines() {
        if line.starts_with("export ") {
            let mut pair = line.split_whitespace().nth(1).unwrap().split("=");
            vars.insert(pair.next().unwrap().to_string(), pair.next().unwrap().to_string());
        }
    }
    let mut cmd = std::process::Command::new("sh");
    cmd.current_dir(task_dir.join(task_name))
        .envs(&vars)
        .arg("-c")
        .arg("./entrypoint.sh 2> stderr.log > stdout.log");
    log::info!("Running {:?}", cmd);
    let status = cmd.status()?;
    log::info!("Script ran: {:?}", status);
    for log in &["stderr.log", "stdout.log"] {
        let local_path = task_dir.join(task_name).join(log);
        if local_path.exists() {
            let s3name = Path::new(&config.s3_logs)
                .join(&config.id)
                .join(task_name)
                .join(log)
                .with_extension("gz");
            log::info!("Uploading {} to {:?}", log, s3name);
            let mut gz =
                GzEncoder::new(std::fs::File::open(&local_path)?, flate2::Compression::default());
            let mut content = vec![];
            gz.write_all(&mut content)?;
            bucket.put_object_blocking(s3name.to_str().unwrap(), &content)?;
        } else {
            log::info!("Could not find {}", log);
        }
    }
    let metrics_files = task_dir.join(task_name).join("metrics");
    if metrics_files.exists() {
        if let Some(gr) = &config.graphite {
            let prefix = format!(
                "{}.{}.{}.{}",
                gr.prefix, config.platform, config.id, vars["TRAVIS_BRANCH_SANE"]
            )
            .replace("-", "_");
            let mut socket = TcpStream::connect((gr.host.clone(), gr.port))?;
            let ts = &vars["TIMESTAMP"];
            for line in std::fs::read_to_string(metrics_files)?.lines() {
                let mut tokens = line.split_whitespace();
                writeln!(
                    socket,
                    "{}.{} {} {}",
                    prefix,
                    tokens.next().unwrap().replace("-", "_"),
                    tokens.next().unwrap(),
                    ts
                )?;
            }
        }
    }
    Ok(())
}

fn consider_task(config: &Config, bucket: &Bucket, task: &Object) -> Result<bool> {
    let task_path = Path::new(&task.key);
    let task_name = task_path
        .file_stem()
        .context("Filename can not be splitted")?
        .to_str()
        .unwrap()
        .to_string();
    let done_file = config.workdir.join("taskdone").join(&task_name);
    if done_file.exists() {
        return Ok(false);
    }
    if let Err(e) = do_task(config, bucket, task, &task_name)
        .with_context(|| format!("Running task {}", task.key))
    {
        eprintln!("{:?}", e);
    }
    std::fs::File::create(done_file)?;
    Ok(true)
}

fn run(config: &Config) -> Result<bool> {
    std::thread::sleep(Duration::from_secs(10));
    let credentials = Credentials::new(
        config.aws_credentials.as_ref().map(|cred| &*cred.access_key),
        config.aws_credentials.as_ref().map(|cred| &*cred.secret_key),
        None,
        None,
        None,
    )
    .unwrap();
    let bucket = Bucket::new(&config.s3_bucket, config.region.clone(), credentials)?;
    let tasks_prefix = std::path::PathBuf::from(&config.s3_tasks).join(&config.platform).join("");
    let objects_parts =
        bucket.list_blocking(tasks_prefix.to_str().unwrap().to_string(), Some("/".to_string()))?;
    let mut done_anything = false;
    for parts in objects_parts {
        for item in parts.0.contents {
            done_anything = consider_task(config, &bucket, &item)? || done_anything;
        }
    }
    Ok(done_anything)
}

fn main_loop() -> Result<()> {
    let cf_path = config_path();
    log::info!("Reading config from {:?}", cf_path);
    let config = read_config(cf_path)?;
    log::debug!("{:?}", config);
    let lock = config.workdir.join("lock");
    log::info!("Locking {:?}", lock);
    std::fs::create_dir_all(&config.workdir)?;
    std::fs::create_dir_all(&config.workdir.join("taskdone"))?;
    let lock =
        std::fs::File::create(&lock).with_context(|| format!("Creating lock file {:?}", lock))?;
    loop {
        if let Ok(_) = lock.try_lock_exclusive() {
            log::info!("Lock taken, fetching task list");
            match run(&config) {
                Ok(done_something) => {
                    if !done_something {
                        let dur = Duration::from_secs(config.idle_sleep_secs as _);
                        log::info!("No task left, sleeping for {:?}", dur);
                        std::thread::sleep(dur);
                    }
                }
                Err(e) => {
                    eprintln!("{:?}", e);
                }
            }
        } else {
            log::info!("Already locked, retry in 1 sec...");
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
