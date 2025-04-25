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
use std::fs::File;
use std::io::Write;
use std::net::TcpStream;
use std::os::unix::prelude::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicI32;
use std::time::Duration;
use wait_timeout::ChildExt;

lazy_static::lazy_static! {
    static ref CHILD: std::sync::Arc<AtomicI32> = std::sync::Arc::new(AtomicI32::new(0));
}

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
    #[serde(default = "default_products")]
    s3_products: String,
    platform: String,
    graphite: Option<Graphite>,
    #[serde(default = "default_idle_sleep_secs")]
    idle_sleep_secs: usize,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default = "default_timeout_runtime_secs")]
    timeout_runtime_secs: usize,
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

fn default_products() -> String {
    "products".to_string()
}

fn default_idle_sleep_secs() -> usize {
    5 * 60
}

fn default_timeout_runtime_secs() -> usize {
    300
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
    if let Ok(c) = std::env::var("TRACT_MINION_CONFIG") {
        PathBuf::from(c)
    } else if std::path::Path::new("minion.toml").exists() {
        PathBuf::from("minion.toml")
    } else {
        dirs::home_dir().context("HOME does not exist").unwrap().join(".minion.toml")
    }
}

fn read_config(path: impl AsRef<Path>) -> Result<Config> {
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Opening config {:?}", path.as_ref()))?;
    let config = toml::from_str(&text)
        .with_context(|| format!("Parsing configuration file {:?}", path.as_ref()))?;
    Ok(config)
}

fn dl_task(task_name: &str) -> Result<()> {
    let config = config()?;
    let bucket = bucket(&config)?;
    let task_url = std::path::PathBuf::from(&config.s3_tasks)
        .join(&config.platform)
        .join(task_name)
        .with_extension("tgz");
    log::info!("Retrieving task {}", task_name);
    let task_dir = config.workdir.join("current");
    if task_dir.exists() {
        std::fs::remove_dir_all(&task_dir)?;
    }
    std::fs::create_dir_all(&task_dir)?;
    let (reader, mut writer) = pipe::pipe_buffered();
    let task_dir_2 = task_dir.clone();
    let bucket_2 = bucket.clone();
    std::thread::spawn(move || {
        bucket_2.get_object_stream_blocking(task_url.to_str().unwrap(), &mut writer).unwrap();
    });
    let uncompressed = flate2::read::GzDecoder::new(reader);
    tar::Archive::new(uncompressed).unpack(task_dir_2)?;
    Ok(())
}

fn vars(task_name: &str) -> Result<HashMap<String, String>> {
    let config = config()?;
    log::info!("Running task {}", task_name);
    let task_dir = config.workdir.join("current");
    let vars_file = task_dir.join(task_name).join("vars");
    let mut vars: HashMap<String, String> = config.env.clone();
    if vars_file.exists() {
        log::info!("Reading vars...");
        for line in std::fs::read_to_string(vars_file)?.lines() {
            if line.starts_with("export ") {
                let mut pair = line.split_whitespace().nth(1).unwrap().split("=");
                vars.insert(pair.next().unwrap().to_string(), pair.next().unwrap().to_string());
            }
        }
    } else {
        log::info!("No vars file");
    }
    Ok(vars)
}

fn run_task(task_name: &str) -> Result<()> {
    let config = config()?;
    let bucket = bucket(&config)?;
    log::info!("Running task {}", task_name);
    let task_dir = config.workdir.join("current");
    let vars: HashMap<String, String> = vars(task_name)?;
    let mut cmd = std::process::Command::new("sh");
    cmd.current_dir(task_dir.join(task_name))
        .envs(&vars)
        .arg("-c")
        .arg("./entrypoint.sh 2> stderr.log > stdout.log");
    log::info!("Running {:?}", cmd);
    let status = cmd.spawn()?.wait_timeout(Duration::from_secs(config.timeout_runtime_secs as _));
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
            bucket
                .put_object_blocking(s3name.to_str().unwrap(), &content)
                .with_context(|| format!("uploading {}", s3name.to_str().unwrap()))?;
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
            let mut socket = TcpStream::connect((gr.host.clone(), gr.port))
                .with_context(|| format!("Opening socket to {:?}", gr))?;
            let ts = &vars["TIMESTAMP"];
            for line in std::fs::read_to_string(metrics_files)?.lines() {
                let mut tokens = line.split_whitespace();
                let graphite = format!(
                    "{}.{} {} {}",
                    prefix,
                    tokens.next().unwrap().replace("-", "_"),
                    tokens.next().unwrap(),
                    ts
                );
                log::trace!("Sending to graphite: {graphite}");
                writeln!(socket, "{graphite}").context("Writing to graphite socket")?;
            }
        }
    }
    let product_dir = task_dir.join(task_name).join("product");
    if product_dir.exists() {
        let tar_name = format!("{}.{}", task_name, config.id);
        let s3name =
            Path::new(&config.s3_products).join(&config.id).join(task_name).with_extension("tgz");
        let mut buf = vec![];
        let tgz = flate2::write::GzEncoder::new(&mut buf, flate2::Compression::default());
        tar::Builder::new(tgz).append_dir_all(tar_name, product_dir)?;
        bucket
            .put_object_blocking(s3name.to_str().unwrap(), &buf)
            .with_context(|| format!("uploading {}", s3name.to_str().unwrap()))?;
    }
    Ok(())
}

#[derive(Debug)]
struct Timeout;
impl std::error::Error for Timeout {}

impl std::fmt::Display for Timeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Timeout")
    }
}

fn subcommand(command: &str, task_name: &str, timeout: Duration) -> Result<()> {
    let mut process = unsafe {
        std::process::Command::new(std::env::args().next().unwrap())
            .arg(command)
            .arg(&task_name)
            .pre_exec(|| {
                if libc::setpgid(0 as i32, 0 as i32) < 0 {
                    libc::perror(std::ptr::null());
                }
                Ok(())
            })
            .spawn()?
    };
    CHILD.store(process.id() as i32, std::sync::atomic::Ordering::SeqCst);
    match process.wait_timeout(timeout)? {
        Some(status) => {
            log::info!("dl-task {} return status {}", task_name, status);
            if status.success() {
                Ok(())
            } else {
                anyhow::bail!("running {} {}, got {:?}", command, task_name, status)
            }
        }
        None => {
            log::warn!("{} {} timeout after {:?}", command, task_name, timeout,);
            unsafe { libc::kill(-(process.id() as i32), libc::SIGTERM) };
            process.wait()?;
            CHILD.store(0, std::sync::atomic::Ordering::SeqCst);
            Err(Timeout).with_context(|| format!("running {} {}", command, task_name))
        }
    }
}

fn consider_task(config: &Config, task: &Object) -> Result<bool> {
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
    for attempt in 0.. {
        match subcommand("dl-task", &task_name, Duration::from_secs(60)) {
            Err(e) if e.root_cause().is::<Timeout>() && attempt < 5 => continue,
            Err(e) => Err(e)?,
            Ok(()) => break,
        }
    }
    let vars = vars(&task_name)?;
    let timeout = vars.get("TIMEOUT").map(|s| &**s).unwrap_or("1800").parse()?;
    let _ = subcommand("run-task", &task_name, Duration::from_secs(timeout));
    std::fs::File::create(&done_file)?;
    Ok(true)
}

fn run(config: &Config) -> Result<bool> {
    std::thread::sleep(Duration::from_secs(10));
    let bucket = bucket(config)?;
    let tasks_prefix = std::path::PathBuf::from(&config.s3_tasks).join(&config.platform).join("");
    let objects_parts =
        bucket.list_blocking(tasks_prefix.to_str().unwrap().to_string(), Some("/".to_string()))?;
    let mut done_anything = false;
    for parts in objects_parts {
        for item in parts.0.contents {
            done_anything = consider_task(config, &item)? || done_anything;
        }
    }
    Ok(done_anything)
}

fn config() -> Result<Config> {
    let cf_path = config_path();
    log::info!("Reading config from {:?}", cf_path);
    let config = read_config(cf_path)?;
    log::debug!("{:?}", config);
    Ok(config)
}

fn bucket(config: &Config) -> Result<Bucket> {
    let credentials = Credentials::new(
        config.aws_credentials.as_ref().map(|cred| &*cred.access_key),
        config.aws_credentials.as_ref().map(|cred| &*cred.secret_key),
        None,
        None,
        None,
    )
    .unwrap();
    let bucket = Bucket::new(&config.s3_bucket, config.region.clone(), credentials)?;
    Ok(bucket)
}

fn main_loop() -> Result<()> {
    let config = config()?;
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

extern "C" fn signal_handler(sig: libc::size_t) -> libc::size_t {
    let child_id = CHILD.load(std::sync::atomic::Ordering::SeqCst);
    if child_id != 0 {
        unsafe {
            libc::kill(-(child_id as i32), sig as _);
        }
    }
    eprintln!("** Caught signal, cleanup...");
    std::process::exit(1);
}

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .parse_env("TRACT_MINION_LOG")
        .init();
    unsafe {
        libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| &**s) == Some("dl-task") {
        let task_id = &args[2];
        log::info!("Worker starting on dl-task {}", task_id);
        if let Err(e) = dl_task(task_id) {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    } else if args.get(1).map(|s| &**s) == Some("run-task") {
        let task_id = &args[2];
        log::info!("Worker starting on run-task {}", task_id);
        if let Err(e) = run_task(task_id) {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    } else if args.get(1).map(|s| &**s) == Some("-d") {
        let _config = config().unwrap();

        log::info!("Deamonizing");
        let stdout = File::create("tract-ci-minion.out").unwrap();
        let stderr = File::create("tract-ci-minion.err").unwrap();

        let daemonize = daemonize::Daemonize::new()
            .working_directory(std::env::current_dir().unwrap())
            .pid_file("tract-ci-minion.pid")
            .stdout(stdout)
            .stderr(stderr);
        daemonize.start().unwrap();
        if let Err(e) = main_loop() {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    } else {
        if let Err(e) = main_loop() {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    }
}
