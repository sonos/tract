use std::collections::HashMap;
use std::io::{BufRead, Write};

const SETS: &[&str] = &["node", "real", "simple", "pytorch-operator", "pytorch-converted"];
const VERSIONS: &[&str] = &["1.4.1", "1.5.0", "1.6.0", "1.7.0", "1.8.1", "1.9.0", "1.10.2", "1.11.0", "1.12.0", "1.13.0"];

// const SETS: &[&str] = &["node"];
// const VERSIONS: &[&str] = &["1.4.1"];

fn run_set(set: &str, ver: &str) -> HashMap<String, usize> {
    let filter = format!("{set}_{ver}::").replace(['.', '-'].as_ref(), "_");
    let mut command = std::process::Command::new("cargo");
    command.arg("test").arg("--all-features");
    if set == "real" {
        command.arg("--release");
    }
    command.arg("--").arg("--ignored").arg(filter);
    let output = command.output().unwrap();
    let mut unexpected: HashMap<String, usize> = HashMap::default();
    for line in std::io::BufReader::new(&mut &*output.stdout).lines() {
        let line = line.unwrap();
        if line.ends_with("ok") {
            let test_id =
                line.split_whitespace().nth(1).unwrap().split("::").nth(2).unwrap().to_string();
            let level = line.split_whitespace().nth(1).unwrap().split("::").nth(1).unwrap();
            let level = match level {
                "nnef" => 3,
                "optim" => 2,
                "plain" => 1,
                _ => panic!(),
            };
            let entry = unexpected.entry(test_id.clone()).or_insert(0);
            *entry = (*entry).max(level);
        }
    }
    unexpected
}

fn process_unexpected(set: &str, ver: &str, unexpected: HashMap<String, usize>) {
    let file = format!("{set}-{ver}.txt");
    eprintln!("## {file} ##");
    let mut specs: HashMap<String, String> = HashMap::new();
    for line in std::fs::read_to_string(&file).unwrap().lines() {
        let test = line.split_whitespace().next().unwrap().to_string();
        let entry = specs.entry(test).or_default();
        if entry.len() < line.len() {
            *entry = line.to_string();
        }
    }
    for (test_id, level) in unexpected.into_iter() {
        eprintln!("* {test_id} level: {level}");
        let spec = specs
            .entry(test_id.to_string())
            .or_insert_with(|| format!("{test_id} not-nnef not-typable"));
        if level >= 3 {
            *spec =
                spec.split_whitespace().filter(|t| t != &"not-nnef").collect::<Vec<_>>().join(" ");
        }
        if level >= 2 {
            *spec = spec
                .split_whitespace()
                .filter(|t| t != &"not-typable")
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    let mut file = std::fs::OpenOptions::new().write(true).truncate(true).open(file).unwrap();
    let mut specs: Vec<String> = specs.into_iter().map(|e| e.1).collect();
    specs.sort();
    let buffer = specs.join("\n");
    file.write_all(buffer.as_bytes()).unwrap();
}

fn main() {
    let mut sets: HashMap<(&str, &str), _> = HashMap::default();
    for &set in SETS {
        for &ver in VERSIONS {
            eprintln!("Running {set} {ver}");
            sets.insert((set, ver), run_set(set, ver));
        }
    }
    for ((set, ver), unexpected) in sets.into_iter() {
        process_unexpected(set, ver, unexpected);
    }
}
