# CI bench infrastructure

## Principles

* `.github/workflows/bench.yml` builds the `tract` CLI on GitHub-hosted runners and
  runs the bench suite on self-hosted bench machines. Nightly it appends one row per
  run to the `bench-data` branch; on a pull request it compares against that reference
  and posts a single fan-in comment (plus a full table in the run's job summary).
* The suite is `tract bench-suite`, driven by the `.travis/benches.toml` manifest. Each
  `[[bench]]` runs in a fresh child process so the memory readings get a cold process;
  models are fetched over HTTP from the manifest's `base_url`.
* Small embedded boards that can't host a runner or build are driven as *dinghy
  targets*: a sidekick runner cross-runs the static CLI on the board over dinghy's ssh
  transport (per-target coordinates live in the sidekick's `.dinghy.toml`). These jobs
  are gated by the `BENCH_DINGHY_ENABLED` repository variable.

## Testing locally

Build the CLI with the `bench-suite` feature and run the manifest against the model
mirror (default: the public bucket in `benches.toml`):

```
cargo run -p tract-cli --features bench-suite -- \
    bench-suite --manifest .travis/benches.toml --skip-runtimes
```

`--skip-runtimes` keeps it to the plain-CPU net benches; drop it to also sweep the
accelerator/LLM benches on whatever backends the machine has.

## Model mirror (http)

Benches fetch each model over HTTP from `base_url`. To serve them from a local mirror,
sync the model bucket to a directory and expose it with any static file server, e.g.
nginx:

```
server {
    root /home/user/models/;
    location / { }
}
```

Point a run at it with `--base-url http://<mirror>/` (or the `TRACT_BENCH_BASE_URL`
knob). `--no-cache` streams each model straight from the mirror instead of caching it
on disk, for read-only targets.
