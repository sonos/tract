# CLAUDE.md — tract contributor rules (read fully; this is auto-loaded)

tract is Sonos' Rust NN inference engine. Full guide: AGENTS.md. Architecture/
reference: doc/. This file is the rules an agent must follow to contribute cleanly.

## Before you commit
- Format with stable rustfmt: `cargo fmt --all`. The repo's `rust-toolchain.toml`
  pins the stable channel, so bare `cargo fmt` picks the same rustfmt CI checks
  against — don't override the toolchain. Metal files too, on Linux.
- `cargo clippy --workspace` clean.

## Commit messages
- One short paragraph: what was wrong + the fix. Nothing else.
- No consequence chains ("X broke Y broke Z"), no "Result:/Symptom:" sections,
  no bullet lists of every place the bug surfaced.

## Inline comments
- Default to NONE. Names carry the meaning. A comment signals a hidden
  constraint / invariant / workaround — not narration.
- Never describe the diff or history ("used to be X", "previously…"). Comments
  describe current code only.
- No section-banner comments; split into functions instead.

## Doc comments (`///` / `//!`)
- DO add a concise one on public / non-trivial items — ops, declutter & codegen
  passes, public fns. State what it is, its contract, valid inputs, and which
  rules it interacts with. This is the one place to be more generous than before.
- Same anti-narration rule: document the *current contract*, not benchmarks,
  perf numbers, issue numbers, or history ("Measured on…", "Regression:…").

## How to change a model
- Use `TypedModelPatch` / `Rewriter` / `ModelTransform`. Do NOT hand-roll
  model-walk loops or rebuild a fresh TypedModel.
- Don't touch `pulse` / `pulse-opl` casually — subtle streaming invariants.

## Public API
- The public surface is `api/rs/src/lib.rs`. Check there, not internal `pub`
  items. Apps/examples/bindings use `api/rs` only.

## Tests
- Add op tests to the `suite-*` crates; add synthetic NNEF cases under
  `harness/nnef-test-cases/` (driven by `runme.sh` + `--assert-output-bundle`).
  If the CLI can't express the assertion, extend the CLI.
- No new Rust integration tests for the above; no mocking internals — prefer
  real model round-trips.

## Idioms / avoid
- No new `unsafe` outside linalg kernels without explicit permission.
  No abstraction beyond the task — three similar lines beat a premature helper.

## Pull requests
- Open with a 1–2 sentence summary of what and why.
- Follow-up questions/review replies are handled by a HUMAN, not the bot. The
  maintainer wants to talk to the author, not prompt an LLM.
