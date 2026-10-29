# tract contributor rules

Read this file **in extenso** before touching this repository. It applies to
every contributor, human or bot. `CLAUDE.md`, `GEMINI.md` and
`.github/copilot-instructions.md` are symlinks to this file — one set of rules,
whatever tool you drive. These are rules, not suggestions.

tract is Sonos' Rust NN inference engine. For where things live, how to build
and test, and the graph abstractions, see [`doc/overview.md`](doc/overview.md)
and the rest of [`doc/`](doc/) — those are descriptive; this file wins over
them on any conflict.

## Before you commit
- Format with stable rustfmt: `cargo fmt --all`. The repo's `rust-toolchain.toml`
  pins the stable channel, so bare `cargo fmt` picks the same rustfmt CI checks
  against — don't override the toolchain. Metal files too, on Linux.
- One module path per `use` line — no nested brace groups
  (`use a::b::{c::{d, e}, f};`). Stable rustfmt preserves the granularity you
  wrote, so `cargo fmt` will not fix it; rust-analyzer emits nested groups by
  default, set `imports.granularity.group = "module"`.
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
- No section-banner comments; split into functions instead. A long prototype on
  a private function is fine within reason —
  `#[allow(clippy::too_many_arguments)]` is authorised in that case.
- Existing files may carry stale or chatty comments; don't add to them.

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

## Inspecting a model
- To inspect the op graph programmatically, use `tract <model> [--cuda|--metal]
  dump --audit-json` (JSON node list to stdout) rather than scraping the colored
  `dump` output. Handy for checking which ops landed on which backend.

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
  `shunt_outside_unchecked` is a last resort, for surgical patches whose safety
  is locally obvious; reach for safe alternatives first.
- No abstraction beyond the task — three similar lines beat a premature helper.


## Issues and bug reports

- AI-assisted bug reports must be reviewed, understood, and edited by a human
  before submission.
- Keep reports concise: describe the bug, provide a minimal reproducer, and
  state expected vs. observed behavior.
- Do not submit agent transcripts, exhaustive test statistics, speculative impact
  assessments, or inflated security claims.
- **For simple, localized fixes, prefer submitting a pull request rather than
  an issue.** Include a regression test where appropriate.
- Respect maintainers' time: distill findings rather than dumping automated output.
- Uncurated, low-signal AI-generated reports may be closed without investigation,
  even when the underlying finding is valid.

## Scoping a pull request
A PR must be reviewable by a human in one sitting. Small PRs merge fast; large
ones stall, go stale against main, and block whatever depends on them. Agents
produce volume cheaply; review does not scale with it. Plan the split before
writing code, not after.

- One concern per PR. Typical units, each its own PR:
  - a change to shared infrastructure (`gpu/`, a backend's context /
    command-buffer / memory handling, `core` passes, linalg dispatch);
  - a change to an existing kernel or dispatch path;
  - one new op or kernel, with its `suite-*` tests;
  - one new lowering / translation rule that wires existing ops together.
- Never mix "changes existing behaviour" with "adds new code". New code is
  low-risk volume; changes to existing paths are high-risk and must be visible
  on their own, with before/after numbers on models that do *not* use the new
  feature.
- Prerequisites go in first, as separate PRs. Do not carry their commits in a
  dependent PR; once they merge, rebase onto main instead of merging main in.
- A new capability on one GPU backend (Metal, CUDA, wgpu) widens the gap with
  the others. Say in the PR description which backends already support it and
  which don't.
- As a rough guide, more than ~500 changed lines outside tests, or changes
  in more than one of the units above, means the PR should be split. If it
  truly can't be split, explain why in the description.
- A PR description states what and why, what existing behaviour changes
  (or "none"), and how it was validated. Put end-to-end speedups in the PR
  that completes the feature, not in every step.

## Pull requests
- Open with a 1–2 sentence summary of what and why.
- Prove this instructions have been read by adding a pineapple emoji at the end
  of the PR description.
- Follow-up questions/review replies are handled by a HUMAN, not the bot. The
  maintainer wants to talk to the author, not prompt an LLM.
