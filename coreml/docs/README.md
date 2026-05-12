# tract-coreml — PR companion docs

Reading order:

1. **[pr-prep.md](./pr-prep.md)** — comprehensive overview: what the PR adds, models tested, op coverage, code composition, performance, upstream items, validation, open issues. The single doc to read if you only read one.
2. **[ops-coverage.pdf](./ops-coverage.pdf)** ([source](./ops-coverage.md)) — printable: the canary corpus + every translator → MIL op mapping + gaps.
3. **[benchmarks.pdf](./benchmarks.pdf)** ([source](./benchmarks.md)) — printable: the headline performance table + the full per-canary measurement matrix with op histograms, ANE notes, methodology.
4. **[tract-upstream-feedback.md](./tract-upstream-feedback.md)** — 8 BLOCKERS / 16 IMPROVEMENTS / 6 ANNOYANCES / 4 OPEN QUESTIONS this work surfaced about tract itself. Confidence-marked. Several entries are candidates for separate focused PRs.
5. **[tract-rnn-preservation-handout.md](./tract-rnn-preservation-handout.md)** — proposal for preserving GRU/LSTM as first-class typed ops in tract IR (one upstream change → four downstream backends benefit, including this PR's CoreML / tract-metal MPS / tract-cuda cuDNN / tract-cpu future linalg fastpath). Not part of this PR; included so you can see the rationale referenced in the PR body's DFN3 discussion.

PDFs were rendered from the `.md` source via:

```sh
pandoc --from=markdown-citations --pdf-engine=typst \
    -V geometry:margin=14mm -V fontsize=9.5pt \
    -o coreml/docs/ops-coverage.pdf coreml/docs/ops-coverage.md
```

If you'd rather review only the `.md` source and skip the PDFs, you can — they're rendered from the same content.
