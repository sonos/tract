# Speak some text in the browser

Supertone's Supertonic reads text aloud through tract compiled to WebAssembly,
on WebGPU or on the CPU. The page offers ten voices, the languages of the
chosen release, three quality levels (flow-matching steps) and three speeds.
Long text is spoken a few sentences at a time, each part played as soon as it
is ready. The text never leaves the machine.

## How it works

The four graphs are Supertone's own ONNX files, fetched from Hugging Face at a
pinned revision and kept in the browser cache. Nothing is converted or
re-hosted.

| Release | Repository | Languages | Download |
| --- | --- | --- | --- |
| Supertonic 3 | `supertone-oss-archive/supertonic-3` @ `aafc6e32` | 31 | 398 MB |
| Supertonic 2 | `supertone-oss-archive/supertonic-2` @ `7edbcd0b` | 5 | 263 MB |

`site/models.mjs` has the full revisions and byte sizes.

`src/engine.rs` is the whole pipeline: the duration predictor sets the length
of the speech, the text encoder reads the characters, the vector estimator
turns noise into a latent in a few steps, and the vocoder makes 44.1 kHz audio.
Each graph is prepared once with symbolic lengths (the only assertion is
`text_length >= 1`), so a new sentence costs no preparation. `src/text.rs` is
the text front-end of Supertone's reference code: Unicode normalization,
symbol clean-up, language tags, and splitting into chunks. The noise comes from
a seeded splitmix64 generator, so a seed gives the same audio natively and in
the browser.

On WebGPU every op of the four graphs runs on the device; only the masks
derived from the inputs are computed on the CPU before the graph starts.

`supertonic-native <model dir> <voice> <lang> <text> <out.wav>` runs the same
pipeline natively, on a snapshot of either repository. `SUPERTONIC_RUNTIME`
picks `cpu` or `wgpu`.

The crate uses the public `tract` facade with `onnx`, `wgpu` and
`unstable-wgpu-async`, and no default features.

## License

The weights are under the BigScience OpenRAIL-M license, whose use
restrictions (its Attachment A) pass on to whoever uses the demo: the page says
so, links the license, and labels the audio as synthesized speech.

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`dist/` is the static site. `build.sh` writes `build.json`, the commit and date
the page shows next to its source link.
