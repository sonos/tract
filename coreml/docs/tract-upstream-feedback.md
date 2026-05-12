# tract upstream feedback — bugs, improvements, annoyances

— maintained by Ckristian Zoli during tract-coreml development (started 2026-05-09)

> **Purpose.** A running log of friction points encountered while building `tract-coreml` against `sonos/tract`. Each entry captures: what we hit, where we hit it, the workaround we took, and the suggested upstream fix. Eventually surfaces as either separate `sonos/tract` issues, PR-comment discussion items with the maintainer, or in-tree Phase-N closure-doc references.
>
> **Add as we go.** New items go to the bottom of the appropriate section. Don't delete entries when fixed — mark them `RESOLVED ↑` with the fix's PR/commit so the history is preserved.

> **Companion docs (PR-prep)**: see [`pr-prep.md`](./pr-prep.md) §5 for a priority-ranked summary of the highest-leverage items, and [`draft-pr-body.md`](./draft-pr-body.md) for how this list factors into the PR description. Standalone proposal handouts: [`tract-rnn-preservation-handout.md`](./tract-rnn-preservation-handout.md) (preserve `Gru`/`Lstm` as typed ops upstream).

> **Current totals**: §1 has 8 BLOCKERS · §2 has 16 IMPROVEMENTS · §3 has 6 ANNOYANCES · §4 has 4 OPEN QUESTIONS. The B8 blocker (com.microsoft domain) and the I13/I14/I15/I16 improvements were added in Phase 5 from the LLM-class probes (Phi-3 + SmolLM2 + ViT + YOLOv8 + Real-ESRGAN).

---

## How to use this file

When you hit something in tract that surprises you, blocks you, or you have to work around — add an entry under the right section:

- **§1 BLOCKERS** — prevented us from running a model end-to-end until worked around.
- **§2 IMPROVEMENTS** — would significantly simplify backend authoring or unblock cases tract currently handles awkwardly.
- **§3 ANNOYANCES** — papercuts we worked around with little cost. Useful as upstream PR fodder when convenient but not load-bearing.
- **§4 OPEN QUESTIONS FOR THE MAINTAINER** — places where tract's behaviour is consistent but we'd like to understand the rationale (or propose alternatives).

Each entry:

```
### Title — short imperative phrase

**Where we hit it.** Phase X / Milestone Y / [model name].
**Symptom.** What broke or surprised us.
**Workaround.** What we did to make progress.
**Suggested upstream fix.** What would we change in tract, in a sentence.
**Priority for upstream.** P0 (blocker class) / P1 (real impact) / P2 (nice-to-have).
```

---

## §1 BLOCKERS — prevented running until worked around

### B1. ONNX `InferenceFact` override doesn't propagate to runtime-input const-folding

**Where we hit it.** Phase 4 / RVM canary / `rvm_mobilenetv3_fp32.onnx`.
**Symptom.** RVM exports `downsample_ratio` as a runtime F32 input that flows into `Concat([1, 1, downsample_ratio, downsample_ratio]) → Resize.scales`. With `inf.set_input_fact(slot, dt_shape)` + `analyse(true)`, tract knows the input is `[1] f32` but does NOT const-fold the Concat into a literal `[1, 1, 0.375, 0.375]` (because `downsample_ratio` is a runtime input, not a Const, even after the user "binds" it). The Resize's `output_facts()` then fails at `into_typed()`: "Neither sizes nor scales makes sense".
**Workaround.** Pre-bake the ONNX with a Python helper that replaces the `downsample_ratio` input with a `Constant` node holding the desired value. One-time fix per (model, value) combination.
**Suggested upstream fix.** `InferenceFact` could carry a value (not just type+shape), and `set_input_fact` with a value-carrying fact would substitute the Source with a Const at typed-model construction time. Equivalent: a tract-side helper `model.bake_input_to_const(slot, tensor)`.
**Priority for upstream.** P0 — without this, any ONNX model with a "config knob" input that the model graph then uses for shape arithmetic can't be loaded by tract.

### B4. SAM 2 ONNX export omits value_info for FPN boundary convs

**Where we hit it.** Phase 4 / SAM 2 Hiera-Tiny canary / `vietanhdev/segment-anything-2-onnx-models/sam2_hiera_tiny.encoder.onnx`.
**Symptom.** ONNX shape inference itself flags `/conv_s0/Conv_output_0` and `/conv_s1/Conv_output_0` as rank-0 (no dimensions) — the upstream samexporter export forgot to record value_info for two FPN-side convs that sit between the Hiera trunk and the mask decoder. tract's HIR analyzer (`inf.analyse(true)`) then bails: "Failed analyse for node #710 '/conv_s0/Conv' ConvHir: Impossible to unify 0 with 4". `analyse(false)` doesn't help — the rule is enforced regardless. Same workaround pattern as RVM (B1).
**Workaround.** One-time pre-bake helper in Python that injects the two known shapes (`[1,32,256,256]` and `[1,64,128,128]`) as value_info, then re-runs `onnx.shape_inference.infer_shapes`. ~15 LoC in a separate script; output is a `.fixed.onnx` file consumed by the test.
**Suggested upstream fix.** Either (a) push the fix to `vietanhdev/samexporter` so it adds the trailing value_info entries; or (b) tract could fall back to a more lenient inference mode that propagates ranks from the producer Conv's weight shape when downstream fact is rank-0. (b) generalises beyond just SAM 2.
**Priority for upstream.** P1 for tract — affects any PyTorch ONNX export where the exporter forgot some intermediate value_info. Common in community exports.

### B5. NNEF Parakeet/Nemotron load with symbolic shapes that block all backends until manually substituted

**Where we hit it.** Phase 4 / Parakeet TDT v3 (preprocessor / decoder / joint sub-models, NNEF format).
**Symptom.** `tract_nnef::nnef().model_for_path()` returns a TypedModel with symbolic shapes `[B, R, U]` etc. (matching NVIDIA's NNEF export which uses dim-name placeholders for batch/sequence). Without binding, every translator on every backend (CoreML/Metal/CUDA/CPU) skips with "data symbolic shape: B,R,U,640" because they can't compute concrete sizes. The Sonos `tract` CLI handles this implicitly via `--input-from-bundle ...io.npz`, which provides concrete input shapes that drive symbol substitution. But programmatic users have to know to call `model.substitute_symbols()` manually.
**Workaround.** In our Parakeet test, build a `HashMap<Symbol, TDim>` mapping `B → 1, R → 1, U → 1, T → 1, S → 118960` (per-submodel, derived from the io.npz bundle), then call `model.substitute_symbols(&subs)`. Without this: 0 CoremlOps. With: 7 (preprocessor) / 15 (decoder) / 2 (joint).
**Suggested upstream fix.** Either: (a) add a convenience method `tract_nnef::Nnef::load_with_input_shapes(path, &[(name, shape)])` that combines load + symbol substitution in one call; (b) document that NNEF models with symbolic dims need an explicit substitution step before any backend transform, with a code snippet in `tract-nnef` rustdoc; (c) ship a `tract_core::transform::concretize_dims` model transform that derives the symbol bindings from a `[(input_name, shape)]` slice.
**Priority for upstream.** P1 — affects every user loading a Sonos-format NNEF model programmatically. CLI users get this for free via `--input-from-bundle`; library users hit a wall.

### B6. NNEF Nemotron preprocessor `into_decluttered` panics when concretizing audio-length dim

**Where we hit it.** Phase 4 / Nemotron Speech Streaming en 0.6B (preprocessor sub-model, NNEF format).
**Symptom.** Loading `nemotron-streaming.preprocessor.nnef.tgz` and calling `model.substitute_symbols(&[(BATCH, 1), (INPUT_SIGNAL__TIME, 118960)])` then `into_decluttered()` panics inside `featurizer_inputView0_0`: a `Reshape` op rejects `Broadcast([Sym(INPUT_SIGNAL__TIME), Val(118960)])` after substitution because the symbol-still-present-but-equal-to-Val reshape rule didn't normalise. Workaround is to leave `INPUT_SIGNAL__TIME` symbolic and only bind `BATCH` (this is what Parakeet's test does — and it works there because Parakeet's preprocessor reshapes don't hit the same code path). Net effect: the Nemotron preprocessor canary can only be probed with the audio-length axis still symbolic, which means the EinSum/Mul that would have translated falls out (symbolic shape).
**Workaround.** Bind `BATCH=1` only; leave `INPUT_SIGNAL__TIME` symbolic. Result: 0 CoremlOps for Nemotron preprocessor (vs 7 for Parakeet preprocessor with `B=1, S=118960` concretized).
**Suggested upstream fix.** In the `Reshape`/`Broadcast` declutter pattern, treat `Sym(s) == Val(v)` after `substitute_symbols` as resolved (e.g., re-run the sym→val substitution at the `Broadcast`/`Reshape` shape level too, not just at outer fact level). Or add a follow-up `concretize_remaining_symbols` declutter pass.
**Priority for upstream.** P1 — actively blocks the Nemotron canary; will block any NNEF model that uses `Reshape(Broadcast([Sym, Val]))` after symbol substitution.

### B3. `RmsNorm` CPU eval errors with F16 input through declutter

**Where we hit it.** Phase 4 / transformer-prep / `layer_norm_chain_folds_to_single_mil_op` synthetic test.
**Symptom.** Wiring an F16 LayerNorm chain (ReduceSum + Mul(1/N) + Sub + Mul(d,d) + ReduceSum + Mul(1/N) + Add(eps) + Rsqrt + Mul(d,_) + Mul(γ) + Add(β)) and calling `into_decluttered()` collapses the variance portion into a single `RmsNorm` op (declutter pattern in `tract-core::ops::nn::rms_norm`). Then `runnable.run()` errors on the CPU side: `Tensor datum type error: tensor is F32, accessed as F16` during the RmsNorm Add. The CPU Eval path inside `RmsNorm::eval` casts input to F32, computes mean-of-squares, then `Add.eval(_, self.eps, DatumType::F32)` — but `self.eps` is whatever dtype the source `Add(_, eps_const)` had (F16 in our test), so the F32 Add asserts dtype-mismatch. The MIL path is unaffected.
**Workaround.** Test runs CPU on the un-decluttered model and CoreML on the decluttered one. Real ONNX models that come pre-set with F32 eps work fine — the bug only surfaces when a user wires an F16 LayerNorm chain by hand. Still affects anyone testing RmsNorm + F16 in isolation.
**Suggested upstream fix.** In the declutter pattern that creates `RmsNorm`, normalise eps to F32 (`eps.cast_to_dt(DatumType::F32)`) before storing. Or in `RmsNorm::eval`, cast eps to F32 at construction-time-of-eval rather than assuming it's already F32.
**Priority for upstream.** P1 — this is an honest bug; the assertion fires deterministically and would land any user who lowers an F16 LayerNorm and runs CPU eval. Easy fix (one cast call).

### B7. CoreML backend produces vertically-shifted MODNet matte vs. CPU backend on identical model

**Where we hit it.** macOS demo app M3 — MODNet 512² live webcam matting (2026-05-10).
**Symptom.** Same ONNX (`v7-webgl-relighting-worktree/models/onnx/modnet.onnx`), same preprocessing (BGRA → RGB → CHW f16, normalized `[-1, 1]`), same input fact `[1, 3, 512, 512]`. Three backends compared, side-by-side in a 4-pane debug UI (live preview | prep input | alpha matte | composite):

- **TRACT CPU** (model stays F32, no `f32_to_f16`, no `CoremlTransform`): matte aligns **perfectly** with the prep input. Hand is correctly included. ~1 fps.
- **TRACT CoreML** (`f32_to_f16` + `CoremlTransform { compute_units: All }`): matte is **shifted up by ~25% of frame height**, and the hand is **missing entirely**. The silhouette also appears stretched to fill the full 512² vertical extent, regardless of where the person actually is in the prep. 22 ms / 28 fps.
- **TRACT Metal**: errored at compile time before producing output (not yet investigated which op fails translation).

Visually unambiguous when both panes are rendered side-by-side. The CPU matte and the CoreML matte do not agree on where the person is.

**Workaround.** None for the demo path — using TRACT CPU sacrifices ~30× throughput and is unusable for live video. The demo is shipping with the bug documented; M4's backend picker lets reviewers see both side-by-side.
**Suggested upstream fix.** Bisect: (a) run CoreML with the model still F32 (skip `f32_to_f16`) — does shift persist? distinguishes Cast-induced bug from CoreML-translation bug; (b) progressively shrink the set of ops `CoremlTransform` claims (force-skip Resize, then InstanceNorm, etc.) until the matte realigns — points at the offending op translator; (c) compare per-op outputs CPU vs CoreML for a fixed deterministic input (random tensor with a known asymmetric distribution) and find the first divergence. Most likely suspect given the symptoms (vertical shift, lost hand at edge): a `Resize`-translator coordinate-transform-mode mismatch, or an `InstanceNorm` axis-order bug after `f32_to_f16` decomposition.
**Priority for upstream.** P0 — silently produces wrong output, no error surfaced. Any segmentation model with Resize in its decoder will exhibit this.

---

### B2. MediaPipe-style ONNX exports bake symbolic dims into conv output facts

**Where we hit it.** Phase 3 / Milestone E / `mediapipe_selfie.onnx` and `mediapipe_selfie_fp16.onnx`.
**Symptom.** PyTorch-exported models with `dynamic_axes={'pixel_values': {0: 'batch_size'}}` write the symbolic name `batch_size` into the output FACT of every internal Conv (e.g. `Conv_2.output[0].fact = batch_size,16,128,128`). Calling `set_input_fact(0, dt_shape([1, 3, 256, 256]))` overrides the input but tract can't unify the new `Val(1)` with the conv's baked-in `Sym(batch_size)`. Error: `Impossible to unify Val(1) with Sym(batch_size)`.
**Workaround.** **UPDATE 2026-05-11 (SmolLM2 probe):** `inference_model.with_ignore_value_info(true)` before `analyse(true)` skips the strict-unify on value_info entries and lets the user-provided input fact propagate freely. This is what SmolLM2-135M's probe uses to get past the symbolic-batch-vs-Gather-indices clash that aborts analyse otherwise. **The MediaPipe Selfie probe (the original B2 reproducer) hasn't been re-validated with this workaround yet** — worth retrying. Earlier, no workaround had landed; the MediaPipe Selfie probe was `#[ignore]`'d with the symbolic-dim issue documented as deferred.
**Suggested upstream fix.** Either (a) `tract-hir`'s analyse pass should propagate concrete-input substitutions through unification more aggressively (so `set_input_fact` actually refines downstream value_info instead of fighting it); or (b) ship a public helper `model.substitute_symbol("batch_size", 1)` that walks all facts; or (c) make `with_ignore_value_info(true)` the default (and let users opt back in to strict-unify), since `set_input_fact` is the more authoritative source.
**Priority for upstream.** P1 — affects most PyTorch ONNX exports with dynamic shapes, which is increasingly common in production. The `with_ignore_value_info(true)` workaround unblocks SmolLM2-class LLMs; the maintainer-discoverability of this knob is itself an issue (it's a public API on `InferenceModel` but not documented for this use case).

### B8. tract-onnx has no `com.microsoft` operator domain support — entire onnxruntime-genai LLM corpus is unloadable

**Where we hit it.** Phase 5 / Phi-3-mini-4k LLM probe (`microsoft/Phi-3-mini-4k-instruct-onnx :: cuda/cuda-fp16`, 7.6 GB).
**Symptom.** `tract/onnx/src/model.rs:124` does a flat `op_type → builder` lookup that ignores `pbnode.domain`. Microsoft's `onnxruntime-genai` ONNX exporter — used for **every official Phi-2 / Phi-3 / Phi-3.5 / Phi-4 ONNX in the `microsoft/*-onnx` HF repos** — emits fused contrib-domain ops like `com.microsoft.GroupQueryAttention`, `com.microsoft.SkipSimplifiedLayerNormalization`, `com.microsoft.RotaryEmbedding`, `com.microsoft.MultiHeadAttention`, `com.microsoft.MatMulNBits` (quantized). All of them surface as `UnimplementedOp` and `analyse` aborts before the model fully types. On Phi-3-mini, **26% of nodes (96/363) come from this contrib domain** and carry essentially all attention + norm semantics.

There's a secondary lower-priority blocker that surfaces before the contrib-op issue is even reached: the strict shape unification trips at node #295 of the Phi-3 cuda-fp16 export with `Impossible to unify closed shapes of different rank (found  and 1)` — a Const-vs-Sub-consumer rank clash. Worth filing separately as part of the same upstream PR.
**Workaround.** None landed in tract-coreml. For the LLM probe path we pivoted to `onnx-community/SmolLM2-135M-Instruct-ONNX` (Optimum-exported standard ONNX, no contrib ops); that pivot worked and SmolLM2 now translates to 183 CoremlOps. But every onnxruntime-genai-exported LLM stays unreachable.
**Suggested upstream fix.** Two-layer fix: (a) make the op-builder lookup in `tract-onnx::model::Onnx::parse_node` domain-aware (`(domain, op_type) → builder`); (b) register at least pass-through "decompose to standard ONNX" handlers for the four hot contrib ops above (each one is implementable in terms of existing tract ops — `GroupQueryAttention` = RoPE + KV concat + masked SDPA, etc.). Or short-term: a CLI / API "decompose contrib ops" pre-pass that rewrites the ONNX before tract ingestion.
**Priority for upstream.** P1 — blocks the entire onnxruntime-genai-exported LLM family. Every modern Microsoft / partner LLM (Phi family, much of the deployment story for Apple/Qualcomm/Intel NPU partners) ships exclusively in this format.

---

## §2 IMPROVEMENTS — would significantly simplify backend authoring

### I1. Tract's CPU `Resize` is F32-only — ✅ **PATCHED LOCALLY 2026-05-12; ready as standalone upstream PR**

**Where we hit it.** Phase 3 / Milestone E (MODNet) and Phase 4 (RVM).
**Symptom.** When we `f32_to_f16` a model containing Resize, the resulting model has F16 inputs/outputs to Resize. Calling `runnable.run` on the CPU model errors: `Tensor datum type error: tensor is F16, accessed as F32`. We can't get a CPU baseline for any model with Resize when running through the F16 pipeline.
**Workaround (was).** Drop the CPU baseline for these models; report only the CoreML number. Honest about it in the perf table ("CPU baseline blocked by tract's F16 Resize").
**Local patch (now).** ~10 LoC in `tract/onnx-opl/src/resize.rs` `EvalOp::eval`: cast input tensor to F32 at entry (no-op when already F32), cast scales to F32 (`f32_to_f16` rewrites the scales constant too), do the existing F32 interpolation arithmetic, cast result back to the input's original dtype at exit. Plus 2 new unit tests (`linear_resize_2d_f16_upsample` and `nearest_resize_1d_f16_downsample`) covering the F16 round-trip. All 5 onnx-opl resize tests + 237 tract-core lib tests pass; 4 model-level perf tests confirm the unblock end-to-end (MODNet 241× / RVM 111× / InceptionV3 34× / Real-ESRGAN 12.7× tract-CPU vs tract-CoreML).
**Suggested upstream fix.** The local patch is the suggested upstream fix; cleanly separable into a standalone PR. Pattern of "cast to F32 internally" is shared with `random.rs` already in onnx-opl.
**Priority for upstream.** P1 — CoreML/Metal/CUDA backends all want to compare F16 inference to a CPU baseline. Without it we can't validate numerical correctness for segmentation / matting models.

### I2. `prost`-generated map fields encode in non-deterministic order

**Where we hit it.** Phase 3 / Milestone G (compile cache).
**Symptom.** Each MIL `Operation` has `inputs: HashMap<String, Argument>` and `attributes: HashMap<String, Value>` (protobuf `map<string, T>` fields). `prost::encode_to_vec()` serialises HashMap entries in iteration order, which Rust randomises per process. So identical Model protos produce different encoded byte sequences across runs — fatal for a content-addressed cache key.
**Workaround.** We hash the proto in canonical (sorted-by-key) order via a custom `hash_*_canonical` walker in `compile_cache.rs`. ~80 LoC of structural-hash code.
**Suggested upstream fix.** Not a tract issue per se, but tract pulls prost. The fix at the prost layer would be to either (a) ship a `prost::encode_canonical()` API that sorts map keys before encoding, or (b) document the non-determinism prominently. Filing upstream prost issue would benefit the whole ecosystem.
**Priority for upstream.** P2 (papercut) for tract; P1 (real footgun) for prost.

### I3. `prost-build` emits Rust doctests from .proto comments

**Where we hit it.** Phase 1 / proto vendoring.
**Symptom.** Coremltools' `.proto` files contain comments like `// Foo (e.g., x = MyOp(a, b))` which `prost-build` translates to Rust `///` doc comments. Rustdoc tries to compile snippets that LOOK like Rust as doctests, producing 113 spurious test failures on `cargo test --doc`.
**Workaround.** `prost_build::Config::new().disable_comments(["."])` in our build.rs.
**Suggested upstream fix.** `prost-build` should default to a more conservative comment translation that wraps everything in non-doctest fences, OR document this gotcha in the README.
**Priority for upstream.** P2 — every prost-build user with .proto comments hits this; tract doesn't, but it's part of the ecosystem we depend on.

### I4. tract's declutter aggressively decomposes high-level ops into fine-grained chains

**Where we hit it.** Phase 3 (InstanceNorm, GlobalAveragePool, Clip, HardSigmoid) and Phase 4 (AveragePool, Expand).
**Symptom.** ONNX ops like `InstanceNormalization` get decomposed into ~6 lower-level ops (ReduceSum + Mul(1/N) + Sub + Reduce<MeanOfSquares> + Add(eps) + Rsqrt + final Mul). Backends that have a direct equivalent (e.g., MIL `mb.instance_norm`) have to either (a) translate each piece individually and trust the framework's compiler to re-fuse (works, but loses semantic intent), or (b) write a multi-node pattern detector to recognise and consolidate the chain (we did this; ~290 LoC for InstanceNorm). Same pattern for AveragePool→SumPool and HardSigmoid→Clip→Min/Max chains.
**Workaround.** Multi-node pattern detector in `instance_norm.rs`. Pure decomposition for the others (relying on Apple's compiler to re-fuse). Empirically Apple's MIL compiler DOES re-fuse InstanceNorm optimally — our explicit `mb.instance_norm` was perf-neutral. So decomposition isn't always bad, but it forces every backend to either accept the cost or write its own re-fuser.
**Suggested upstream fix.** Two options: (a) an opt-out flag on declutter to NOT decompose specific op families (`declutter_options.preserve_instance_norm = true`), or (b) preserve high-level ops as the tract-typed-op layer and only decompose at codegen/optimize time. (b) is more invasive but correct architecturally.
**Priority for upstream.** P1 — affects every backend that wants to use vendor-fastpath ops (CoreML's `instance_norm`, `gelu`, `softmax`, `layer_norm`, etc.).

### I5. EinSum surfaces many axes signatures for what's semantically the same op

**Where we hit it.** Phase 2 / Phase 3 (EinSum-as-conv) and Phase 4 (EinSum-as-matmul, SE-block patterns).
**Symptom.** tract emits `IHW,OI->OHW`, `NIHW,OI->OHW`, `IHW,OI->NOHW`, `NIHW,OI->NOHW`, `NIHW,I->NOHW`, `mkba,kn->n`, `k,kn->mnab`, `IHW,OI->O`, `I,OI->OHW` — depending on the upstream op (Conv vs Linear), whether N got dropped during declutter, whether the output feeds a Concat (which forces N back), etc. Each surfaces as a distinct `EinSum.axes` value that the backend has to pattern-match. We've ended up with 5 conv-shaped patterns (in `einsum.rs`) and 4 matmul-shaped patterns (in `matmul.rs`); the next CNN we probe will probably surface more.
**Workaround.** Pattern-match each axes signature explicitly. Good test coverage prevents surprises but the maintenance burden grows linearly with the canary corpus.
**Suggested upstream fix.** An `EinSum::canonical_form()` method that normalises axes to a small set of well-known patterns (e.g., always uses `NIHW` for 4D image tensors, always emits the explicit `N` even if it's 1). Or a backend-facing query "is this einsum equivalent to a 1×1 conv?" that backends can call instead of reading axes.
**Priority for upstream.** P1 — every accelerator backend will hit this. tract-metal/cuda/coreml all need to recognise these patterns.

### I7. `RmsNorm` declutter has no opt-out and forces backends to recognise the lowered form

**Where we hit it.** Phase 4 / transformer-prep.
**Symptom.** ONNX `LayerNormalization` after `into_decluttered()` becomes a 6-op chain: `ReduceSum(x) + Mul(_, 1/N) + Sub(x, mean) + RmsNorm(d) + Mul(_, gamma) + Add(_, beta)`. The `RmsNorm` op exists in tract-core and absorbs the `(d² → reduce mean → +eps → rsqrt → ×d)` portion. This is a clever simplification — but `RmsNorm` itself is NOT a standard ONNX op, and accelerator backends that don't implement it specifically will see it as a CPU residual that fragments the LayerNorm chain into THREE subgraphs around the RmsNorm boundary. Until we wrote a standalone RmsNorm translator, our LayerNorm fold detector never got to run because the union-find pass split the chain.
**Workaround.** Implement a standalone RmsNorm translator (in `tract-coreml`: `~210 LoC` in `ops/rms_norm.rs`) that lowers to `square + reduce_mean + add(eps) + rsqrt + mul`. Then a separate multi-node pre-scan (`ops/layer_norm.rs`) detects the LayerNorm chain and folds it to MIL `layer_norm`.
**Suggested upstream fix.** Either (a) preserve `LayerNormalization` as a first-class typed op (architectural; same pattern as I4); (b) ship `RmsNorm` as a "preferred"-op opt-in via declutter options so backends with native LayerNorm can request the chain in original form; or (c) document the post-declutter LayerNorm chain shape prominently so backend authors know to look for it.
**Priority for upstream.** P1 — every transformer model has multiple LayerNorms; backends without RmsNorm support will leave them all as CPU residuals.

### I10. tract surfaces rank-6 einsums with unit "decorator" axes that overflow MIL's rank-5 limit

**Where we hit it.** Phase 4 / SAM 2 Hiera-Tiny / `dcbamk,kn->cabmn` QKV-projection einsum.
**Symptom.** tract's declutter inserts a unit-dim "decorator" axis (`d` in `dcbamk,kn->cabmn` with d=1) that pushes input A's rank from 5 to 6. MIL caps tensor rank at 5 across all ops (matmul, transpose, reshape). Even though after stripping the unit `d` axis the canonical-form rank is 5, the MLPackage external input is declared at the original rank 6 — and CoreML rejects the MLPackage with `Rank of the input 'in0' is 6, but it should be between 1 and 5`.
**Workaround landed.** RESOLVED locally on `general_matmul.rs` ↑ commit pending: added `a_external_shape` / `b_external_shape` fields to the matmul plan = `a_shape` / `b_shape` with strip-A/B unit-dim positions removed. The MLPackage declares the external (post-strip) shape; CoremlOp does the squeeze at the tract-side ↔ MLPackage-side boundary. perm_a / perm_b recomputed in external-position coordinates. Allows einsums up to rank 6+ as long as POST-STRIP rank ≤ 5. SAM 2 Hiera-Tiny: 8 of 9 stranded `dcbamk,kn->cabmn` einsums absorbed by this fix (the remaining 1 was a different `NHWI,OI->NOHbWa` neck conv).
**Upstream fix still desirable.** The local workaround handles the matmul case but every other translator (binop, activation, reduce, etc.) still gets rank-6 inputs from the rank-padding upstream. Cleaner upstream fix: a declutter pass that removes unit-dim axes from einsum signatures + the upstream rank-padding when they're load-bearing only for tract's tensor-fact bookkeeping.
**Priority for upstream.** P2 (lowered after our workaround) — only Conv-shape einsums with extra decorator axes (`NHWI,OI->NOHbWa`) still surface stranded.

### I12. tract emits "M-less degenerate" matmul einsums (rank-1 output, no M axis)

**Where we hit it.** Phase 4 / Parakeet TDT joint sub-model / `akm,nk->n` einsum.
**Symptom.** Parakeet's joint network surfaces einsum signatures like `akm,nk->n` where input A is `[a=1, k=1024, m=1]` and weight is `[n=640, k=1024]`. Semantically this is `weight @ input_flattened` (a matrix-vector product, gemv) — input has only "strip" axes (a, m) and a K, so when those degenerate to 1, the matmul is rank-2 weight × rank-1 vector → rank-1 output. There's no M axis at all. Our `general_matmul.rs` requires at least one M axis (since matmul is conceptually `[..., M, K] @ [..., K, N] → [..., M, N]`); a degenerate M=1 case has `M` *implied* but not present in the einsum signature.
**Workaround.** Skip these einsums (they fall back to CPU). On Parakeet joint: 3 stranded EinSums of this shape (1 per linear layer). Could be handled by adding an "M-less" code path to `general_matmul` that creates a synthetic M=1 dim, runs the matmul as `[1, K] @ [K, N] → [1, N]`, and squeezes the leading 1 before output. ~30 LoC. Not done — adds workaround complexity to the translator.
**Suggested upstream fix.** When tract lowers a Linear or matmul op to EinSum, ensure an explicit M axis is present even when M=1. Concretely: `[input, weight].matmul()` with `input.rank == 1` should lower to einsum `Mk,nk->Mn` (with M=1 as a broadcast dim) rather than `k,nk->n`. Or, equivalently, the conv-shaped lowering already has this pattern (`NIHW,OI->NOHW` keeps explicit N).
**Priority for upstream.** P2 — surfaces in any model with rank-1 vector × matrix products in the post-encoder layers. Common in classifier heads + ASR joint networks.

### I11. tract's view of MLPackage rank diverges from upstream-rank-padded actual rank

**Where we hit it.** Phase 4 / SAM 2 Hiera-Tiny / general_matmul attention chain.
**Symptom.** Our binop / reduce / activation / general_matmul translators all rank-pad outputs to 4 (so chains stay rank-4 internally inside an MLPackage). But tract's `output_fact` for these ops can be rank 3 (it computes broadcast/reduce semantics in pure tract terms, not knowing about MLPackage rank-padding). Downstream translators that read `outlet_fact(predecessor)` then disagree about rank with the actual MLPackage value flowing in. CoreML rejects with `Tensor descriptor rank doesn't match MPSGraphTensor` or `perm tensor length must equal input tensor rank`.
**Workaround.** Bridge logic in our matmul: explicitly emit a Reshape from `pad_to_ml_rank(tract_a_shape)` to `tract_a_shape` at the input boundary, so downstream MIL ops see consistent ranks. This adds a no-op-ish Reshape per matmul input (cheap; CoreML compiler may fold it). Each translator that reads `outlet_fact` rank needs awareness of the rank-padding convention.
**Suggested upstream fix.** Not really a tract issue — it's a quirk of OUR backend's rank-padding convention. But: a tract API that exposes "what the MLPackage backend actually sees" (or a documented convention for when to rank-pad vs not) would help every accelerator backend keep its translators in sync. Concretely: a per-op trait method `mlpackage_output_rank(tract_rank) -> usize` so backends can declare their padding policy in one place.
**Priority for upstream.** P3 — own-side documented; only relevant if tract grows more accelerator backends that need rank-pad conventions.

### I9. Hiera windowed attention surfaces ~17 distinct EinSum axes signatures

**Where we hit it.** Phase 4 / SAM 2 Hiera-Tiny image encoder.
**Symptom.** Hiera's hierarchical windowed attention emits einsums with 5–6 axes per operand, encoding (batch × window × head × seq-in-window × dim-per-head) plus various unit-dim "decorator" axes that tract inserts during declutter. After our extension (strip + expand + multi-K/M/N), 50/78 stranded EinSums got absorbed into one of these patterns, but 28 remain — all with batch axes at DIFFERING positions in input 0 vs input 1 (e.g. `abmk,akcbn->ambn` where `b` is at position 1 in A but position 3 in B). These are the Q@K and attn@V patterns; they need a Transpose pre-matmul to canonicalise batch order.
**Workaround.** None yet — these stranded EinSums act as CPU residuals and split the model into 110 CoremlOps. End-to-end runs but each subgraph boundary costs a Core ML prediction call (~40 ms cross-domain dispatch on M1/M2). SAM 2 first end-to-end perf with this state: 4.4 s/inference at 1024×1024 (vs the ~50–150 ms a fully-fused encoder should get). Adding batch-reorder support (a Transpose pre-matmul + post-matmul transpose pair when output batch order also differs) would absorb the remaining 28 EinSums.
**Suggested upstream fix.** Same flavour as I5 (EinSum signature explosion): an `EinSum::canonical_form()` method that returns a normalised representation with a small set of well-known patterns. Or a tract pass that inserts Transpose ops to bring all matmul-shaped einsums into a canonical `[..., M, K] @ [..., K, N] → [..., M, N]` layout before backends see them. Either approach removes the burden of pattern-matching dozens of axes signatures from every backend.
**Priority for upstream.** P1 — affects every transformer model with windowed/local attention (Swin, Hiera, BEiT, etc.) on every accelerator backend.

### I8. Union-find subgraph builder treats `Source` nodes as blocking CPU residuals

**Where we hit it.** Phase 4 / transformer-prep / LayerNorm chain.
**Symptom.** Not a tract bug — a tract-coreml subgraph-design oversight that *might* generalise to other backends. Our `identify_subgraphs` had an "all-or-nothing" rule: a node only joins a subgraph if ALL its non-const data predecessors are translatable. The `TypedSource` node (the model input boundary) isn't translatable, so any op that reads source directly (e.g. `Sub(x, mean)` in LayerNorm where `x` IS the source) was blocked from joining the subgraph that contained `mean`. Result: the LayerNorm chain split into pre-source-Sub and post-source-Sub halves at the union-find layer, before our LayerNorm fold detector even ran.
**Workaround.** In `identify_subgraphs`, exclude `TypedSource` predecessors from the all-or-nothing check (Source can't participate in a wiring cycle since it does no compute). One-line fix.
**Suggested upstream fix.** None for tract directly. But — a tract convention "Source nodes are always non-blocking from a backend-fusion perspective" or a helper API `model.is_pure_input(outlet) -> bool` would let other backends avoid this same oversight.
**Priority for upstream.** P3 (own-side documented in our crate; might inform tract-metal/cuda design).

### I13. Declutter rewrites `LayerNorm` chains into `RmsNorm` chains (semantically suspect)

**Where we hit it.** Phase 5 / ViT-base coverage probe.
**Symptom.** Loading `Xenova/vit-base-patch16-224` ONNX (which exports standard `LayerNormalization` ops) and running `into_decluttered()` produces a graph with **25× `RmsNorm`** and **0× LayerNorm**. Each transformer block has a pre-attn LayerNorm and a pre-FFN LayerNorm; after declutter they all surface as `RmsNorm`. But these aren't the same op: `LayerNorm(x) = (x - mean(x)) * 1/sqrt(var(x) + eps) * gamma + beta` (subtracts mean), whereas `RmsNorm(x) = x * 1/sqrt(mean(x²) + eps) * gamma` (does NOT subtract mean). For ViT to produce correct outputs after this rewrite, either (a) the chain still includes a separate Sub(x, mean) step that we're not seeing in the histogram, (b) tract's RmsNorm has a `subtract_mean` flag we don't know about, or (c) the rewrite is genuinely a correctness bug in declutter.
**Workaround.** None on our side. Our standalone RmsNorm translator runs against these ops; downstream output is whatever the (potentially-incorrect) declutter produced. ViT inference works end-to-end but we have NOT done a CPU-vs-CoreML correctness comparison (synthetic input is OOD for ViT, similar to MODNet — would need an ImageNet sample).
**Suggested upstream fix.** Either (a) document the declutter's intent (does it preserve mean-subtraction via a separate Sub op? does RmsNorm have a hidden subtract-mean field?), or (b) preserve `LayerNormalization` as a first-class typed op (architectural — same pattern as I4); backends with native LayerNorm support get the cleaner form, and the LayerNorm→RmsNorm rewrite stays opt-in for the cases where it's mathematically valid (e.g., post-residual where mean is approximately 0).
**Priority for upstream.** P0 if it's a correctness bug, P1 otherwise. Affects every transformer model in the corpus.

### I14. Tract's declutter decomposes `Resize(nearest, integer scales)` into a tile chain that loses upsample intent

**Where we hit it.** Phase 5 / Real-ESRGAN x4 coverage probe (and indirectly Phase 4 / YOLOv8n's FPN-neck upsamples — same decomposition surfaces in YOLOv8's PAN path).
**Symptom.** ONNX `Resize` with `mode="nearest"` and integer scales (typical for super-resolution / FPN upsamples) gets rewritten by `into_decluttered()` into a `Reshape → AddAxis → AddAxis → MultiBroadcastTo → Reshape` tile chain. Each step is rank-altering and produces rank-5 intermediate tensors. Backends with a direct `mb.upsample_nearest_neighbor` (CoreML/Metal/CUDA all have one) never see the high-level Resize and have to either (a) reconstruct the upsample by detecting the 5-op tile pattern, or (b) leave the chain as-is and hope each piece fuses well downstream. We did (b) for now; both Real-ESRGAN x4 and YOLOv8n's neck strand the AddAxis + MultiBroadcastTo ops.
**Workaround.** Pad each translator's input to defensive rank (Conv/Concat/Slice now defensively reshape to plan-expected rank — see commit). The rank-5 intermediates flow through but are never recognized as an upsample. Marginal cost on Apple's compiler (it likely re-folds them).
**Suggested upstream fix.** Same pattern as I4: preserve `Resize` as a typed op through declutter; only decompose at codegen time when the backend doesn't claim native support. Or expose the decomposition as an opt-out declutter flag (`preserve_resize_nearest = true`).
**Priority for upstream.** P2 — works around in backend, but every accelerator-targeting backend will have to detect the same pattern. SR / segmentation / object-detection FPNs all hit this.

### I15. `into_decluttered` peels batch dim from inner Convs (`NCHW → CHW`) producing rank-3 tensors that backends must rank-pad

**Where we hit it.** Phase 5 / YOLOv8n coverage probe.
**Symptom.** YOLOv8n exports as a fully-explicit-NCHW ONNX (input `[1, 3, 640, 640]`). After `into_decluttered()`, **2 of 39** Convs retain `data_format=NCHW` (rank 4) — the two adjacent to the model input boundary. The other **37 Convs** end up with `data_format=CHW` (rank 3): tract optimized away the unit batch dim from inner Convs as a perf win for its own CPU runtime. MIL `conv` requires rank-4 NCHW input, so without a rank-padding shim every CHW Conv aborts MLPackage compile with "Variadic dimension at [2,-1] of tensor parameter x[0] have unexpected length 1; expected 2." Same family as I6 (rank-3/rank-4 mismatch) but specifically the "tract chooses to peel" side rather than the "backend forgot to align" side.
**Workaround.** Conv (and Concat / Slice) translators now emit a defensive `mb.reshape(x, plan.input_shape)` before the actual op, materializing rank-4 input regardless of upstream rank. Metadata-only when shapes already match. ~10 LoC per translator. After this fix YOLOv8 went from "abort on Conv #5" to "7 CoremlOps, working first-run".
**Suggested upstream fix.** Either (a) stop peeling batch dims from inner Convs when the original ONNX was unambiguously NCHW, or (b) provide an `into_decluttered_options` flag to opt out of the batch-peel rewrite, or (c) document the peel and the convention it follows so backends can rank-pad consistently rather than per-op-by-op. (a) generalises to any backend that wants to keep ONNX-canonical 4D facts.
**Priority for upstream.** P2 (workable in backend, but multiplies per-op rank-padding code across every translator). Surfaces in every CNN with > 2 Convs.

### I6. Translators that consume rank-3 (CHW) tract facts get rank-4 MLPackage values

**Where we hit it.** Phase 4 / RVM / `reduce.rs` and `matmul.rs`.
**Symptom.** When upstream is a Conv with `data_format=CHW`, the conv translator prepends `N=1` in the MLPackage shape. So MLPackage value is rank 4 `[1, C, H, W]` even though tract's fact for the Conv output is rank 3 `[C, H, W]`. Downstream translators that read tract axes from the model fact then index incorrectly into the MLPackage tensor (e.g. Reduce axes `[1, 2]` for H,W in CHW becomes indices `[1, 2]` in the rank-4 NCHW MLPackage value, which collapses C instead of H,W). MIL compile fails with "incompatible shape" assertion.
**Workaround.** In each translator that uses tract axes, shift by `(4 - tract_rank)` to align with the rank-4-padded MLPackage value. Took us a while to track down because the symptom (`reshape` rejecting [1,1] → [1,120]) was several ops downstream of the actual mis-axes.
**Suggested upstream fix.** Not really a tract bug — this is OUR rank-padding convention. But: a tract convention "every translatable op declares whether its MLPackage value has a different rank than its tract fact" would let downstream consumers query and adapt. Or simpler: tract could normalise CHW to NCHW at f32_to_f16 / declutter time so backends never see CHW.
**Priority for upstream.** P2 from tract's perspective; P1 from a backend-author perspective. The CHW vs NCHW split is a real source of bugs.

### I16. Preserve fused contrib ops (`GroupQueryAttention`, `RotaryEmbedding`, etc.) as first-class typed ops

**Where we hit it.** Phase 5 / Phi-3-mini LLM probe (paired with B8).
**Symptom.** When (eventually) tract-onnx grows `com.microsoft` domain support per B8, the natural temptation is to decompose each fused op into standard-ONNX equivalents. **That would be a mistake** — same architectural argument as I4 (general decompose-aggression), I7 (LayerNorm → RmsNorm), and the [RNN preservation handout](./tract-rnn-preservation-handout.md). Modern LLM accelerator backends all have dedicated fast paths:
- **Apple MIL (this PR's target):** `mb.scaled_dot_product_attention`, `mb.rotary_position_embedding` (iOS 18+), routed to ANE.
- **NVIDIA cuDNN:** `cudnnMultiHeadAttnForward` (fused QKV + softmax + masking).
- **Intel oneDNN:** `dnnl::sdpa_forward` (Intel 4th-gen Xeon AMX path).
- **AMD MIOpen:** `miopenMultiHeadAttention`.

If tract-onnx ingests `com.microsoft.GroupQueryAttention` as a single fused tract typed op (e.g. `tract_core::ops::nn::GroupQueryAttention { num_heads, num_kv_heads, head_dim, ... }`), every backend can emit its fast-path equivalent with ~200 LoC of MIL emit. If tract decomposes each one into MatMul + Reshape + RoPE + Softmax + MatMul + Concat, every backend has to write a brittle pattern-detector to re-recognize the fused op. Same cost-multiplier argument: **one upstream fix benefits four backends.**

The contrib ops worth preserving (with their natural high-level tract counterparts):
- `com.microsoft.GroupQueryAttention` → `Sdpa { num_heads, num_kv_heads, head_dim, ... }`
- `com.microsoft.MultiHeadAttention` → `Sdpa { num_heads, num_kv_heads = num_heads, ... }`
- `com.microsoft.RotaryEmbedding` → `RotaryEmbedding { dim, base, ... }`
- `com.microsoft.SkipSimplifiedLayerNormalization` → fused `Add + RmsNorm` (split into the two existing ops is fine if Add+RmsNorm pattern is preserved at codegen)
- `com.microsoft.SimplifiedLayerNormalization` → existing `RmsNorm` (this one's already covered if domain dispatch works)
- `com.microsoft.MatMulNBits` → tract's quantized MatMul (when quantization is in scope)
**Workaround.** None on our side until B8 is resolved. SmolLM2's Optimum export uses raw decomposed attention, so we're getting LLM coverage via the long path — and our `general_matmul` translator handles all 272 of SmolLM2's attention/FFN MatMuls. But MIL has `scaled_dot_product_attention` which goes directly to ANE; we're not engaging it because the einsum-form attention doesn't pattern-match it.
**Suggested upstream fix.** When B8 lands, design the contrib-op translators to PRESERVE the high-level shape rather than decompose. Same as I4 / I7 / RNN-preservation — opt-out decomposition transform (`lower_contrib_ops_to_standard`), default-on for the CPU pipeline, default-off for accelerator backends.
**Priority for upstream.** P1 — gated on B8 landing first; cost of getting this wrong (decomposing) is having to revisit later.

---

## §3 ANNOYANCES — papercuts we worked around with little cost

### A1. `tract-linalg`'s build.rs has a clippy lint that we can't fix from tract-coreml

**Where we hit it.** Phase 3 (clippy hygiene).
**Symptom.** `cargo clippy -p tract-coreml --tests -- -D warnings` fails because `tract-linalg/build.rs` has `clippy::needless_borrows_for_generic_args` warnings that we can't address from a downstream crate.
**Workaround.** Run clippy with `--no-deps` to scope to our crate only.
**Suggested upstream fix.** Run `cargo clippy --workspace --all-targets -- -D warnings` in tract's CI and fix the build-script lints upstream.
**Priority for upstream.** P2.

### A2. `Cast(Const)` chains aren't folded by `into_decluttered` after `f32_to_f16`

**Where we hit it.** Phase 1 / Conv weights.
**Symptom.** When `f32_to_f16` runs after a model has F32 const weights, it wraps each Const in a `Cast(F32→F16, Const)` rather than evaluating the cast eagerly and replacing with a single F16 Const. Backends then have to walk these chains to recover the actual weight value.
**Workaround.** `const_tensor(model, outlet)` helper in `ops/mod.rs` that recursively walks `Cast(Const)` chains and evaluates the cast. ~12 LoC.
**Suggested upstream fix.** A `cast_const` declutter pass after `f32_to_f16` that materialises the cast values, eliminating the Cast nodes.
**Priority for upstream.** P2 — minor inefficiency, easy backend workaround.

### A3. Tract decomposes `AveragePool` into `SumPool(normalize=true)` + nothing else

**Where we hit it.** Phase 4 / RVM.
**Symptom.** Backends that have a direct AvgPool op never see it — it surfaces as SumPool with a `normalize` flag, which is semantically identical to AvgPool. Then we have to write a SumPool translator that detects `normalize=true` and maps to `mb.avg_pool`.
**Workaround.** `avgpool.rs` translator handles both cases (normalize=true → direct mb.avg_pool; normalize=false → mb.avg_pool * area).
**Suggested upstream fix.** Preserve `AveragePool` as a first-class op (or rename the lowered form to `AveragePool` since semantically that's what it is).
**Priority for upstream.** P2.

### A4. `MultiBroadcastTo` is the only path for ONNX `Expand`

**Where we hit it.** Phase 4 / RVM.
**Symptom.** Same pattern as A3 — ONNX `Expand` decomposes into tract's `MultiBroadcastTo`. Backends have to recognise the latter to translate to MIL `tile`. Not a problem, but the rename obscures intent.
**Workaround.** `broadcast.rs` translator handles MultiBroadcastTo.
**Suggested upstream fix.** Either rename to `Expand` or document the equivalence in a comment near the type.
**Priority for upstream.** P3 (documentation).

### A5. Default `cargo test` parallelism races on tract-coreml's compile cache directory

**Where we hit it.** Phase 3 / Milestone G + later.
**Symptom.** When several tests share Conv content, they hit the same cache key in `~/Library/Caches/tract-coreml/v1/`. cache_smoke wipes its entry; another test reading from the same path mid-load gets file-system errors. Intermittently fails ~25% of `cargo test` runs.
**Workaround.** Use unique weight seeds in cache_smoke so its key is one-of-a-kind. Most other tests still fine. For 100% reliability would need `--test-threads=1`.
**Suggested upstream fix.** Not a tract issue — this is our cache design choice. Worth noting in our own docs.
**Priority for upstream.** N/A (own-side fix).

### A6. ONNX op `HardSigmoid` decomposes into Clip + Min/Max + const muls/adds

**Where we hit it.** Phase 4 / RVM.
**Symptom.** RVM has 28 HardSigmoid ops in raw ONNX. After `into_decluttered + f32_to_f16`, they're gone — replaced by chains of BinOp Min/Max with const operands. Since we already cover Min/Max + const absorption, this works, but there's no opportunity to use MIL's direct `mb.hard_sigmoid` op.
**Workaround.** Just let it ride — Apple's compiler likely re-fuses to a hardware fast path. Same lesson as InstanceNorm-fold (perf-neutral).
**Suggested upstream fix.** Same as I4 — opt-out from specific declutter passes.
**Priority for upstream.** P2.

---

## §4 OPEN QUESTIONS FOR THE MAINTAINER

### Q1. Why does tract decompose high-level ops aggressively at declutter time?

We assume the rationale is "fewer op types = simpler optimizer / planner". But for accelerator backends, the high-level ops carry intent that the lower-level chain loses. Curious whether the maintainer has considered a "decompose at codegen, preserve at typed" architecture, and what would push back on that.

### Q2. Should `tract-coreml` live in the same workspace as `tract-metal` / `tract-cuda`, or be a separate crate?

Currently designed as a sibling workspace member. Adds two workspace deps (`tract-onnx-opl`, `sha2`). Worth confirming alignment with Kali on naming, structure, and feature-flag conventions before the PR.

### Q3. CPU regression policy — `cargo test -p tract-core --lib` is what we run. Is there a broader CI gate the PR will be measured against?

Want to make sure we're not surprised by a CI step we haven't validated locally.

### Q4. Co-author credit on commits

The kickoff specifies `Co-Authored-By: Claude Opus 4.7 (1M context)`. Worth confirming this is acceptable to the maintainer before opening the PR. ORT, candle, and other Rust ML projects accept similar patterns; tract's commit history doesn't show much precedent either way.

---

## Index of resolved items

- **I10** (rank-6 einsum overflow) — RESOLVED locally in `general_matmul.rs` v4+v5 via two-sided external-shape boundary:
  - **v4 (input boundary)**: `a_external_shape` / `b_external_shape` = inputs with strip-A/B unit-dim positions removed. CoremlOp squeezes at tract↔MLPackage boundary. SAM 2 Hiera-Tiny: 8 of 9 stranded `dcbamk,kn->cabmn` einsums absorbed.
  - **v5 (output boundary)**: `output_external_shape` = output with expand-out unit-dim positions removed. CoremlOp re-inserts unit dims at consumer side. SAM 2: last `NHWI,OI->NOHbWa` neck conv einsum absorbed. **All 81 SAM 2 EinSums now translated** (down from 78 stranded initially).
  - Upstream tract fix still desirable for cleaner architecture (declutter pass to remove load-bearing-only unit dims), but no longer blocking — every accelerator backend can adopt the same boundary-shape trick.
