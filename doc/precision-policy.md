# RFC: numerical precision policy

**Status: proposal, feedback wanted.** Once agreed, section 2 replaces the
"Numerical precision" section of [`AGENTS.md`](../AGENTS.md). The rest of this
document is the rationale, the inventory of where tract deviates today, and the
migration.

## 1. Why

The current rules are short: no lower precision than declared unless a knob or an
explicit transform asks for it; higher precision is always fine; the platform's
native behaviour for the declared type is the baseline; special values stay
correct. They break down in four places.

- **"Declared" is ambiguous for linear ops.** A matmul with f16 activations and
  Q4_0 weights has an activation type, a weight format, an output type, and
  sometimes an `operating_dt`. A kernel also makes choices the model never
  states: the type weights are unpacked to, whether activations are re-quantised,
  the accumulator, and the epilogue. Whether a Metal kernel that unpacks Q4_0 into
  f16 tiles is "lowering" depends on which declared type you compare against.
- **Opt-ins are process-global.** `TRACT_CUDA_ALLOW_TF32` and `TRACT_AMX_BF16` apply
  to every model in the process and are read at eval or kernel-dispatch time. Two
  models served from one process cannot differ, and the knowledge that "this model
  tolerates TF32" lives in someone's environment rather than with the model.
- **Several kernels already deviate silently.** See the inventory in section 5.
- **The special-values rule is stricter than the code.** Every Metal shader is
  compiled with fast math (`CompileOptions::new()`), while Metal kernels rely on
  `-INFINITY` as a mask and reduction starting value.

## 2. Proposed policy (AGENTS.md wording)

> ## Numerical precision
> Applies to every backend and kernel.
>
> **Declared compute type.** Every op has one:
> - its `operating_dt` (EinSum and every matmul-family op, MoE and recurrent ops
>   included);
> - its `acc_datum_type` (Sdpa);
> - at least f32 for reductions (sums, means, norm statistics, softmax
>   denominators);
> - its input type otherwise.
>
> It is set on the declared model and carried through every rewrite and lowering.
>
> **Fair game:**
> - Reassociation, FMA, fused epilogues. No bit-level matching with other frameworks
>   or with IEEE evaluation order.
> - Computing wider than the declared compute type, the declared output type
>   unchanged.
> - f16 accumulation in matmul-family kernels whose declared compute type is f16.
> - Rounding dequantised weights to the declared compute type.
> - Approximations of transcendental functions and division whose error is bounded
>   and well below the resolution of the declared type.
> - Assuming values are normal: finite, with subnormals allowed to flush to zero.
>   What a kernel produces after an emergent inf or NaN (overflow, 0/0) is
>   unspecified.
>
> **Must hold:**
> - Every hidden value of an op stays at or above its declared compute type: weights
>   and activations as fed to the arithmetic, accumulators, epilogues. Only a
>   tolerance (below) lifts this.
> - `-inf` used as an additive mask or a reduction starting value keeps its meaning:
>   masked positions contribute nothing. A finite sentinel is fine; adding masks
>   together must not produce NaN.
> - `IsNaN` / `IsInf` and their inputs keep IEEE semantics. The plain CPU reference
>   evaluation keeps IEEE semantics.
> - Integer arithmetic, quantized included, is exact.
> - tract never changes the floating-point environment (FTZ/DAZ, rounding mode) of
>   threads it does not own.
>
> **Tolerances.** Going below the declared compute type requires a tolerance,
> `tract.tolerate.<family>.<role> = <formats>`. It is set as a model property, per
> model in a deploy config, or with the knob `TRACT_TOLERATE_<FAMILY>_<ROLE>`. Every
> kernel declares its signature in this vocabulary. A kernel that goes below the
> declared compute type is selected only when tolerated; otherwise the op is
> unsupported on that backend and placed like any other unsupported op. Explicit
> transforms (`f32_to_f16`) change the declared types and are not tolerances.
>
> **Consistency.** The floor above is promised on every backend; identical results
> across backends or platforms are not. Results are reproducible run to run for the
> same build, hardware, configuration and batch composition. They are not promised
> to be independent of batch composition (autobatching may pick other kernels). The
> CLI and every API binding share the same defaults.

Transitional clause, removed in 0.24:

> Until 0.24, kernels marked `legacy("0.23")` may deviate under the `legacy-0.23`
> profile (`TRACT_TOLERATE_PROFILE`). No new marks.

## 3. Concepts

### 3.1 Declared compute type

This is mostly what tract already has, made consistent:

| op | declared compute type | today |
|---|---|---|
| EinSum | `operating_dt` | mandatory field |
| Sdpa | `acc_datum_type` | field |
| reductions, norm statistics, softmax denominators | at least f32 for f16/bf16 inputs | new rule (3.4) |
| element-wise | input type | implicit |
| MoeFfn, RoutedLinear, RoutedMatMul, GatedDeltaNetRecurrent, Mamba2Recurrent | `operating_dt` | **missing** |
| PrefixMatMul | `operating_dt` | `Option`, `None` after the strict (NNEF) lowering |
| GPU GEMM ops | carried from the source op | **dropped** by the translators |

Without a single reference type per op, verdicts on mixed ops are arbitrary:

- f32 activations × Q4_0 weights, kernel unpacks to f16 tiles: compute type f32,
  so this is a deviation.
- f16 activations × Q4_0 weights, same kernel: compute type f16, so it is compliant.
  Q4_0 values need up to 14 significant bits, so the tiles round; rounding to the
  declared compute type is fair game.
- An f16 MoE model where tract's own decomposition casts x to f32: the compute type
  is inherited from MoeFfn (f16), so f16 tiles stay compliant. Today the cast
  silently turns them into a deviation.

### 3.2 Hidden values and the role vocabulary

A *hidden value* is one that is not a tensor in the declared model: kernel
registers and tiles, plus casts that a backend translator inserts. Graph edges are
typed by the IR, and transforms own them. Tolerances only name hidden values.

| family | roles |
|---|---|
| `mm`: matmul, einsum, conv, deconv, routed linear | `w`, `act` (operands as fed to the multiply), `acc`, `epi` (fused epilogue before the final rounding) |
| `attn`: fused SDPA | `qk.{w,act,acc}`, `pv.{w,act,acc}`, `p` (probabilities fed to P·V), `sum` (softmax denominator) |
| `red`: sums, means, norm statistics, unfused softmax | `acc` |
| `approx`: not a type | `exp`, `tanh`, `div`, …, values `precise` / `fast`, for approximations beyond the bar in 3.5 |

Formats: `f32 tf32 f16 bf16 fp8 mxfp8 q8_1 q4_0 … i32`, using ggml names for
block formats and OCP names for MX formats. A scale scheme is part of the format
name.

- **Values are sets, not thresholds,** because f16 and bf16 are not ordered (range
  vs precision). The source format is not part of the key: it is the op's declared
  compute type. So `tract.tolerate.mm.acc = f16`, not `mm_acc_f32_to_f16`.
- **Coupled modes need every role they touch.** TF32 needs both `mm.w` and `mm.act`
  to tolerate `tf32`.
- **Kernel signatures.** Each kernel declares its tuple, e.g. Metal ggml f32 GEMM
  `{w: f16, act: f32, acc: f32}`, CUDA GGML dense `{w: q4_0, act: q8_1, acc: i32}`,
  cuDNN TF32 `{w: tf32, act: tf32, acc: f32}`. With signatures, the policy becomes a
  test: with no tolerance, every reachable kernel's signature is ≥ the declared
  compute type. They also give `dump --audit-json` a per-node `relaxed` field.

### 3.3 Tolerances

A tolerance is a model-scoped *permission* to compute below the declared compute
type, optionally restricted to some nodes.

- **It is a permission, not a command.** A backend with no faster kernel ignores it.
  Where it is granted, the backend should take the fastest permitted path.
- **It is read at transform, codegen or prepare time, never at eval.** Optimised
  models bake the decision in.
- **One declaration gives two names:** a scopable knob (`TRACT_TOLERATE_MM_ACC`) and
  a model property (`tract.tolerate.mm.acc`). The property travels in NNEF.
- **Resolution, from highest priority:**
  1. programmatic override;
  2. the knob scope (a deploy section's `[model.knobs]`, CLI `--tolerate`);
  3. environment;
  4. the model property;
  5. deploy-config global;
  6. off.

  So a deployer can grant or revoke per model, and a global `=none` is the
  bisection switch for accuracy bugs.
- **Node scope:** a compact string, e.g. `"all"` or `"all,-lm_head,-attn.qk"`. It is
  parsed into the existing `NodeFilter`.
- **Ways to set one:** an exporter writes the property; the `tolerate` transform
  (sugar over `set_property` that validates); deploy `[model.knobs]`; CLI
  `--tolerate`.

### 3.4 Reductions accumulate in at least f32

Matmul-style "f16 accumulation is native for f16" does not carry over to reductions:

- **Range.** An f16 sum of squares overflows when rms > √(65504/n): 4.0 at n=4096,
  2.8 at n=8192. LLM residual streams routinely exceed that.
- **Precision.** Simulated f16 sums: a 32-lane f16 accumulator (the arm64fp16 sum
  kernel's scheme) gives 5e-4 relative error on |N(0,1)| data with n=4096, and about
  1e-2 on a softmax-like denominator with n=32768. A sequential f16 sum gives
  8.5e-2 and 7.5e-1 on the same data, and returns 2048 instead of 4096 when summing
  4096 ones.
- **Cost.** Reductions are bandwidth-bound, so f32 accumulation is essentially free.
  Matmul f16 accumulation, by contrast, pays (1.5–2×).
- **Prior art.** torch accumulates half reductions in float (`acc_type`). TensorRT
  normalisation accumulates in f32 regardless of builder flags. ONNX LogCumSumExp
  requires at least f32.

Compound ops split into roles. Statistics are `red.acc` (floor f32). Per-element
steps (exp, normalise, affine) are element-wise (floor = input type). So the
following are compliant:

- softmax storing exp as half before rescaling;
- RmsNorm rounding the normalised value to the input type before gamma;
- fused residual x+r computed in the input type.

### 3.5 Special values and approximations

In a neural network, an emergent inf or NaN means the model is already broken.
"Assume normal" makes overflow behaviour unspecified and legalises
finite-math compilation, flush-to-zero, saturating casts and fp8 e4m3 (which has no
inf). Three carve-outs keep what models actually rely on:

1. **`-inf` masks and reduction starting values.** Metal kernels inject
   `-INFINITY` in 7 shader files while being compiled assume-finite. That is a bug
   under the new rule; the fix is finite sentinels.
2. **`IsNaN` / `IsInf`.**
3. **The reference evaluation stays IEEE.** f16 overflow, the usual failure of
   `f32_to_f16`, then stays visible in `compare`.

Known casualty: a model that produces `-inf` on purpose and clamps it later
(`log(x)` then `clamp`). A declutter rewrite can absorb that pattern if one shows up.

Approximations: tract's CPU tanh, sigmoid, exp and erf are already approximations
with a few ulps of error. Metal fast math is documented and bounded. The bar
("bounded and well below the resolution of the declared type") must be checked for
each function the Metal kernels call. Watch `pow` (its error scales with the
exponent), `log` near 1, and `sin`/`cos` (absolute bounds). A failing call site
switches to `precise::`.

## 4. Mechanism to build

| id | what |
|---|---|
| I1 | Tolerance infrastructure: `declare_tolerance!` (scopable knob + property), resolution (3.3), node scope, `tolerate` transform, CLI `--tolerate`, `--list-tolerances`. |
| I2 | Kernel signatures. CPU MMM kernels already expose an internal type and packings; GPU kernels need annotating. A CI test checks signatures against the declared compute type. The audit gets a `relaxed` field. |
| I3 | The `legacy-0.23` profile (section 6): `legacy("0.23")` markers, a profile knob/property, and a checked-in list of marked kernels that may only shrink. |
| I4 | Compute-type plumbing: `operating_dt` on the MoE and recurrent ops; `PrefixMatMul.operating_dt` made mandatory; GPU translators carry it. |
| I5 | Test tolerances tied to the declared compute type (and to active tolerances) in `Approximation`, ULP-based where possible. |

## 5. Inventory of current deviations

Classes:
- **BUG**: violates even the current rules; fixed directly.
- **FIX**: free or near-free; restores consistency directly.
- **LEGACY**: has a cost; frozen under `legacy-0.23` until a strict path exists.
- **PORT**: an existing global knob becomes a tolerance.

| id | deviation | where | class | way out |
|---|---|---|---|---|
| B1 | MeanOfSquares squares and sums in half (overflows) | CUDA `nn.cu`, `MeanOfSquaresOp` (MeanOp next to it uses float) | BUG | `acc_t = float` |
| B2 | f16 sums on non-last axes / short rows are sequential half | CPU `core/src/ops/nn/reduce.rs` | BUG | f32 accumulation |
| B3 | NNEF export fails for a matmul-shaped EinSum with acc ≠ input type | strict lowering in `prefix_matmul.rs` | BUG | fall back to `tract_core_einsum` |
| B4 | `-INFINITY` masks and starting values under fast-math compilation | Metal, 7 `.metal` files | BUG | finite sentinels, mask addition safe |
| F1 | Sum / Prod accumulate in half | CUDA `nn.cu` `AddOp` / `MulOp` | FIX | f32 acc |
| F2 | Sum / Prod accumulate in half | Metal `nn_ops.metal` | FIX | f32 acc |
| F3 | non-contiguous softmax sums in half | CPU `softmax` generic path | FIX | f32 sum |
| F4 | `arm64fp16_sum_f16_32n` accumulates in half | CPU arm64fp16 | FIX (measure) | widen to f32; LEGACY if not free on ARM |
| F5 | f16 matvec accumulates `half2` partials, ignoring an f32 `operating_dt` | CUDA `mm_mv.cu` | FIX (measure) | f32 partials |
| F6 | MoeFfn casts x to f32, raising the compute type of routed kernels | `MoeFfn::routed_linear_patch` | FIX | needs I4 |
| L1 | dense ggml f32 / Q4_0 GEMM uses f16 tiles at f32 compute | Metal `ggml_mm_mv.metal` | LEGACY | f32-tile instantiation |
| L2 | routed prefill puts Q4_0 in f16 tiles at f32 compute | Metal routed grouped GEMM | LEGACY | f32 tiles; re-enable the cases skipped in the routed suite |
| L3 | f32 weights cast to f16 when activations are f16 | CUDA `convert_matmul_to_cuda` | LEGACY | follow `operating_dt`: cast activations up when it is f32 |
| L4 | GGML dense quantises activations to Q8_1 | CUDA dense GGML path | LEGACY | dequant + f16/f32 GEMM/GEMV; q4 models carry `mm.act=q8_1` (section 9, question 2) |
| L5 | flash attention casts Q, K, V and the mask to f16, f32 models included | CUDA `convert_sdpa_to_cuda_flash_attn` | LEGACY | f32 flash path or decomposed SDPA; f32 models opt in with `attn.*.act=f16` |
| P1 | `TRACT_CUDA_ALLOW_TF32` (global, read at eval) | CUDA cuDNN conv | PORT | `mm.{w,act}=tf32`, decided at lowering as a per-op math type (cuDNN descriptor, cuBLASLt compute type, not per-handle math mode) |
| P2 | `TRACT_AMX_BF16` (global, read in linalg) | x86 AMX | PORT | `mm.{w,act}=bf16`, passed through the kernel-selection query |

L5 measured on VoxCPM2 (f32 export, relative max error against PyTorch, flash attention cast to
f16 / decomposed SDPA in f32): LM `dit_mu` 2.4e-5 / 1e-7 with the stop logits unchanged, local
encoder 6.9e-5 / 1.5e-7, DiT 3.6e-4 / 3.9e-7, 10-step CFM 1.8e-3 / 4.6e-6. A degradation, not a
failure. The decomposed path costs 2–3 % on these short sequences; f32 LLMs at long context are
unmeasured.

Compliant under the proposal, no change:
- Metal fast math (after B4 and the approximation check);
- softmax exp stored as half before rescaling;
- RmsNorm r16 rounding before gamma;
- fused-residual x+r in the input type;
- arm64fp16 f16 matmul accumulation;
- FlashSdpa computing in f32;
- `f32_to_f16` narrowing accumulators (explicit transform).

## 6. Migration

Patch releases ship every 2–4 weeks and minors about 9 months apart, so the
migration runs over 0.23.x.

**`legacy-0.23` freezes; it does not grant.** The profile does not expand into broad
tolerances such as `mm.w=f16`, which would let new kernels take the same shortcuts.
It admits exactly the marked kernels, and the CI-checked list may only shrink.

- **P0, 0.23.11: no behaviour change except bug fixes.**
  - AGENTS.md rewording, with the transitional clause.
  - I1, then I2 + I3. The profile defaults to `legacy-0.23`, so `--audit-json` is
    unchanged and benches are neutral.
  - B1–B4, one PR each.
  - The Metal approximation check.
- **P1, 0.23.12–13: consistency.**
  - F1–F6 and P1–P2, one PR each, with before/after numbers.
  - I4 lands before F6.
  - Strict paths for L1–L5; each drops its marker.
  - Benches run strict alongside legacy (per-bench `knobs`).
- **P2, around 0.23.14, announced one release ahead.**
  - The default becomes `strict`; `legacy-0.23` is one knob away and logs a one-time
    warning.
  - `--print-profile legacy-0.23` emits the equivalent explicit tolerances.
- **P3, 0.24.0.**
  - Profile, markers, alias knobs and the transitional clause removed.

## 7. What tolerances enable afterwards

cuBLASLt TN GEMM on an RTX 5090, random operands, TFLOPS:

| operands → compute | 8192³ | 32×8192×8192 |
|---|---|---|
| f32, strict | 56 | 21 |
| f32, TF32 | 108 | 25 |
| f32, truncated to bf16 inside the kernel | 111 | 25 |
| f16 or bf16, f32 accumulator | ~215 | 46 |
| f16, f16 accumulator | 314 | 48 |
| fp8 e4m3 (per-tensor scale), f32 accumulator | 518 | 120 |

- **Change the kernel's compute mode** (TF32, f16 accumulator): fits a tolerance
  exactly. TF32 gives 1.9× when compute-bound. f16 accumulation gives 1.47× on
  GeForce parts (half-rate f32 accumulation); expect ~1× on data-centre parts.
- **Truncating to bf16 inside the kernel** buys nothing over TF32 on CUDA.
- **Cast to 16 bits around the matmul** (the output type stays f32): 3.8× for f32
  models when compute-bound, 2.2× on decode. Works with f16 today; bf16 needs
  `DatumType::BF16`, which tract does not have.
- **fp8 / Q8 activations** need a scale scheme. Dynamic scaling (per token, MXFP8
  blocks) fits a tolerance. Calibrated scales belong to the exporter.
- **Storing the whole model in bf16** changes types between ops, so it is a
  transform, not a tolerance.
- **Apple GPUs before M5** have no TF32, fp8 or bf16 matrix hardware, and f16 runs at
  the f32 rate. There, tolerances buy bandwidth and register pressure only.

## 8. Prior art

| runtime | how precision is stated |
|---|---|
| ggml | `ggml_prec_set_acc`: "the minimum required accumulator type". `ggml_prec_set_src(op, prec, idx)`: per-source floor, e.g. Q8 allows quantising activations to Q8_0. Floors per role, stored in op params. |
| StableHLO | `DotAlgorithm`: lhs/rhs precision types "independent from the storage types", `accumulation_type`, component counts for emulation (bf16_6x), `allow_imprecise_accumulation`. An unsupported algorithm raises an error rather than falling back. |
| ONNX Runtime | MatMulNBits `accuracy_level`: "the minimum accuracy level of input A". It defaults to 4 (int8 activations) when fusing DQ + MatMul. CUDA TF32 is on by default. bf16 GEMM fast math is opt-in per session. |
| NNAPI | fp32 "must be calculated using at least the range and precision of" IEEE fp32. Relaxation to fp16 is set per model. Conformance tolerance is 5 ULP of the declared type. |
| XNNPACK / LiteRT | fp16 only when the model's `reduced_precision_support` metadata names a supported accumulation type. Forced fp16 fails rather than falls back. A consistent-arithmetic flag exists. |
| TensorRT | Strongly typed networks: the graph's types are the contract. Normalisation accumulates in f32 regardless of flags. Determinism is tied to batch slot and neighbouring inputs. |
| PyTorch / JAX / MLX / OpenVINO / Core ML | All lower f32 by default somewhere: cuDNN conv TF32; JAX GPU `DEFAULT` = TF32; MLX f32 matmul; OpenVINO bf16/f16 in PERFORMANCE mode; Core ML fp16. Each has a global or per-op opt-out. |

Floors per role, permissions carried by the model, and f32 reductions match
existing practice. The end state is stricter than most runtimes' defaults.

## 9. Open questions

1. **Flip the default in a patch release** (around 0.23.14, proposed) or wait for
   0.24?
2. **Q8 activations for 4-bit weights (L4).** ORT and ggml do this by default.
   Proposal: keep the policy strict and have q4 exporters (torch-to-nnef) write
   `tract.tolerate.mm.act=q8_1` into the model.
3. **f32 attention on CUDA (L5).** Is an f32 flash path worth it, or should f32
   models fall back to the decomposed path unless they tolerate f16 attention?
4. **F4.** If widening the arm64fp16 sum to f32 is not free, does it become a legacy
   entry?
5. **Names:** `tolerate`, `legacy-0.23`, `TRACT_TOLERATE_PROFILE`, the family/role
   keys.
6. **Determinism.** Does run-to-run reproducibility hold on every GPU backend today
   (split-K, atomics)? Is a strict-determinism knob wanted?
7. **Vocabulary for later:** imprecise partial accumulation (`mm.acc=f32_fast`, fp8
   `FAST_ACCUM`) and emulated formats (bf16x3, bf16x9, tf32x3) as formats with their
   effective precision.
