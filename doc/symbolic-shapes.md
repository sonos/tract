# Symbolic shapes and `TDim`

A `TypedFact`'s shape is a `Vec<TDim>`, not a `Vec<usize>`: dimensions
that depend on a runtime input (batch size, sequence length, image
side) live in the graph as symbolic expressions until something binds
them to concrete values. This page is about that machinery.

## What `TDim` is

`TDim` (in `tract_data::dim`) is the algebraic data type tract uses
for any dimension that might not be known at graph-build time. It is
an enum of:

- `Val(i64)` — a known integer.
- `Sym(Symbol)` — a named symbolic atom (`B`, `S`, `T`, ...).
- `Add(Vec<TDim>)` / `Mul(Vec<TDim>)` / `MulInt(i64, Box<TDim>)` /
  `Div(Box<TDim>, u64)` — arithmetic.
- `Min(Vec<TDim>)` / `Max(Vec<TDim>)` / `Broadcast(Vec<TDim>)` —
  shape-aware reductions (see below).
- `Ge(...)` / `Eq(...)` — comparison terms that evaluate to `0` or `1`,
  used as boolean indicator terms inside larger expressions.

So `(S+1)/2`, `B*C`, `max(L, 1)` all sit in the graph as `TDim` trees,
and `output_facts` on every op produces output shapes in this algebra.
The optimiser does proper algebra on it (proving `S >= 0`, recognising
that `4*(S/4) + (S%4) == S`, etc.) so rewrites can fire even when the
exact values aren't known.

### Variants worth a closer look

A few of the variants aren't what they look like, and getting their
semantics wrong leads to either over-fitted optimisations or shape
inference failures that look mysterious.

- **`Broadcast(Vec<TDim>)`** is the dimension-wise broadcast rule from
  NumPy / ONNX, lifted to symbolic shapes. `Broadcast(1, X) = X`,
  `Broadcast(X, X) = X`, and `Broadcast(X, Y)` with both proven `> 1`
  must have `X == Y` or the model is inconsistent. The simplifier
  reduces what it can; what it can't goes into the optimised graph
  as a literal `Broadcast` term. This is **not** the same as
  `Max(X, Y)` — broadcast asserts compatibility, `Max` does not.
  Op shape inference for elementwise binops produces `Broadcast`
  per axis, never `Max`. In the TDim textual syntax (the format
  `parse_tdim` and CLI `--set` accept) the binary operator is `#`, so
  `S#1` parses to `Broadcast([S, 1])` (which simplifies to `S`). It
  pretty-prints back as `broadcast((S), (1))`.

- **`MulInt(i64, Box<TDim>)`** is the canonical form for "integer
  scalar times a symbolic dimension". It is structurally distinct
  from `Mul(Vec<TDim>)` even though `MulInt(2, x)` and
  `Mul([Val(2), x])` denote the same value; the simplifier
  canonicalises into `MulInt` whenever one factor is a known
  integer, which makes pattern matching cheap.

- **`Div(Box<TDim>, u64)`** is integer division by a known positive
  divisor. The denominator must be a `u64` constant so the rest of
  the simplifier can reason about it ("is this multiple of 4
  recoverable from a `(_+3)/4` ceiling?"). There is no symbolic /
  symbolic division.

- **`Min` / `Max`** are honest min/max, used for clipped indexing and
  for shapes that get clamped (`Slice` with bounds that may exceed
  the dim, `MaxPool` with `ceil_mode = false`, etc.). Unlike
  `Broadcast`, they don't assert anything about the inputs.

- **`Ge(a, b)` / `Eq(a, b)`** evaluate to `Val(1)` if the relation
  holds, `Val(0)` otherwise. They show up as multiplicative gates
  in larger expressions — `Ge(end, start) * (end - start)` gives a
  clamped non-negative length. The simplifier proves and discharges
  many of them; the rest sit in the graph and resolve at run time.

## Symbols and the scope

A `Symbol` is an interned name. It lives in a `SymbolScope`, and every
`TypedModel` carries its own:

```rust
let scope = &model.symbols;
let s = scope.sym("S");                  // get-or-create
let expr = scope.parse_tdim("4 * S + 1")?;
```

Two symbols with the same name from different scopes are different
symbols. When you compare or substitute, the scope identity matters.
`model.symbols` is the only one you should reach for in normal use.

## Where symbols come from

The model loaders create symbols on your behalf:

- **ONNX**: a dynamic dim with a `dim_param` (e.g. `"batch_size"`) is
  parsed into a symbol of the same name in the model's scope. A
  dim_param of `"?"` or `"unk__N"` becomes an unknown without a name —
  use `set_input_fact` to constrain it.
- **NNEF / tract-OPL**: `dim_param` fragments in the textual graph
  become symbols at load time. The OPL serialiser writes them back
  out, so a round-tripped model keeps the same symbolic shape it had
  before.
- **Programmatic**: any code touching a model can call
  `model.symbols.sym(name)` to create one and use the resulting
  `Symbol` inside a `TDim` it builds.

## Binding them from the library

Two patterns, depending on whether the binding is fixed for the
deployment or varies per call.

**Bake the values in at build time** — replace symbols with constants
(or with other symbols) and re-run the optimiser against the new
shapes. The right call when the dim is a knob you set once at
deployment time (input resolution, fixed batch):

```rust
use std::collections::HashMap;

let s = model.symbols.sym("S");
let b = model.symbols.sym("B");
let mut subs = HashMap::new();
subs.insert(s, 224.into());           // S = 224
subs.insert(b, 1.into());             // B = 1
let model = model.set_symbols(&subs)?;
```

After this, the shapes are pure `Val(_)` and downstream passes treat
the model as fully concrete.

**Let the runtime resolve at call time** — keep the symbols in the
graph and just feed concrete-shaped tensors. The runtime matches each
input tensor's actual shape against the symbolic input fact and
records the symbol bindings for the run:

```rust
// Input fact is (B, 3, S, S, f32); inputs[0] is shape (1, 3, 224, 224).
// The plan infers B=1, S=224 from the input, propagates through every
// dependent shape, and runs.
let outputs = runnable.run(tvec!(input.into()))?;
```

This is the right choice when the binding varies per call (dynamic
batch, variable sequence length). The model's optimised form stays
shared; only the per-call shape-resolution table changes.

The two paths are not exclusive: bake the dimensions that never change
(e.g. a known batch size in a server with fixed concurrency) and leave
the rest symbolic for the runtime to bind.

## Setting input facts on an `InferenceModel` (ONNX-only)

This step lives on the `InferenceModel`, not on `Model`. It exists
because ONNX expresses shape and type partially: a dim can be a known
int, a named `dim_param`, **or simply absent** (`?` / `unk__N`), and
element types can be missing from `graph.input` annotations too. NNEF
does not have that problem — a NNEF model arrives already fully typed,
so `tract::nnef().load(path)` returns a `Model` directly and there is
nothing to set.

For ONNX, pin whatever the loader left unresolved before
`.into_model()`:

```rust
let mut m = tract::onnx()?.load(path)?;
m.set_input_fact(0, "B,3,224,224,f32")?;     // names a symbol B
let model = m.into_model()?;
```

The string form is parsed against `m`'s symbol scope, so `B` here is
the same symbol the rest of the graph already references (whatever the
ONNX `dim_param` named it). `into_model()` then runs analysis with the
pinned shape and produces a typed `Model` in the same form a NNEF load
would have given you directly.

## CLI counterparts

The CLI does the same things via flags:

- `-i N,3,224,224,f32` — set an input fact (per-input).
- `--set S=224 --set B=1` — bind symbols to constants. Equivalent to
  `set_symbols` on the library side.
- `--input-from-bundle io.npz` — derive concrete input shapes from
  the tensors actually present in the bundle; runs both "set input
  facts" and "concretise symbols" in one step. The library equivalent
  is the `set_input_fact` + `set_symbols` pair above.

## ONNX gotchas

Some ONNX exports carry `value_info` annotations that contradict
post-rewrite shapes, or output type annotations that disagree with
what tract infers. Three escape hatches on the `Onnx` loader:

```rust
let onnx = tract::onnx()?
    .with_ignore_output_shapes(true)?     // drop graph.output shape annotations
    .with_ignore_output_types(true)?      // drop graph.output type annotations
    .with_ignore_value_info(true)?;       // drop intermediate value_info
```

Which one bites depends on the exporter (and its version);
`with_ignore_output_shapes(true)` is the most frequently useful when
an export's `graph.output` shape annotations have not been kept in
sync with the actual graph.
