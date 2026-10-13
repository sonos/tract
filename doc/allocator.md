# Allocator choice

tract allocates on every eval. A plan hands each op its output tensor fresh,
so a streaming graph mallocs and frees a handful of intermediates per pulse,
and a load does one large burst while parsing and optimizing. The allocator is
therefore part of the engine's cost, and it belongs to the application: tract
is a library, so nothing below is something the engine can pick for you.

## Use jemalloc

glibc's allocator is the expensive one on the graphs tract runs. On small
pulsed graphs — the ones whose per-turn cost is mostly bookkeeping rather than
a steady-state matmul — jemalloc is worth 5 to 9% of eval time on aarch64 and
riscv64, and about half that on x86_64. It also cuts the minor-fault count of a
load by an order of magnitude: glibc can settle into a regime where a per-eval
intermediate is mapped and unmapped on every pass. musl's allocator has the
same failure mode at a smaller grain: a process can land in a regime where the
same small slot groups are unmapped and remapped on every eval.

Darwin's allocator has neither problem, and jemalloc measured slower there:
keep the system allocator on Apple targets.

The gain is not reachable by tuning glibc. Raising `M_MMAP_THRESHOLD` and
`M_TRIM_THRESHOLD` (or their `MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_`
environment forms) removes most of the fault churn and none of the time: what
costs is the allocator's own fast path, not the mapping.

```toml
[dependencies]
tikv-jemallocator = "0.6"
```

```rust
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

## Set `retain:false`

jemalloc's `opt.retain` defaults to true on 64-bit targets: a purged extent
keeps both its mapping and its dirty pages, so the allocator can reuse the
address range without asking the kernel again. Whether those pages ever come
back is up to the kernel, and on some of them they do not. A graph holding
68 MB of live tensors sat in a 338 MB resident set on a 5.4 aarch64 kernel,
while the same binary on a 4.9 kernel of the same SoC stayed at 115 MB. The
pages are dirty and anonymous, not lazily freed, so nothing reclaims them
under memory pressure short of swap.

Turning retain off restores a resident set that tracks the live one, and costs
nothing in eval time — the extra mapping work lands in the load path, where it
is worth about 8% of the time to a ready model on a big graph. It is already
jemalloc's own default on 32-bit targets.

Bake it into the binary, so a deployment cannot lose it by clearing the
environment:

```rust
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"retain:false\0";
```

The symbol name carries tikv-jemalloc-sys' `_rjem_` prefix; the unprefixed
`malloc_conf` is what a build without that prefix wants. The same string can
be passed at run time as `_RJEM_MALLOC_CONF` for a quick check.

Do not reach for `dirty_decay_ms:0` / `muzzy_decay_ms:0` on top of it. Every
free then purges immediately, so a pulsed graph pays the mapping churn again on
each eval, and most of what jemalloc brings is gone.

## Resident set right after load

jemalloc keeps freed pages mapped for its dirty decay period (10 s by default)
so it can hand them out again without a fault. A load frees most of what it
allocates within that period, so the resident set measured right after a model
is ready can be a large multiple of what it holds on a small model. It settles
once the decay runs. An application that wants the settled figure at once can
purge every arena after loading:

```rust
unsafe {
    tikv_jemalloc_sys::mallctl(
        c"arena.4096.purge".as_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
    );
}
```

`4096` is `MALLCTL_ARENAS_ALL`.

## Benching

`tract`'s own CLI links jemalloc with this configuration under its `jemalloc`
feature, which `bench-suite` turns on (Apple targets excepted), so the bench
numbers describe the configuration above rather than a glibc or musl build. The
CLI purges right before its load and model-ready readings, so the RSS it records
there is the settled figure.
