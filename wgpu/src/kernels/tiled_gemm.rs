//! The tiled GEMM core the matmul and convolution kernels share.

/// Threads of a tiled matmul or convolution workgroup, eight rows of threads
/// by sixteen over a `tile_m` by [`MATMUL_TILE`] output tile: each computes
/// `tile_m / 8` rows by [`TILED_COLS`] columns.
pub const TILED_THREADS: usize = 128;
const TILED_COLS: usize = 4;

/// Row extent of a tiled matmul or convolution workgroup over `rows` output
/// rows: the multiple of eight closest above an even split of `rows` into as
/// many tiles as [`MATMUL_TILE`]-row tiles would take, so the last tile carries
/// as few rows past `rows` as the tile count allows.
pub fn tile_rows(rows: usize) -> usize {
    let tiles = rows.div_ceil(MATMUL_TILE).max(1);
    rows.div_ceil(tiles).div_ceil(8).max(1) * 8
}

/// The body shared by [`matmul_tiled_module`] and [`conv_tiled_module`], from
/// the tile origin to the stores. Expects `lid`, `row0`, `col0`, `k_len` (the
/// reduction length) and `interior` (the output tile lies inside the result) in
/// scope, `ld_a(actx, row, k, guard)` and `ld_b(bctx, k, col, guard)` reading
/// one operand value (zero outside it when `guard`) given the `vec4<u32>`
/// contexts `actx` and `bctx` the kernel sets up, and `store_out(<store_args>row,
/// col, acc)`. `k_prologue`, when not empty, runs at the top of each `k` slice
/// (`k0` its first index) ahead of a barrier and the loads. `a_k_fast` and
/// `b_n_fast` are WGSL booleans telling whether `a` is
/// contiguous along `k` and `b` along columns; either way a thread fills one
/// whole `vec4` of the tile, and neighbouring threads walk the contiguous axis
/// so their loads coalesce. Tiles are only ever stored a whole `vec4` at a
/// time: a store to one lane at a runtime index goes wrong on Metal.
///
/// `a` is staged as rows of `vec4` along `k` and `b` as `k` rows of `vec4`
/// along columns, in the tensor type: one workgroup load hands a thread four
/// `k` steps of a row, and f16 tiles halve the workgroup-memory traffic that
/// bounds this kernel. The workgroup covers `tile_m` rows (a multiple of eight,
/// at most [`MATMUL_TILE`]) from `row0`. Accumulation is in f32, one `vec4`
/// per row a thread owns.
///
/// [`matmul_tiled_module`]: crate::kernels::matmul::matmul_tiled_module
/// [`conv_tiled_module`]: crate::kernels::conv::conv_tiled_module
pub(crate) fn tiled_gemm_core(
    t: &str,
    a_k_fast: &str,
    b_n_fast: &str,
    store_args: &str,
    k_prologue: &str,
    tile_m: usize,
) -> String {
    let tile = MATMUL_TILE;
    let tk = MATMUL_TILE_K;
    let tk4 = tk / 4;
    let tile4 = tile / 4;
    let threads = TILED_THREADS;
    let tn = TILED_COLS;
    let cols_threads = tile / tn;
    let tm = tile_m / (threads / cols_threads);
    let mut s = format!(
        "    let tx = lid % {cols_threads}u;
    let ty = lid / {cols_threads}u;
    let a_k_fast = {a_k_fast};
    let b_n_fast = {b_n_fast};
"
    );
    for i in 0..tm {
        for j in 0..tn {
            s += &format!("    var c{i}_{j} = 0.0;\n");
        }
    }
    let k_prologue = if k_prologue.is_empty() {
        String::new()
    } else {
        format!("{k_prologue}        workgroupBarrier();\n")
    };
    s += &format!(
        "    for (var k0 = 0u; k0 < k_len; k0 += {tk}u) {{
        let guard = !interior || k0 + {tk}u > k_len;
{k_prologue}        for (var i = lid; i < {aq}u; i += {threads}u) {{
            if (a_k_fast) {{
                let r = i / {tk4}u;
                let kq = i % {tk4}u;
                let k = k0 + kq * 4u;
                a_tile[r * {tk4}u + kq] = vec4<{t}>(ld_a(actx, row0 + r, k, guard), ld_a(actx, row0 + r, k + 1u, guard), ld_a(actx, row0 + r, k + 2u, guard), ld_a(actx, row0 + r, k + 3u, guard));
            }} else {{
                let r = i % {tile_m}u;
                let kq = i / {tile_m}u;
                let k = k0 + kq * 4u;
                a_tile[r * {tk4}u + kq] = vec4<{t}>(ld_a(actx, row0 + r, k, guard), ld_a(actx, row0 + r, k + 1u, guard), ld_a(actx, row0 + r, k + 2u, guard), ld_a(actx, row0 + r, k + 3u, guard));
            }}
        }}
        for (var i = lid; i < {bq}u; i += {threads}u) {{
            if (b_n_fast) {{
                let kk = i / {tile4}u;
                let cq = i % {tile4}u;
                let c = col0 + cq * 4u;
                b_tile[kk * {tile4}u + cq] = vec4<{t}>(ld_b(bctx, k0 + kk, c, guard), ld_b(bctx, k0 + kk, c + 1u, guard), ld_b(bctx, k0 + kk, c + 2u, guard), ld_b(bctx, k0 + kk, c + 3u, guard));
            }} else {{
                let kk = i % {tk}u;
                let cq = i / {tk}u;
                let c = col0 + cq * 4u;
                b_tile[kk * {tile4}u + cq] = vec4<{t}>(ld_b(bctx, k0 + kk, c, guard), ld_b(bctx, k0 + kk, c + 1u, guard), ld_b(bctx, k0 + kk, c + 2u, guard), ld_b(bctx, k0 + kk, c + 3u, guard));
            }}
        }}
        workgroupBarrier();
        for (var kq = 0u; kq < {tk4}u; kq++) {{
",
        aq = tile_m * tk4,
        bq = tk * tile4,
    );
    for i in 0..tm {
        s += &format!(
            "            let a{i} = vec4<f32>(a_tile[(ty * {tm}u + {i}u) * {tk4}u + kq]);\n"
        );
    }
    for (sub, comp) in ["x", "y", "z", "w"].iter().enumerate() {
        s += &format!(
            "            let b{sub} = vec4<f32>(b_tile[(kq * 4u + {sub}u) * {tile4}u + tx]);\n"
        );
        for i in 0..tm {
            for (j, cj) in ["x", "y", "z", "w"].iter().enumerate() {
                s += &format!("            c{i}_{j} = fma(a{i}.{comp}, b{sub}.{cj}, c{i}_{j});\n");
            }
        }
    }
    s += "        }\n        workgroupBarrier();\n    }\n";
    s += &format!("    let r = row0 + ty * {tm}u;\n    let c = col0 + tx * {tn}u;\n");
    for i in 0..tm {
        for j in 0..tn {
            s += &format!("    store_out({store_args}r + {i}u, c + {j}u, c{i}_{j});\n");
        }
    }
    s
}

/// The workgroup tiles [`tiled_gemm_core`] stages its operands in.
pub(crate) fn tiled_gemm_tiles(t: &str, tile_m: usize) -> String {
    let tile = MATMUL_TILE;
    let tk = MATMUL_TILE_K;
    format!(
        "var<workgroup> a_tile: array<vec4<{t}>, {}>;\nvar<workgroup> b_tile: array<vec4<{t}>, {}>;\n",
        tile_m * tk / 4,
        tile * tk / 4
    )
}

/// Output columns of each tiled matmul or convolution workgroup, and the most
/// rows one covers (see [`tile_rows`]).
pub const MATMUL_TILE: usize = 64;
/// `k` values staged in workgroup memory per step of the tiled matmul.
pub const MATMUL_TILE_K: usize = 16;
