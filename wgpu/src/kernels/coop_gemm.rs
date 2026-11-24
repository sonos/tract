//! Cooperative-matrix GEMM for channel-first products, `out[n][m] = sum_k b[n][k] * a[k][m]`.
//!
//! One 32-lane subgroup owns a tile of 8x8 blocks. The operands are indexed as nested arrays so
//! that every constant offset is its own dimension, which the Metal compiler turns into an address
//! immediate; a constant added to a `u32` index is computed at run time instead. A short product
//! with a long `k` splits `k` over several subgroups of one workgroup. An `f16` product accumulates
//! in `f32` where the device has such a tile, so it needs no `TRACT_WGPU_ALLOW_F16_ACCUMULATION`.
//! Browsers have none; there the product runs in `f16`, behind that knob.

use std::fmt::Write;

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::WgpuQueue;
use crate::kernels::coop_dialect::{Dialect, Side, accumulator};
use crate::kernels::matmul::Transposes;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, LayoutKind, ShaderDtype, binary_ops_wgsl, epilogue_body_at,
    unary_ops_wgsl,
};
use crate::kernels::subgroup::{
    bindings, chosen_subgroup, dispatch_groups, enables, knob_on, limits_of, stride_pair,
};
use crate::utils::element_offset;

const LANES: u32 = 32;
const BLOCK: usize = 8;
const MIN_K: usize = 24;
/// Workgroups the larger register tile must still leave.
const MIN_TILE_GROUPS: usize = 196;
/// Workgroups under which a long `k` is split over several subgroups.
const MAX_SPLIT_GROUPS: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Plan {
    /// Output blocks of 8 along `m` and `n`.
    tm: usize,
    tn: usize,
    /// `k` consumed by one unrolled loop iteration.
    step: usize,
    /// Subgroups the `k` iterations are split over.
    split: usize,
}

/// Register tile of at most eight blocks if it leaves enough workgroups to fill the device, at
/// most six otherwise. The tile divides the product, and the squarest one loads the least.
fn tile(m: usize, n: usize) -> Option<(usize, usize)> {
    let (blocks_m, blocks_n) = (m / BLOCK, n / BLOCK);
    for (cap, floor) in [(8, MIN_TILE_GROUPS), (6, 0)] {
        let best = (1..=6)
            .flat_map(|tn| (1..=6).map(move |tm| (tm, tn)))
            .filter(|(tm, tn)| {
                tm * tn <= cap
                    && blocks_m.is_multiple_of(*tm)
                    && blocks_n.is_multiple_of(*tn)
                    && (blocks_m / tm) * (blocks_n / tn) >= floor
            })
            .min_by_key(|(tm, tn)| 1000 * (tm + tn) / (tm * tn));
        if best.is_some() {
            return best;
        }
    }
    None
}

fn plan(m: usize, k: usize, n: usize) -> Option<Plan> {
    if m == 0 || n == 0 || k < MIN_K {
        return None;
    }
    if !m.is_multiple_of(BLOCK) || !k.is_multiple_of(BLOCK) || !n.is_multiple_of(BLOCK) {
        return None;
    }
    let (tm, tn) = tile(m, n)?;
    let step = [32, 16, 8].into_iter().find(|s| k.is_multiple_of(*s))?;
    let iterations = k / step;
    let groups = (m / (BLOCK * tm)) * (n / (BLOCK * tn));
    let split = match iterations {
        16.. if groups < MAX_SPLIT_GROUPS => 4,
        8..16 if groups < MAX_SPLIT_GROUPS => 2,
        _ => 1,
    };
    Some(Plan { tm, tn, step, split })
}

/// The epilogue computed in `f32` on operands stored as `f16`.
fn widened(body: &str) -> String {
    body.lines()
        .map(|line| {
            if line.trim_start().starts_with("let e") {
                format!("{}\n", line.replacen(" = ", " = f32(", 1).replace("];", "]);"))
            } else {
                format!("{line}\n")
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn shader(
    dialect: Dialect,
    dt: ShaderDtype,
    acc: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    plan: Plan,
    m: usize,
    k: usize,
) -> String {
    let Plan { tm, tn, step, split } = plan;
    let (t, st) = (dt.wgsl(), acc.wgsl());
    let (bm, bn) = (BLOCK * tm, BLOCK * tn);
    let iterations = k / step;
    let span = tm * tn * 64;
    let mut s = enables(dt, true, false);
    let [a_type, b_type] = match dialect {
        Dialect::Naga => [
            format!("array<array<array<array<{t}, {bm}>, {}>, {step}>>", m / bm),
            format!("array<array<array<array<{t}, {step}>, {iterations}>, {bn}>>"),
        ],
        Dialect::Dawn => [
            format!("array<array<array<{t}, {}>, {}>>", BLOCK * m, step / BLOCK),
            format!("array<array<array<{t}, {}>, {tn}>>", BLOCK * k),
        ],
    };
    s.push_str(&bindings(dt, extras, &a_type, &b_type));
    writeln!(s, "var<workgroup> stage: array<{st}, {}>;", split * span).unwrap();
    s.push_str(&unary_ops_wgsl(st));
    s.push_str(&binary_ops_wgsl(st));
    let subgroup_id = if split > 1 { ", @builtin(subgroup_id) sg: u32" } else { "" };
    writeln!(
        s,
        "@compute @workgroup_size({})\nfn matmul_coop_{}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32{subgroup_id}) {{",
        LANES as usize * split,
        dt.suffix()
    )
    .unwrap();
    s.push_str("    let lane = li & 31u;\n");
    for (i, j) in blocks(tm, tn) {
        writeln!(s, "    var acc_{i}_{j} = {};", dialect.accumulator(st)).unwrap();
    }
    let first = if split > 1 { "sg" } else { "0u" };
    writeln!(s, "    for (var kb = {first}; kb < {iterations}u; kb += {split}u) {{").unwrap();
    for q in (0..step).step_by(BLOCK) {
        s.push_str("    {\n");
        for j in 0..tn {
            let place = format!("b[wid.x][{}][kb][{q}]", j * BLOCK);
            let block = (&*format!("b[wid.x][{j}]"), &*format!("kb * {step}u + {q}u"));
            let load = dialect.load(Side::Left, t, &place, block, k);
            writeln!(s, "        let b_{j} = {load};").unwrap();
        }
        for i in 0..tm {
            let place = format!("a[kb][{q}][wid.y][{}]", i * BLOCK);
            let offset = format!("wid.y * {bm}u + {}u", i * BLOCK);
            let block = (&*format!("a[kb][{}]", q / BLOCK), &*offset);
            let load = dialect.load(Side::Right, t, &place, block, m);
            writeln!(s, "        let a_{i} = {load};").unwrap();
        }
        for (i, j) in blocks(tm, tn) {
            let acc = format!("acc_{i}_{j}");
            let sum = dialect.multiply_add(&format!("b_{j}"), &format!("a_{i}"), &acc);
            writeln!(s, "        {acc} = {sum};").unwrap();
        }
        s.push_str("    }\n");
    }
    s.push_str("    }\n");
    let base = if split > 1 { format!("sg * {span}u + ") } else { String::new() };
    for (i, j) in blocks(tm, tn) {
        let offset = format!("{base}{}u", (j * tm + i) * 64);
        let store = dialect.store(&format!("acc_{i}_{j}"), "stage", &offset, BLOCK);
        writeln!(s, "    {store}").unwrap();
    }
    s.push_str("    workgroupBarrier();\n    if (li < 32u) {\n");
    let body = epilogue_body_at(epilogue, "col", "oe");
    let body = if acc != dt { widened(&body) } else { body };
    let store = if acc != dt { "f16(v)" } else { "v" };
    for (i, j) in blocks(tm, tn) {
        for half in 0..2 {
            let at = (j * tm + i) * 64 + 32 * half;
            let sum = (0..split)
                .map(|q| format!("stage[{}u + lane]", at + q * span))
                .collect::<Vec<_>>()
                .join(" + ");
            write!(
                s,
                "    {{\n        let col = wid.x * {bn}u + {}u + (lane >> 3u);\n        let row = wid.y * {bm}u + {}u + (lane & 7u);\n        let oe = row + col * {m}u;\n        var v = {sum};\n{body}        outp[params.off_out + oe] = {store};\n    }}\n",
                j * BLOCK + 4 * half,
                i * BLOCK,
            )
            .unwrap();
        }
    }
    s.push_str("    }\n}\n");
    s
}

fn blocks(tm: usize, tn: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..tn).flat_map(move |j| (0..tm).map(move |i| (i, j)))
}

/// Runs the product when the device, the layout and the shape allow the kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dispatch(
    q: &WgpuQueue,
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    dt: ShaderDtype,
    f16_accumulation: bool,
    m: usize,
    k: usize,
    n: usize,
    prefix: usize,
    out_shape: &[usize],
) -> TractResult<bool> {
    let caps = *q.context().subgroup_caps();
    let (max_invocations, max_x) = limits_of(q);
    let subgroup = chosen_subgroup(&caps).filter(|c| c.size == LANES && !c.varying && !c.pin);
    let fits = prefix == 1
        && knob_on(&caps)
        && caps.matrix
        && subgroup.is_some()
        && stride_pair(a, t.a) == (1, m as isize)
        && stride_pair(b, t.b) == (1, k as isize)
        && stride_pair(output, t.c) == (1, m as isize)
        && element_offset(a, 0) == 0
        && element_offset(b, 0) == 0;
    let Some(acc) = accumulator(&caps, dt, f16_accumulation).filter(|_| fits) else {
        return Ok(false);
    };
    let Some(plan) = plan(m, k, n) else { return Ok(false) };
    let threads = LANES * plan.split as u32;
    if threads > max_invocations.min(max_x) {
        return Ok(false);
    }
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let pipeline = q.context().chain_pipeline(
        ("matmul_coop", dt, acc, plan, m, k, epilogue.to_vec(), extras.len()),
        layout,
        EntryPoint::typed("matmul_coop", dt),
        || shader(Dialect::CURRENT, dt, acc, epilogue, extras.len(), plan, m, k),
    )?;
    let groups = [(n / (BLOCK * plan.tn)) as u32, (m / (BLOCK * plan.tm)) as u32, 1];
    dispatch_groups(
        q,
        "matmul_coop",
        &pipeline,
        layout,
        a,
        b,
        extras,
        output,
        out_shape,
        prefix,
        m,
        k,
        n,
        t,
        0,
        0,
        groups,
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::TRACT_WGPU_SUBGROUPS;

    #[test]
    fn tile_divides_the_product_and_keeps_the_device_busy() {
        // Large output: eight blocks, squarest first.
        let (tm, tn) = tile(3600, 960).unwrap();
        assert_eq!((3600 / 8 % tm, 960 / 8 % tn, tm * tn <= 8), (0, 0, true));
        assert_eq!((tm * tn, (450 / tm) * (120 / tn) >= MIN_TILE_GROUPS), (8, true));
        // Small output: at most six blocks.
        let (tm, tn) = tile(240, 136).unwrap();
        assert!(tm * tn <= 6);
        // 17 blocks of 8 along n leave no choice but one.
        assert_eq!(tile(240, 136).unwrap().1, 1);
    }

    #[test]
    fn plan_declines_what_the_kernel_cannot_run() {
        assert!(plan(3600, 160, 960).is_some());
        assert!(plan(84, 64, 384).is_none());
        assert!(plan(3600, 160, 961).is_none());
        assert!(plan(3600, 164, 960).is_none());
        assert!(plan(3600, 16, 64).is_none());
        assert!(plan(0, 64, 64).is_none());
    }

    #[test]
    fn long_k_is_split_only_when_the_output_is_small() {
        assert_eq!(plan(3600, 160, 960).unwrap().split, 1);
        assert_eq!(plan(40, 672, 112).unwrap().split, 4);
        assert_eq!(plan(240, 816, 136).unwrap().split, 1);
        assert_eq!(plan(40, 256, 40).unwrap().split, 2);
        assert_eq!(plan(8, 768, 768).unwrap().step, 32);
        assert_eq!(plan(96, 24, 40).unwrap().step, 8);
    }

    #[test]
    fn f16_operands_accumulate_in_f32() {
        let plan = plan(576, 96, 576).unwrap();
        let source =
            shader(Dialect::Naga, ShaderDtype::F16, ShaderDtype::F32, &[], 0, plan, 576, 96);
        assert!(source.contains("coop_mat8x8<f32, C>()"));
        assert!(source.contains("coop_mat8x8<f16, A>"));
        assert!(source.contains("var<workgroup> stage: array<f32"));
        assert!(!source.contains("subgroup_id"));
    }

    #[test]
    fn dawn_accumulates_in_f16_from_arrays_of_eight_rows() {
        let plan = plan(576, 96, 576).unwrap();
        let source =
            shader(Dialect::Dawn, ShaderDtype::F16, ShaderDtype::F16, &[], 0, plan, 576, 96);
        assert!(source.contains("subgroup_matrix_result<f16, 8, 8>()"));
        assert!(source.contains("array<array<array<f16, 4608>, 4>>"));
        assert!(source.contains("var<workgroup> stage: array<f16"));
        assert!(!source.contains("coop_mat") && !source.contains("coopLoad"));
    }

    /// Device values. Needs `TRACT_WGPU_SUBGROUPS=1` set before the first device is created:
    /// `TRACT_WGPU_SUBGROUPS=1 cargo test -p tract-wgpu --lib coop_gemm -- --test-threads=1`.
    #[test]
    fn coop_gemm_matches_the_reference() -> TractResult<()> {
        if !TRACT_WGPU_SUBGROUPS.get() {
            eprintln!("cooperative GEMM test skipped: TRACT_WGPU_SUBGROUPS is off");
            return Ok(());
        }
        let caps = *crate::wgpu_context().subgroup_caps();
        if accumulator(&caps, ShaderDtype::F16, false).is_none() {
            eprintln!("cooperative GEMM test skipped: {caps:?}");
            return Ok(());
        }
        // A tile of eight, a tile of six with a short k, an odd block count, a single strip, and
        // k split over four and over two subgroups.
        for (m, k, n) in [
            (576, 96, 576),
            (240, 136, 816),
            (72, 88, 56),
            (8, 256, 256),
            (40, 672, 112),
            (40, 256, 40),
        ] {
            check::<f16>(m, k, n).with_context(|| format!("f16 {m}x{k}x{n}"))?;
            check::<f32>(m, k, n).with_context(|| format!("f32 {m}x{k}x{n}"))?;
        }
        Ok(())
    }

    fn check<T: Datum + num_traits::Float + num_traits::FromPrimitive>(
        m: usize,
        k: usize,
        n: usize,
    ) -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;
        let dt = T::datum_type();
        let is_f16 = dt == DatumType::F16;
        let fill = |seed: u64, len: usize, scale: f64| -> Vec<T> {
            let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    T::from_f64(((s >> 33) as f64 / u32::MAX as f64 - 0.5) * scale).unwrap()
                })
                .collect()
        };
        let a_h = fill(11, k * m, 1.0);
        let b_h = fill(12, n * k, (8.0 / k as f64).sqrt());
        let bias_h = fill(13, n, 0.5);
        let a = Tensor::from_shape(&[k, m], &a_h)?.into_device()?;
        let b = Tensor::from_shape(&[n, k], &b_h)?.into_device()?;
        let bias = Tensor::from_shape(&[n], &bias_h)?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(dt, &[n, m])?;
        let epilogue = [
            ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false },
            ChainStep::Unary("abs".into()),
        ];
        crate::set_dispatch_capture(true);
        let _ = crate::take_dispatch_labels();
        crate::kernels::matmul::wgpu_matmul_dispatch(
            Transposes { a: true, b: true, c: true },
            &epilogue,
            &a,
            &b,
            &[&bias],
            &output,
            false,
        )?;
        let got = pollster::block_on(crate::to_host_async(&output))?;
        let labels = crate::take_dispatch_labels();
        ensure!(labels.contains(&"matmul_coop"), "the kernel did not run: {labels:?}");
        let got = got.to_plain_array_view::<T>()?;
        let got = got.as_slice().context("the output is not contiguous")?;
        for ni in 0..n {
            for mi in 0..m {
                let sum = (0..k)
                    .map(|q| b_h[ni * k + q].to_f64().unwrap() * a_h[q * m + mi].to_f64().unwrap())
                    .sum::<f64>();
                let want = (sum + bias_h[ni].to_f64().unwrap()).abs();
                let have = got[ni * m + mi].to_f64().unwrap();
                // f16: one unit in the last place of the result. f32: 1e-5.
                let tolerance = if is_f16 { (want * 2f64.powi(-10)).max(5e-4) } else { 1e-5 };
                ensure!((have - want).abs() <= tolerance, "[{ni}][{mi}] {have} instead of {want}");
            }
        }
        Ok(())
    }
}
