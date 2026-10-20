//! ggml IQ2_S: 256-value super-blocks of 8-value points from a 1024-entry grid.
//!
//! The quantizer is llama.cpp `quantize_row_iq2_s_ref`, without importance matrix.
use super::ggml_k::QK_K;
use super::ggml_k::nearest_int;
use super::iq2_s_kgrid::KGRID_2BIT_1024;
use super::opaque::impl_opaque_block_quant;
use super::opaque::load_f16;
use super::opaque::store_f16;
use super::*;
use std::sync::OnceLock;

const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;
/// Packed 2-bit codes whose digits are at most 2. 43690 is eight digits of 2.
const KMAP_SIZE: usize = 43692;

/// ggml IQ2_S block: f16 `d`, the low index bytes and sign bytes of the 32 grid
/// points, their high index bits in 8 bytes, then sixteen 4-bit group scales, 82 bytes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct IQ2SF;

/// The ggml IQ2_S block format.
pub const IQ2_S: IQ2SF = IQ2SF;

/// Dequant bytes for levels 0, 1, 2. ggml's `iq2s_grid` stores these, not the
/// 1/3/5 points the neighbour search uses.
const CODE_BYTE: [u8; 3] = [0x08, 0x19, 0x2b];

struct Search {
    /// On-grid points as odd levels 1, 3, 5, in `kgrid_2bit_1024` order.
    grid: Vec<[i8; 8]>,
    /// Packed index → grid index, or `-(neighbour_offset + 1)` when off-grid.
    map: Vec<i32>,
    /// Off-grid runs: count, then grid indices of the nearest shell.
    neighbours: Vec<u16>,
}

fn codes() -> &'static [[u8; 8]; 1024] {
    static CODES: OnceLock<[[u8; 8]; 1024]> = OnceLock::new();
    CODES.get_or_init(|| {
        let mut codes = [[0u8; 8]; 1024];
        for (i, &code) in KGRID_2BIT_1024.iter().enumerate() {
            for (k, byte) in codes[i].iter_mut().enumerate() {
                let level = usize::from((code >> (2 * k)) & 3);
                *byte = CODE_BYTE[level];
            }
        }
        codes
    })
}

fn search() -> &'static Search {
    static SEARCH: OnceLock<Search> = OnceLock::new();
    SEARCH.get_or_init(build_search)
}

fn odd_levels(index: u32) -> [i8; 8] {
    let mut pos = [0i8; 8];
    for (i, slot) in pos.iter_mut().enumerate() {
        let level = (index >> (2 * i)) & 3;
        *slot = (2 * level + 1) as i8;
    }
    pos
}

fn l2(pg: &[i8; 8], pos: &[i8; 8]) -> i32 {
    let mut d2 = 0i32;
    for k in 0..8 {
        let diff = i32::from(pg[k]) - i32::from(pos[k]);
        d2 += diff * diff;
    }
    d2
}

/// Nearest shell only. ggml asks for one distance shell (`nwant = 1`) and
/// `qsort`s by distance then grid index, so the shell ends up in index order.
fn build_search() -> Search {
    let grid: Vec<[i8; 8]> =
        KGRID_2BIT_1024.iter().copied().map(|code| odd_levels(u32::from(code))).collect();
    let mut map = vec![-1i32; KMAP_SIZE];
    for (i, pos) in grid.iter().enumerate() {
        let mut index = 0u16;
        for (k, &p) in pos.iter().enumerate() {
            let q = (u16::from(p as u8) - 1) / 2;
            index |= q << (2 * k);
        }
        map[usize::from(index)] = i as i32;
    }
    let mut neighbours = Vec::new();
    for (i, slot) in map.iter_mut().enumerate() {
        if *slot >= 0 {
            continue;
        }
        let pos = odd_levels(i as u32);
        let mut best = i32::MAX;
        for pg in &grid {
            let d2 = l2(pg, &pos);
            if d2 < best {
                best = d2;
            }
        }
        let start = neighbours.len();
        neighbours.push(0);
        let mut n = 0u16;
        for (j, pg) in grid.iter().enumerate() {
            if l2(pg, &pos) == best {
                neighbours.push(j as u16);
                n += 1;
            }
        }
        neighbours[start] = n;
        *slot = -((start as i32) + 1);
    }
    Search { grid, map, neighbours }
}

fn pack_levels(levels: &[i8]) -> u16 {
    let mut u = 0u16;
    for (i, &level) in levels.iter().enumerate() {
        u |= u16::from(level as u8) << (2 * i);
    }
    u
}

fn find_best_neighbour(
    neighbours: &[u16],
    grid: &[[i8; 8]],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l_out: &mut [i8],
) {
    let num = usize::from(neighbours[0]);
    assert!(num > 0);
    let mut best = f32::MAX;
    let mut grid_index = -1i32;
    for j in 1..=num {
        let pg = &grid[usize::from(neighbours[j])];
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let diff = scale * f32::from(pg[i]) - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best {
            best = d2;
            grid_index = i32::from(neighbours[j]);
        }
    }
    assert!(grid_index >= 0);
    let pg = &grid[grid_index as usize];
    for i in 0..8 {
        l_out[i] = (pg[i] - 1) / 2;
    }
}

fn snap_octet(
    search: &Search,
    xval: &[f32],
    waux: &[f32],
    id: f32,
    scale: f32,
    levels: &mut [i8],
) -> bool {
    for i in 0..8 {
        let l = nearest_int(0.5 * (id * xval[i] - 1.0)).clamp(0, 2);
        levels[i] = l as i8;
    }
    let grid_index = search.map[usize::from(pack_levels(levels))];
    if grid_index < 0 {
        let off = (-grid_index - 1) as usize;
        find_best_neighbour(&search.neighbours[off..], &search.grid, xval, waux, scale, levels);
        false
    } else {
        true
    }
}

/// One group of 16. Returns the group scale and writes two grid/sign pairs.
fn quantize_group(
    xb: &[f32],
    sigma2: f32,
    search: &Search,
    ib: usize,
    qs: &mut [u8],
    qh: &mut [u8],
) -> f32 {
    let mut weight = [0.0f32; 16];
    let mut waux = [0.0f32; 16];
    let mut xval = [0.0f32; 16];
    let mut signs = [0u8; 2];
    for i in 0..16 {
        weight[i] = 0.25 * sigma2 + xb[i] * xb[i];
        waux[i] = weight[i].sqrt();
    }
    for k in 0..2 {
        let mut s = 0u8;
        for i in 0..8 {
            let v = xb[8 * k + i];
            if v >= 0.0 {
                xval[8 * k + i] = v;
            } else {
                xval[8 * k + i] = -v;
                s |= 1u8 << i;
            }
        }
        signs[k] = s;
    }
    let max = xval.iter().copied().fold(xval[0], f32::max);
    if max < GROUP_MAX_EPS_IQ2_S {
        return 0.0;
    }
    let mut levels = [0i8; 16];
    let mut best = 0.0f32;
    // 2 * kMaxQ - 1, with kMaxQ = 3. The search never emits level 3.
    let mut scale = max / 5.0;
    let mut on_grid = [true, true];
    for is in -9..=9 {
        let id = (5.0 + is as f32 * 0.1) / max;
        let this_scale = 1.0 / id;
        let mut aux = [0i8; 16];
        let mut on_aux = [true, true];
        for k in 0..2 {
            on_aux[k] = snap_octet(
                search,
                &xval[8 * k..8 * k + 8],
                &waux[8 * k..8 * k + 8],
                id,
                this_scale,
                &mut aux[8 * k..8 * k + 8],
            );
        }
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for i in 0..16 {
            let q = 2.0 * f32::from(aux[i]) + 1.0;
            sumqx += weight[i] * xval[i] * q;
            sumq2 += weight[i] * q * q;
        }
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            scale = sumqx / sumq2;
            best = scale * sumqx;
            levels = aux;
            on_grid = on_aux;
        }
    }
    if on_grid.iter().any(|on| !on) && scale > 0.0 {
        let id = 1.0 / scale;
        for k in 0..2 {
            if on_grid[k] {
                continue;
            }
            snap_octet(
                search,
                &xval[8 * k..8 * k + 8],
                &waux[8 * k..8 * k + 8],
                id,
                scale,
                &mut levels[8 * k..8 * k + 8],
            );
        }
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for i in 0..16 {
            let q = 2.0 * f32::from(levels[i]) + 1.0;
            sumqx += weight[i] * xval[i] * q;
            sumq2 += weight[i] * q * q;
        }
        if sumq2 > 0.0 {
            scale = sumqx / sumq2;
        }
    }
    if scale < 0.0 {
        scale = -scale;
        signs[0] = !signs[0];
        signs[1] = !signs[1];
    }
    for k in 0..2 {
        let u = pack_levels(&levels[8 * k..8 * k + 8]);
        let grid_index = search.map[usize::from(u)];
        assert!(grid_index >= 0, "iq2_s point {u} is not on the grid");
        let grid_index = grid_index as u16;
        let i8 = 2 * ib + k;
        qs[i8] = (grid_index & 0xff) as u8;
        // Two high bits per octet, four octets per qh byte.
        qh[i8 / 4] |= ((grid_index >> 8) as u8) << (2 * (i8 % 4));
        qs[32 + i8] = signs[k];
    }
    scale
}

fn quant_iq2_s(x: &[f32], quant: &mut [u8]) {
    let search = search();
    quant.fill(0);
    let mut sumx2 = 0.0f32;
    for &xi in x {
        sumx2 += xi * xi;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    let mut group_scales = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut qs = [0u8; 64];
    let mut qh = [0u8; 8];
    for ib in 0..16 {
        group_scales[ib] =
            quantize_group(&x[16 * ib..16 * ib + 16], sigma2, search, ib, &mut qs, &mut qh);
        if group_scales[ib] > max_scale {
            max_scale = group_scales[ib];
        }
    }
    if max_scale == 0.0 {
        return;
    }
    // Nibbles are fit against max/31. The stored f16 is that value times 0.9875,
    // and dequant rebuilds the group scale as d * (nibble + 0.5) * 0.25.
    let d = max_scale / 31.0;
    store_f16(&mut quant[0..2], d * 0.9875);
    let id = 1.0 / d;
    let mut scale_bytes = [0u8; 8];
    for ib in 0..16 {
        let l = nearest_int(0.5 * (id * group_scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            scale_bytes[ib / 2] = l;
        } else {
            scale_bytes[ib / 2] |= l << 4;
        }
    }
    quant[2..66].copy_from_slice(&qs);
    quant[66..74].copy_from_slice(&qh);
    quant[74..82].copy_from_slice(&scale_bytes);
}

fn dequant_iq2_s(quant: &[u8], block: &mut [f32]) {
    let codes = codes();
    let d = load_f16(&quant[0..2]);
    let qs = &quant[2..66];
    let qh = &quant[66..74];
    let scales = &quant[74..82];
    let mut out_i = 0;
    for ib32 in 0..8 {
        let db0 = d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25;
        let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
        for l in 0..4 {
            let dl = if l < 2 { db0 } else { db1 };
            // `<<` and `&` associate left, so the mask applies after the shift.
            let shift = (8 - 2 * l) as u32;
            let hi = (u32::from(qh[ib32]) << shift) & 0x300;
            let index = usize::from(u16::from(qs[4 * ib32 + l]) | hi as u16);
            let signs = qs[32 + 4 * ib32 + l];
            for (j, &byte) in codes[index].iter().enumerate() {
                let mag = if signs & (1u8 << j) != 0 { -1.0 } else { 1.0 };
                block[out_i] = dl * f32::from(byte) * mag;
                out_i += 1;
            }
        }
    }
}

impl_opaque_block_quant!(IQ2SF, "IQ2_S", 256, 82, quant_iq2_s, dequant_iq2_s);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::block_quant::ggml_k::tests::assert_ggml_ref;
    use crate::frame::block_quant::ggml_k::tests::assert_roundtrip;
    use crate::frame::block_quant::ggml_k::tests::ramp;

    #[test]
    fn iq2_s_roundtrip() {
        assert_roundtrip(&IQ2_S, 1.25);
    }

    /// Bytes and values from llama.cpp `quantize_row_iq2_s_ref` / `dequantize_row_iq2_s`.
    /// ggml's own max error on this ramp is about 0.98, so the bytes are the check.
    #[test]
    fn iq2_s_matches_ggml_ref() {
        assert_ggml_ref(
            &IQ2_S,
            &ramp(256, 0),
            "a82379f32f67b63064b6416db7ae6d479f7d47932f799d2f67b6308db630d3b7416d29b5522bb5526aa5566aa5d44aadd44aa9955aa995522bb5522ba5566aa5564ad3a6699ad6b56d9bdfdcceadeedccbeb",
            &[(0, 0xbfb9_6cc0), (32, 0xbebf_6800), (255, 0x3fad_7640)],
        );
        // A skipped group keeps grid index 0, whose points dequantize to a small
        // positive value, so the zero prefix of a non-zero block does not come back as 0.
        assert_ggml_ref(
            &IQ2_S,
            &ramp(256, 32),
            "f42200000000b630d3b6416db7ae6d479f7d47932f799d2f67b630d3b630d3b7416d00000000b5526aa5566aa5d44aadd44aa9955aa995522bb5522ba5566aa5564a00b6699ad6b56d9b00dddfbfffedddfc",
            &[(0, 0x3c5e_8000), (32, 0xbebb_bc00), (255, 0x3fa8_6560)],
        );
        assert_ggml_ref(&IQ2_S, &[0f32; 256], &"00".repeat(82), &[(0, 0), (255, 0)]);
    }
}
