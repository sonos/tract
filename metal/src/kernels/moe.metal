#include <metal_stdlib>
using namespace metal;

enum RouteGateMode : uint {
    RouteGateSoftmaxTopk = 0,
    RouteGateSoftmaxAll = 1,
    RouteGateSigmoid = 2,
    RouteGateRaw = 3,
};

// Top-k selection over precomputed router scores [token_count, num_experts].
// The score matmul runs through the tiled/mv GGML kernels beforehand (full
// GPU occupancy), so this kernel only does the tiny top-k per token.
//
// One SIMDGROUP per token, scores register-resident (8 regs x 32 lanes =
// 256 experts max), winner found by simd_min over packed (desc score,
// asc expert id) keys. No runtime-indexed register arrays: a previous
// one-thread version spilled its best_scores[k] arrays to stack memory and
// burned ~0.17 ms per dispatch on the resulting serial memory round trips.
[[kernel]] void route_select_topk_f32(
    device const float *scores_in [[buffer(0)]],
    device long *route_token_ids [[buffer(1)]],
    device long *route_expert_ids [[buffer(2)]],
    device float *route_weights [[buffer(3)]],
    constant uint &token_count [[buffer(4)]],
    constant uint &num_experts [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &gate_mode [[buffer(7)]],
    device const float *wg_bias [[buffer(8)]],
    constant uint &has_wg_bias [[buffer(9)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    constexpr uint MAX_TOPK = 16;
    constexpr uint REGS = 8; // 8 * 32 lanes = 256 experts max

    if (token >= token_count || k > MAX_TOPK) {
        return;
    }

    // Expert e lives in lane e % 32, register e / 32: on score ties the
    // packed-key min then selects the smallest expert id, matching the
    // ascending-scan strict-'>' insertion of the reference implementation.
    device const float *scores = scores_in + token * num_experts;
    float s[REGS];
    for (uint r = 0; r < REGS; r++) {
        const uint e = r * 32 + lane;
        float v = -INFINITY;
        if (e < num_experts) {
            v = scores[e];
            if (has_wg_bias != 0) {
                v += wg_bias[e];
            }
        }
        s[r] = v;
    }

    float lmax = s[0];
    for (uint r = 1; r < REGS; r++) {
        lmax = max(lmax, s[r]);
    }
    const float max_all = simd_max(lmax);

    float denom_all = 0.0f;
    if (gate_mode == RouteGateSoftmaxAll) {
        float lsum = 0.0f;
        for (uint r = 0; r < REGS; r++) {
            lsum += exp(s[r] - max_all); // exp(-INF - max) == 0 for padding
        }
        denom_all = simd_sum(lsum);
    }

    float s0 = 0.0f;
    float denom_topk = 0.0f;
    for (uint slot = 0; slot < k; slot++) {
        float lbest = s[0];
        uint lbest_r = 0;
        for (uint r = 1; r < REGS; r++) {
            if (s[r] > lbest) {
                lbest = s[r];
                lbest_r = r;
            }
        }
        // Order-preserving uint key: max score via simd_max, then the
        // smallest expert id among the lanes holding that score.
        const uint b = as_type<uint>(lbest);
        const uint mono = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        const uint win_mono = simd_max(mono);
        const uint cand = (mono == win_mono) ? (lbest_r * 32 + lane) : 0xFFFFFFFFu;
        const uint win_e = simd_min(cand);
        const float win_s = as_type<float>(
            (win_mono & 0x80000000u) ? (win_mono ^ 0x80000000u) : ~win_mono);

        if (slot == 0) {
            s0 = win_s;
        }
        denom_topk += exp(win_s - s0);

        if (lane == win_e % 32) {
            const uint win_r = win_e / 32;
            for (uint r = 0; r < REGS; r++) {
                if (r == win_r) {
                    s[r] = -INFINITY;
                }
            }
        }

        if (lane == 0) {
            const uint route = token * k + slot;
            route_token_ids[route] = long(token);
            route_expert_ids[route] = long(win_e);
            if (gate_mode == RouteGateRaw) {
                route_weights[route] = win_s;
            } else if (gate_mode == RouteGateSigmoid) {
                route_weights[route] = 1.0f / (1.0f + exp(-win_s));
            } else if (gate_mode == RouteGateSoftmaxAll) {
                route_weights[route] = exp(win_s - max_all) / denom_all;
            } else {
                // RouteGateSoftmaxTopk: denominator only known after the
                // last slot; store the numerator now, divide below.
                route_weights[route] = exp(win_s - s0);
            }
        }
    }

    if (gate_mode == RouteGateSoftmaxTopk && lane == 0) {
        for (uint slot = 0; slot < k; slot++) {
            route_weights[token * k + slot] /= denom_topk;
        }
    }
}

[[kernel]] void route_topk_f32(
    device const float *x [[buffer(0)]],
    device const float *wg [[buffer(1)]],
    device long *route_token_ids [[buffer(2)]],
    device long *route_expert_ids [[buffer(3)]],
    device float *route_weights [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &d_model [[buffer(6)]],
    constant uint &num_experts [[buffer(7)]],
    constant uint &k [[buffer(8)]],
    constant uint &gate_mode [[buffer(9)]],
    device const float *wg_bias [[buffer(10)]],
    constant uint &has_wg_bias [[buffer(11)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    constexpr uint MAX_TOPK = 16;
    constexpr uint MAX_EXPERTS = 256;
    threadgroup float scores[MAX_EXPERTS];

    if (token >= token_count || k > MAX_TOPK || num_experts > MAX_EXPERTS) {
        return;
    }

    // One simdgroup per expert, lanes striding the d_model dot. The previous
    // one-thread-per-expert layout ran a 2880-element scalar dot per lane and
    // left the GPU >95% idle (~0.28 ms per call at decode).
    const uint simd_lane = lane % 32;
    const uint num_sg = max(tptg / 32, 1u);
    device const float * xrow = x + token * d_model;
    for (uint e = lane / 32; e < num_experts; e += num_sg) {
        float partial = 0.0f;
        device const float * wrow = wg + e * d_model;
        for (uint d = simd_lane; d < d_model; d += 32) {
            partial += xrow[d] * wrow[d];
        }
        float score = simd_sum(partial);
        if (simd_lane == 0) {
            if (has_wg_bias != 0) {
                score += wg_bias[e];
            }
            scores[e] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane != 0) {
        return;
    }

    float best_scores[MAX_TOPK];
    int best_experts[MAX_TOPK];
    for (uint i = 0; i < MAX_TOPK; i++) {
        best_scores[i] = -INFINITY;
        best_experts[i] = -1;
    }

    float max_all = -INFINITY;
    for (uint expert = 0; expert < num_experts; expert++) {
        float score = scores[expert];
        max_all = max(max_all, score);

        for (uint slot = 0; slot < k; slot++) {
            if (score > best_scores[slot]) {
                for (uint move = k - 1; move > slot; move--) {
                    best_scores[move] = best_scores[move - 1];
                    best_experts[move] = best_experts[move - 1];
                }
                best_scores[slot] = score;
                best_experts[slot] = int(expert);
                break;
            }
        }
    }

    float denom = 1.0f;
    if (gate_mode == RouteGateSoftmaxTopk) {
        float max_selected = best_scores[0];
        denom = 0.0f;
        for (uint slot = 0; slot < k; slot++) {
            denom += exp(best_scores[slot] - max_selected);
        }
    } else if (gate_mode == RouteGateSoftmaxAll) {
        denom = 0.0f;
        for (uint expert = 0; expert < num_experts; expert++) {
            float score = scores[expert];
            denom += exp(score - max_all);
        }
    }

    for (uint slot = 0; slot < k; slot++) {
        const uint route = token * k + slot;
        const float score = best_scores[slot];
        route_token_ids[route] = long(token);
        route_expert_ids[route] = long(best_experts[slot]);
        if (gate_mode == RouteGateRaw) {
            route_weights[route] = score;
        } else if (gate_mode == RouteGateSigmoid) {
            route_weights[route] = 1.0f / (1.0f + exp(-score));
        } else if (gate_mode == RouteGateSoftmaxAll) {
            route_weights[route] = exp(score - max_all) / denom;
        } else {
            route_weights[route] = exp(score - best_scores[0]) / denom;
        }
    }
}

[[kernel]] void routed_combine_f32(
    device const float *route_values [[buffer(0)]],
    device const long *route_token_ids [[buffer(1)]],
    device const float *route_weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &route_count [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &d_model [[buffer(6)]],
    constant uint &routes_per_token [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = token_count * d_model;
    if (gid >= total) {
        return;
    }

    const uint token = gid / d_model;
    const uint dim = gid - token * d_model;
    float acc = 0.0f;
    if (routes_per_token != 0) {
        // Token-major routes (route_topk layout: token*k + slot): each output
        // element only touches its own k routes instead of scanning all of
        // them (512x fewer reads at a 512-token prefill chunk with k=4).
        const uint base = token * routes_per_token;
        for (uint slot = 0; slot < routes_per_token; slot++) {
            const uint route = base + slot;
            acc += route_weights[route] * route_values[route * d_model + dim];
        }
    } else {
        for (uint route = 0; route < route_count; route++) {
            if ((uint)route_token_ids[route] == token) {
                acc += route_weights[route] * route_values[route * d_model + dim];
            }
        }
    }
    output[gid] = acc;
}

[[kernel]] void clamped_swiglu_f32(
    device const float *gate_in [[buffer(0)]],
    device const float *up_in [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant float &alpha [[buffer(3)]],
    constant float &limit [[buffer(4)]],
    constant uint &len [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= len) {
        return;
    }

    const float gate = min(gate_in[gid], limit);
    const float up = clamp(up_in[gid], -limit, limit);
    const float glu = gate / (1.0f + exp(-alpha * gate));
    output[gid] = (up + 1.0f) * glu;
}

// q8_0 block layout (must match ggml_mm_mv.metal).
typedef struct {
    half d;
    int8_t qs[32];
} moe_block_q8_0;

// Quantize rows of an f16 buffer into q8_0 blocks: KV-cache shadow
// maintenance for the GPT-OSS fused attention. Grid (heads, rows, blocks
// from b0); one simdgroup per block; elements past `valid` (from row start)
// quantize to zero so gemvs over padded lengths read exact zeros.
[[kernel]] void gpt_oss_kv_quantize_q8_0(
    device const half *src [[buffer(0)]],
    device char *dst [[buffer(1)]],
    constant uint &src_head_stride [[buffer(2)]],
    constant uint &src_row_stride [[buffer(3)]],
    constant uint &dst_head_stride_blocks [[buffer(4)]],
    constant uint &dst_row_stride_blocks [[buffer(5)]],
    constant uint &src_row_offset [[buffer(6)]],
    constant uint &dst_row_offset [[buffer(7)]],
    constant uint &b0 [[buffer(8)]],
    constant uint &valid [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint head = tgpig.x;
    const uint row = tgpig.y;
    const uint block = b0 + tgpig.z;

    device const half *srow =
        src + (uint64_t)head * src_head_stride
            + (uint64_t)(row + src_row_offset) * src_row_stride;
    device moe_block_q8_0 *brow = (device moe_block_q8_0 *)dst
        + (uint64_t)head * dst_head_stride_blocks
        + (uint64_t)(row + dst_row_offset) * dst_row_stride_blocks
        + block;

    const uint ix = block * 32 + lane;
    const float v = ix < valid ? (float)srow[ix] : 0.0f;
    const float amax = simd_max(fabs(v));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) {
        brow->d = (half)d;
    }
    brow->qs[lane] = (int8_t)rint(v * id);
}

// Fused per-route expert bias add: out[route, col] = value[route, col] +
// bias[expert_ids[route], col]. Replaces a gather of a full [routes, n]
// bias matrix plus a separate binary add (two passes and a 20+ MB
// intermediate per MoE matmul at prefill).
[[kernel]] void routed_bias_add_f32(
    device const float *value [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device const long *route_expert_ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &route_count [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = route_count * n;
    if (gid >= total) {
        return;
    }
    const uint route = gid / n;
    const uint col = gid - route * n;
    const uint expert = (uint)route_expert_ids[route];
    out[gid] = value[gid] + bias[expert * n + col];
}

// Sum split-k gemv partials over the chunk axis:
// out[head][i] = sum_c partial[(head*chunks + c)][i], i over m*n.
[[kernel]] void gpt_oss_sum_chunks_f16(
    device const half *partials [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant uint &heads [[buffer(2)]],
    constant uint &chunks [[buffer(3)]],
    constant uint &plane [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = heads * plane;
    if (gid >= total) {
        return;
    }
    const uint head = gid / plane;
    const uint i = gid - head * plane;
    float acc = 0.0f;
    for (uint c = 0; c < chunks; c++) {
        acc += (float)partials[(uint64_t)(head * chunks + c) * plane + i];
    }
    out[gid] = (half)acc;
}

// Fused flash-attention decode for GPT-OSS, two phases sharing K/V reads
// across the GQA group (each key is streamed once per KV head, serving all
// `group` q heads at once).
//
// Phase 1 (part): grid [Hkv, n_chunks, S]; each threadgroup runs an online
// f32 softmax over its chunk of keys for all q heads of its kv head, one
// simdgroup per key slice, and writes per-simdgroup partials (m, l, acc[D])
// to scratch. Phase 2 (merge): one threadgroup per output row combines the
// partials, folds the per-head SINK logit into the denominator, and writes
// the f16 output row. K/V are seq-major capacity buffers; q/out dense
// [Hq, S, D]. Requires D <= 64 and group <= 8.
constant constexpr uint FLASH_MAX_GROUP = 8;
constant constexpr uint FLASH_MAX_DPL = 2; // D <= 64
constant constexpr uint FLASH_SG = 8;      // simdgroups per threadgroup

// GQA group size and D-elements-per-lane specialized at PSO build time so
// every register array indexes with compile-time constants (a runtime bound
// would spill the accumulators to stack memory).
constant uint FC_GROUP [[function_constant(0)]];
constant uint FC_DPL [[function_constant(1)]];
// Single-chunk mode: merge the simdgroup partials in threadgroup memory and
// write the output row directly, skipping the merge dispatch entirely.
constant bool FC_FUSE_MERGE [[function_constant(2)]];

[[kernel]] void gpt_oss_flash_attn_part_f16(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device const float *mask [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant uint &s_len [[buffer(5)]],
    constant uint &t_len [[buffer(6)]],
    constant uint &d [[buffer(7)]],
    constant uint &k_head_stride [[buffer(8)]],
    constant uint &v_head_stride [[buffer(9)]],
    constant uint &v_seq_stride [[buffer(10)]],
    constant uint &chunk [[buffer(11)]],
    constant uint &n_chunks [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    device const float *sinks [[buffer(14)]],
    device half *out [[buffer(15)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    // Lane-per-key: each lane owns one key of a 32-key block, so scores,
    // exps and mask adds all run 32-wide with a single simd_max/simd_sum
    // per block instead of one reduction per key. K rows are seq-major
    // (each lane streams its own row); V is transposed so the AV phase
    // reads each dim's row contiguously along the block.
    const uint kv_head = tgpig.x;
    const uint chunk_ix = tgpig.y;
    const uint qpos = tgpig.z;
    const uint simd_lane = lane % 32;
    const uint simd_ix = lane / 32;

    const uint j_lo = chunk_ix * chunk;
    const uint j_hi = min(t_len, j_lo + chunk);

    device const half *kh = k + (uint64_t)kv_head * k_head_stride;
    device const half *vh = v + (uint64_t)kv_head * v_head_stride;
    device const float *mrow = mask + (uint64_t)qpos * t_len;

    // q for the whole GQA group staged in threadgroup memory: the score
    // loop reads it dim by dim as a broadcast.
    threadgroup float q_tg[FLASH_MAX_GROUP * 64];
    for (uint i = lane; i < FC_GROUP * d; i += FLASH_SG * 32) {
        const uint g = i / d;
        const uint dim = i % d;
        q_tg[g * 64 + dim] =
            (float)q[(uint64_t)((kv_head * FC_GROUP + g) * s_len + qpos) * d + dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float p_tg[FLASH_SG * 32];

    float m[FLASH_MAX_GROUP];
    float l[FLASH_MAX_GROUP];
    float acc[FLASH_MAX_GROUP][FLASH_MAX_DPL];
    for (uint g = 0; g < FC_GROUP; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint c = 0; c < FC_DPL; c++) acc[g][c] = 0.0f;
    }

    for (uint j0 = j_lo + simd_ix * 32; j0 < j_hi; j0 += FLASH_SG * 32) {
        const uint blk = min(32u, j_hi - j0);
        const uint j = j0 + simd_lane;
        const bool live = simd_lane < blk;
        device const half4 *krow4 = (device const half4 *)(kh + (uint64_t)j * d);
        const float mj = live ? mrow[j] : -INFINITY;
        const uint d4 = d / 4;
        for (uint g = 0; g < FC_GROUP; g++) {
            // Scores: each lane dots its own K row against the shared q,
            // vectorized 4-wide to keep the load count down.
            float sc = 0.0f;
            if (live) {
                threadgroup const float4 *q4 =
                    (threadgroup const float4 *)(q_tg + g * 64);
                for (uint dim4 = 0; dim4 < d4; dim4++) {
                    sc += dot(float4(krow4[dim4]), q4[dim4]);
                }
            }
            sc = live ? sc * scale + mj : -INFINITY;
            const float m_new = max(m[g], simd_max(sc));
            const float corr = exp(m[g] - m_new);
            const float p = live ? exp(sc - m_new) : 0.0f;
            l[g] = l[g] * corr + simd_sum(p);
            m[g] = m_new;
            p_tg[simd_ix * 32 + simd_lane] = p;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // AV: lanes switch to dims; each streams its dim's transposed V
            // row contiguously across the block.
            for (uint c = 0; c < FC_DPL; c++) {
                const uint dim = simd_lane + 32 * c;
                float a = 0.0f;
                if (dim < d) {
                    device const half4 *vrow4 = (device const half4 *)(
                        vh + (uint64_t)dim * v_seq_stride + j0);
                    threadgroup const float4 *p4 =
                        (threadgroup const float4 *)(p_tg + simd_ix * 32);
                    const uint b4n = blk / 4;
                    for (uint b4 = 0; b4 < b4n; b4++) {
                        a += dot(float4(vrow4[b4]), p4[b4]);
                    }
                    for (uint b = b4n * 4; b < blk; b++) {
                        a += p_tg[simd_ix * 32 + b] * (float)(
                            (device const half *)vrow4)[b];
                    }
                }
                acc[g][c] = acc[g][c] * corr + a;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (FC_FUSE_MERGE) {
        // Merge the simdgroup partials here and write the output rows: one
        // dispatch per layer, no scratch round trip.
        threadgroup float tg_m[FLASH_SG * FLASH_MAX_GROUP];
        threadgroup float tg_l[FLASH_SG * FLASH_MAX_GROUP];
        threadgroup float tg_acc[FLASH_SG * 64];
        if (simd_lane == 0) {
            for (uint g = 0; g < FC_GROUP; g++) {
                tg_m[simd_ix * FLASH_MAX_GROUP + g] = m[g];
                tg_l[simd_ix * FLASH_MAX_GROUP + g] = l[g];
            }
        }
        for (uint g = 0; g < FC_GROUP; g++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < FC_DPL; c++) {
                const uint ix = simd_lane + 32 * c;
                if (ix < d) tg_acc[simd_ix * 64 + ix] = acc[g][c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_ix == 0) {
                float m_all = -INFINITY;
                for (uint sg = 0; sg < FLASH_SG; sg++) {
                    m_all = max(m_all, tg_m[sg * FLASH_MAX_GROUP + g]);
                }
                const float sink = sinks[kv_head * FC_GROUP + g];
                const float m_fin = max(m_all, sink);
                float l_fin = exp(sink - m_fin);
                float w[FLASH_SG];
                for (uint sg = 0; sg < FLASH_SG; sg++) {
                    const float mp = tg_m[sg * FLASH_MAX_GROUP + g];
                    w[sg] = mp == -INFINITY ? 0.0f : exp(mp - m_fin);
                    l_fin += tg_l[sg * FLASH_MAX_GROUP + g] * w[sg];
                }
                const uint row = (kv_head * FC_GROUP + g) * s_len + qpos;
                device half *orow = out + (uint64_t)row * d;
                for (uint ix = simd_lane; ix < d; ix += 32) {
                    float o = 0.0f;
                    for (uint sg = 0; sg < FLASH_SG; sg++) {
                        o += tg_acc[sg * 64 + ix] * w[sg];
                    }
                    orow[ix] = (half)(o / l_fin);
                }
            }
        }
        return;
    }

    // Per-simdgroup partial: [m, l, acc[d]] per (row, chunk, simdgroup).
    const uint stride = 2 + d;
    for (uint g = 0; g < FC_GROUP; g++) {
        const uint row = (kv_head * FC_GROUP + g) * s_len + qpos;
        device float *part = partials
            + (uint64_t)((row * n_chunks + chunk_ix) * FLASH_SG + simd_ix) * stride;
        if (simd_lane == 0) {
            part[0] = m[g];
            part[1] = l[g];
        }
        for (uint c = 0; c < FC_DPL; c++) {
            const uint ix = simd_lane + 32 * c;
            if (ix < d) part[2 + ix] = acc[g][c];
        }
    }
}

// Phase 2: one threadgroup (single simdgroup) per output row.
[[kernel]] void gpt_oss_flash_attn_merge_f16(
    device const float *partials [[buffer(0)]],
    device const float *sinks [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &s_len [[buffer(3)]],
    constant uint &d [[buffer(4)]],
    constant uint &n_parts [[buffer(5)]],
    constant float &scale_unused [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint stride = 2 + d;
    device const float *parts = partials + (uint64_t)row * n_parts * stride;

    float m_all = -INFINITY;
    for (uint p = 0; p < n_parts; p++) {
        m_all = max(m_all, parts[p * stride]);
    }
    const float sink = sinks[row / s_len];
    const float m_fin = max(m_all, sink);
    float l_fin = exp(sink - m_fin);
    for (uint p = 0; p < n_parts; p++) {
        const float mp = parts[p * stride];
        l_fin += mp == -INFINITY ? 0.0f : parts[p * stride + 1] * exp(mp - m_fin);
    }
    device half *orow = out + (uint64_t)row * d;
    for (uint ix = lane; ix < d; ix += 32) {
        float o = 0.0f;
        for (uint p = 0; p < n_parts; p++) {
            const float mp = parts[p * stride];
            o += mp == -INFINITY ? 0.0f : parts[p * stride + 2 + ix] * exp(mp - m_fin);
        }
        orow[ix] = (half)(o / l_fin);
    }
}

// Row softmax for GPT-OSS attention: probs = softmax over T keys of
// (score*scale + mask[row % s_len]) with a per-head SINK logit participating
// in the denominator only. Rows are [num_q_heads, s_len] flattened; one
// threadgroup per row.
[[kernel]] void gpt_oss_sinks_softmax_f16(
    device const half *scores [[buffer(0)]],
    device const float *mask [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device half *probs [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &t_len [[buffer(5)]],
    constant uint &s_len [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    constant uint &row_stride [[buffer(8)]],
    constant uint &mask_off [[buffer(9)]],
    constant uint &mask_stride [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float partials[32];
    if (row >= rows) {
        return;
    }
    const uint head = row / s_len;
    const uint mrow = row % s_len;
    device const half *srow = scores + (uint64_t)row * row_stride;
    device const float *mrow_p = mask + (uint64_t)mrow * mask_stride + mask_off;
    device half *prow = probs + (uint64_t)row * row_stride;
    const float sink = sinks[head];
    const uint simd_lane = lane % 32;
    const uint simd_ix = lane / 32;
    const uint n_simd = max(tptg / 32, 1u);

    // Pass 1: max of logits (sink included).
    float m = sink;
    for (uint j = lane; j < t_len; j += tptg) {
        m = max(m, (float)srow[j] * scale + mrow_p[j]);
    }
    m = simd_max(m);
    if (simd_lane == 0) partials[simd_ix] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partials[0];

    // Pass 2: denominator (sink seeds it).
    float den = 0.0f;
    for (uint j = lane; j < t_len; j += tptg) {
        den += exp((float)srow[j] * scale + mrow_p[j] - m);
    }
    den = simd_sum(den);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) partials[simd_ix] = den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    den = partials[0] + exp(sink - m);

    // Pass 3: write normalized probabilities (sink column dropped). The
    // padding columns (row_stride > t_len, q8 block alignment) are zeroed so
    // padded-length consumers read exact zeros.
    for (uint j = lane; j < t_len; j += tptg) {
        prow[j] = (half)(exp((float)srow[j] * scale + mrow_p[j] - m) / den);
    }
    for (uint j = t_len + lane; j < row_stride; j += tptg) {
        prow[j] = (half)0.0f;
    }
}
