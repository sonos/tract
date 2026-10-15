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
        const uint b = as_type<uint>(lbest == 0.0f ? 0.0f : lbest);
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
