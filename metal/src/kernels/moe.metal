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
    uint valid_mask = 0;
    for (uint r = 0; r < REGS; r++) {
        const uint e = r * 32 + lane;
        float v = -INFINITY;
        if (e < num_experts) {
            valid_mask |= 1u << r;
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
        uint best_key = 0;
        uint best_expert = 0xFFFFFFFFu;
        for (uint r = 0; r < REGS; r++) {
            if ((valid_mask & (1u << r)) == 0) {
                continue;
            }
            const uint bits = as_type<uint>(s[r] == 0.0f ? 0.0f : s[r]);
            const uint key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            if (best_expert == 0xFFFFFFFFu || key > best_key) {
                best_key = key;
                best_expert = r * 32 + lane;
            }
        }
        const uint win_mono = simd_max(best_key);
        const uint cand = best_key == win_mono ? best_expert : 0xFFFFFFFFu;
        const uint win_e = simd_min(cand);
        const float win_s = as_type<float>(
            (win_mono & 0x80000000u) ? (win_mono ^ 0x80000000u) : ~win_mono);

        if (slot == 0) {
            s0 = win_s;
        }
        denom_topk += exp(win_s - s0);

        if (lane == win_e % 32) {
            valid_mask &= ~(1u << (win_e / 32));
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
