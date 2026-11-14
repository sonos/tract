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

// Token-major routes (token * k + slot): each output element sums its own k
// routes.
template <typename T>
[[kernel]] void routed_combine(
    device const float *route_values [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device T *output [[buffer(2)]],
    constant uint &token_count [[buffer(3)]],
    constant uint &d_model [[buffer(4)]],
    constant uint &routes_per_token [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= token_count * d_model) {
        return;
    }
    const uint token = gid / d_model;
    const uint dim = gid - token * d_model;
    const uint base = token * routes_per_token;
    float acc = 0.0f;
    for (uint slot = 0; slot < routes_per_token; slot++) {
        const uint route = base + slot;
        acc += route_weights[route] * route_values[route * d_model + dim];
    }
    output[gid] = T(acc);
}

template [[host_name("routed_combine_f32")]] kernel decltype(routed_combine<float>) routed_combine<float>;
template [[host_name("routed_combine_f16")]] kernel decltype(routed_combine<half>) routed_combine<half>;

inline float routed_activate(float v, uint activation) {
    if (activation == 1) {
        return v / (1.0f + exp(-v));
    } else if (activation == 2) {
        return 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    } else if (activation == 3) {
        return max(v, 0.0f);
    }
    return v;
}

// RoutedLinear postamble over [routes, n] projections: bias at the route's
// expert, then the activation (0 none, 1 silu, 2 tanh gelu, 3 relu), times
// `up` for a pair, or clamped SwiGLU (4) of the pair.
[[kernel]] void routed_postamble_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const float *bias_up [[buffer(3)]],
    device const long *expert_ids [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &routes [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    constant uint &activation [[buffer(8)]],
    constant float &alpha [[buffer(9)]],
    constant float &limit [[buffer(10)]],
    constant uint &flags [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= routes * n) {
        return;
    }
    const uint route = gid / n;
    const uint col = gid - route * n;
    const ulong bias_ix = (ulong)expert_ids[route] * n + col;
    float g = gate[gid] + ((flags & 2) ? bias[bias_ix] : 0.0f);
    if ((flags & 1) == 0) {
        out[gid] = routed_activate(g, activation);
        return;
    }
    float u = up[gid] + ((flags & 4) ? bias_up[bias_ix] : 0.0f);
    if (activation == 4) {
        g = min(g, limit);
        u = clamp(u, -limit, limit);
        out[gid] = (u + 1.0f) * g / (1.0f + exp(-alpha * g));
    } else {
        out[gid] = routed_activate(g, activation) * u;
    }
}
