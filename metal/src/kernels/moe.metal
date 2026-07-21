#include <metal_stdlib>
using namespace metal;

enum RouteGateMode : uint {
    RouteGateSoftmaxTopk = 0,
    RouteGateSoftmaxAll = 1,
    RouteGateSigmoid = 2,
    RouteGateRaw = 3,
};

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
    uint lane [[thread_index_in_threadgroup]])
{
    constexpr uint MAX_TOPK = 16;
    constexpr uint MAX_EXPERTS = 256;
    threadgroup float scores[MAX_EXPERTS];

    if (token >= token_count || k > MAX_TOPK || num_experts > MAX_EXPERTS) {
        return;
    }

    if (lane < num_experts) {
        float score = 0.0f;
        for (uint d = 0; d < d_model; d++) {
            score += x[token * d_model + d] * wg[lane * d_model + d];
        }
        if (has_wg_bias != 0) {
            score += wg_bias[lane];
        }
        scores[lane] = score;
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
    uint gid [[thread_position_in_grid]])
{
    const uint total = token_count * d_model;
    if (gid >= total) {
        return;
    }

    const uint token = gid / d_model;
    const uint dim = gid - token * d_model;
    float acc = 0.0f;
    for (uint route = 0; route < route_count; route++) {
        if ((uint)route_token_ids[route] == token) {
            acc += route_weights[route] * route_values[route * d_model + dim];
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
