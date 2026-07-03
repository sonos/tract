#include <metal_stdlib>
using namespace metal;

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
