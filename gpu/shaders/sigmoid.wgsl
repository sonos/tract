struct Buffer {
    data: [[stride(4)]] array<f32>; // 4 represents 4 bytes per value
};

[[group(0), binding(0)]]
var<storage, read> in: Buffer;

[[group(0), binding(1)]]
var<storage, write> out: Buffer;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    out.data[global_id.x] = 1.0 / (1.0 + exp(-1.0 * in.data[global_id.x]));
}
