struct Tensor {
    shape: vec4<u32>;
    strides: vec4<u32>;
};

struct Buffer {
    data: [[stride(4)]] array<f32>; // 4 represents 4 bytes per value
};

[[group(0), binding(0)]]
var<uniform> u_tensor: Tensor;

[[group(0), binding(1)]]
var<storage, read> in: Buffer;

[[group(0), binding(2)]]
var<storage, write> out: Buffer;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let id = global_id.x * u_tensor.strides.x + global_id.y * u_tensor.strides.y + global_id.z * u_tensor.strides.z;
    var w: i32 = 0;
    loop {
        if (u32(w) >= u_tensor.shape.w) {
            break;
        }
        out.data[id + u32(w) * u_tensor.strides.w] = tanh(in.data[id + u32(w) * u_tensor.strides.w]);
        w = w + 1;
    }
}
