//! On wasm, each `set_pipeline` / `set_bind_group` / `dispatch_workgroups` is a
//! wasm-bindgen import. A compute pass is packed into one JS call instead.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
let installed = false;
const pipes = new Map();
const bgs = new Map();
export function tract_install_record_hooks() {
  if (installed) return;
  installed = true;
  const origCP = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function(desc) {
    const p = origCP.call(this, desc);
    if (desc && desc.label) pipes.set(desc.label, p);
    return p;
  };
  const origBG = GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup = function(desc) {
    const g = origBG.call(this, desc);
    if (desc && desc.label) bgs.set(desc.label, g);
    return g;
  };
  const origBegin = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function(desc) {
    const pass = origBegin.call(this, desc);
    globalThis.__tractPass = pass;
    return pass;
  };
}
export function tract_record_pass(data) {
  const pass = globalThis.__tractPass;
  if (!pass) throw new Error("tract-wgpu: no compute pass");
  const u32 = new Uint32Array(data.buffer, data.byteOffset, data.byteLength >> 2);
  let last = null;
  for (let i = 0; i < u32.length; i += 6) {
    const pl = pipes.get("tp" + u32[i]);
    const bg = bgs.get("tb" + u32[i + 1]);
    if (!pl) throw new Error("tract-wgpu: missing pipeline tp" + u32[i]);
    if (!bg) throw new Error("tract-wgpu: missing bind group tb" + u32[i + 1]);
    if (pl !== last) { pass.setPipeline(pl); last = pl; }
    pass.setBindGroup(0, bg, [u32[i + 2]]);
    pass.dispatchWorkgroups(u32[i + 3], u32[i + 4], u32[i + 5]);
  }
}
"#)]
extern "C" {
    pub fn tract_install_record_hooks();
    pub fn tract_record_pass(data: &[u8]);
}
