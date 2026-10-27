// Background replacement on the inference device: the camera frame, tract's
// mask texture and the background are blended in one render pass per frame.
const FULLSCREEN = `
struct VsOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: VsOut;
  out.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}`;

const BLUR = FULLSCREEN + `
struct Blur { step: vec2f, pad: vec2f };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur: Blur;
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  var sum = vec3f(0.0);
  var total = 0.0;
  for (var i = -6; i <= 6; i++) {
    let w = exp(-f32(i * i) / 18.0);
    sum += w * textureSampleLevel(source, linearSampler, in.uv + f32(i) * blur.step, 0.0).rgb;
    total += w;
  }
  return vec4f(sum / total, 1.0);
}`;

// The mask arrives at model resolution. Refinement resamples it on a 5x5
// mask-texel neighbourhood, weighting each tap by how close the frame colour
// there is to the frame colour at the output pixel, so the edge follows the
// image rather than the low-resolution mask grid.
const COMPOSITE = FULLSCREEN + `
struct Params { colour: vec4f, maskTexel: vec2f, mode: u32, refine: u32 };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var maskTex: texture_2d<f32>;
@group(0) @binding(3) var blurTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn refinedAlpha(uv: vec2f, centre: vec3f) -> f32 {
  var sum = 0.0;
  var total = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let offset = vec2f(f32(x), f32(y));
      let p = uv + offset * params.maskTexel;
      let d = textureSampleLevel(frameTex, linearSampler, p, 0.0).rgb - centre;
      let w = exp(-dot(offset, offset) / 4.5 - dot(d, d) / 0.02);
      sum += w * textureSampleLevel(maskTex, linearSampler, p, 0.0).r;
      total += w;
    }
  }
  return smoothstep(0.15, 0.85, sum / total);
}
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  let fg = textureSampleLevel(frameTex, linearSampler, in.uv, 0.0).rgb;
  var alpha = textureSampleLevel(maskTex, linearSampler, in.uv, 0.0).r;
  if (params.refine != 0u) { alpha = refinedAlpha(in.uv, fg); }
  var bg = params.colour.rgb;
  if (params.mode == 1u) {
    bg = textureSampleLevel(blurTex, linearSampler, in.uv, 0.0).rgb;
  }
  return vec4f(mix(bg, fg, alpha), 1.0);
}`;

const MODES={colour:0,blur:1};
const SAMPLED=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT;

export class Compositor {
  static async create(device,maskTexture,canvas) {
    const format=navigator.gpu.getPreferredCanvasFormat();
    const context=canvas.getContext('webgpu');
    if(!context) throw new Error('This browser cannot draw WebGPU output to an OffscreenCanvas.');
    context.configure({device,format,alphaMode:'opaque'});
    const pipeline=async(code,target)=>{
      const module=device.createShaderModule({code});
      return device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format:target}]},primitive:{topology:'triangle-list'}});
    };
    const [composite,blur]=await Promise.all([pipeline(COMPOSITE,format),pipeline(BLUR,'rgba8unorm')]);
    return new Compositor(device,maskTexture,canvas,context,composite,blur);
  }

  constructor(device,maskTexture,canvas,context,composite,blur) {
    Object.assign(this,{device,maskTexture,context,composite,blur});
    this.width=canvas.width;this.height=canvas.height;
    this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'clamp-to-edge',addressModeV:'clamp-to-edge'});
    this.frame=device.createTexture({size:[this.width,this.height],format:'rgba8unorm',usage:SAMPLED});
    const blurSize=[Math.max(1,this.width>>2),Math.max(1,this.height>>2)];
    this.blurred=[0,1].map(()=>device.createTexture({size:blurSize,format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT}));
    const blurUniform=step=>{const buffer=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,new Float32Array([...step,0,0]));return buffer;};
    const blurGroup=(source,step)=>device.createBindGroup({layout:blur.getBindGroupLayout(0),entries:[
      {binding:0,resource:this.sampler},{binding:1,resource:source.createView()},{binding:2,resource:{buffer:blurUniform(step)}}]});
    this.blurGroups=[blurGroup(this.frame,[1/blurSize[0],0]),blurGroup(this.blurred[0],[0,1/blurSize[1]])];
    this.params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.mode='blur';this.colour=[0.27,0.48,0.46];
    this.bind();
  }

  bind() {
    this.group=this.device.createBindGroup({layout:this.composite.getBindGroupLayout(0),entries:[
      {binding:0,resource:this.sampler},{binding:1,resource:this.frame.createView()},{binding:2,resource:this.maskTexture.createView()},
      {binding:3,resource:this.blurred[1].createView()},{binding:4,resource:{buffer:this.params}}]});
  }

  setColour(hex) {
    const value=parseInt(hex.slice(1),16);
    this.colour=[(value>>16&255)/255,(value>>8&255)/255,(value&255)/255];
    this.mode='colour';
  }

  setBlur() {this.mode='blur';}

  uploadFrame(source) {
    this.device.queue.copyExternalImageToTexture({source},{texture:this.frame},[this.width,this.height]);
  }

  draw(refine) {
    const maskTexel=[1/this.maskTexture.width,1/this.maskTexture.height];
    const params=new ArrayBuffer(32);
    new Float32Array(params,0,6).set([...this.colour,1,...maskTexel]);
    new Uint32Array(params,24,2).set([MODES[this.mode],refine?1:0]);
    this.device.queue.writeBuffer(this.params,0,params);
    const encoder=this.device.createCommandEncoder({label:'pixel-lab composite'});
    const pass=(view,pipeline,group)=>{
      const render=encoder.beginRenderPass({colorAttachments:[{view,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
      render.setPipeline(pipeline);render.setBindGroup(0,group);render.draw(3);render.end();
    };
    if(this.mode==='blur') {
      pass(this.blurred[0].createView(),this.blur,this.blurGroups[0]);
      pass(this.blurred[1].createView(),this.blur,this.blurGroups[1]);
    }
    pass(this.context.getCurrentTexture().createView(),this.composite,this.group);
    this.device.queue.submit([encoder.finish()]);
  }
}
