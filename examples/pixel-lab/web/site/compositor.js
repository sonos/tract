// Background replacement on the compositor device: the camera frame, tract's
// mask texture and the background are blended in one render pass per frame.
// With motion on, the mask is sampled at x + flow(x), where flow runs from the
// displayed frame back to the frame the mask was inferred on (motion.js).
// A poor match does not move that pixel. Copied mask whose surface no longer
// matches the live frame is dropped, so a closed hand does not keep its outline.
const stamp = new URL(import.meta.url).searchParams.get('stamp') || Date.now();
const {MotionWarp} = await import(`./motion.js?stamp=${stamp}`);
const {presentationKey, ringSlot, shownPicture} = await import(`./pairing.mjs?stamp=${stamp}`);
const FULLSCREEN = `
struct VsOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: VsOut;
  out.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}`;

// Full-resolution separable gaussian. Low is a 5 px radius (sigma 2.5). High is 12 px (sigma 6).
// The loop covers the wider one; taps past the chosen radius are skipped.
const BLUR = FULLSCREEN + `
struct Blur { step: vec2f, radius: f32, sigma: f32 };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur: Blur;
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  var sum = vec3f(0.0);
  var total = 0.0;
  let spread = max(blur.sigma, 1.0);
  for (var i = -12; i <= 12; i++) {
    if (f32(abs(i)) > blur.radius) { continue; }
    let w = exp(-f32(i * i) / (2.0 * spread * spread));
    sum += w * textureSampleLevel(source, linearSampler, in.uv + f32(i) * blur.step, 0.0).rgb;
    total += w;
  }
  return vec4f(sum / max(total, 1e-6), 1.0);
}`;

// The mask arrives at model resolution. Refinement resamples it on a 5x5
// mask-texel neighbourhood, weighting each tap by how close the frame colour
// there is to the frame colour at the output pixel, so the edge follows the
// image rather than the low-resolution mask grid.
const COMPOSITE = FULLSCREEN + `
struct Params { colour: vec4f, maskTexel: vec2f, mode: u32, refine: u32, motion: u32, haveKey: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var maskTex: texture_2d<f32>;
@group(0) @binding(3) var blurTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var flowTex: texture_2d<f32>;
@group(0) @binding(6) var currentLuma: texture_2d<f32>;
@group(0) @binding(7) var keyLuma: texture_2d<f32>;
@group(0) @binding(8) var keyFrame: texture_2d<f32>;
fn maskShift(uv: vec2f) -> vec2f {
  if (params.motion == 0u) { return vec2f(0.0); }
  // Move the whole measured step, or none of it. Scaling the step by a middling
  // confidence leaves the mask a fraction behind the person. 0.35 still drops a bad
  // match, and it keeps a fast hand whose blur would have fallen under 0.5.
  let flow = textureSampleLevel(flowTex, linearSampler, uv, 0.0);
  return select(vec2f(0.0), flow.xy / vec2f(textureDimensions(flowTex)), flow.z >= 0.35);
}
// 1 when this pixel still looks like the pixel the mask was copied from, else 0.
// A partial fade would mix in the flat background colour, which is the fringe and the
// leftover hand. The cut is wide enough for a motion-blurred edge and still short of
// skin sitting on a different surface.
fn localAgrees(uv: vec2f, shift: vec2f) -> f32 {
  if (params.motion == 0u) { return 1.0; }
  if (params.haveKey == 1u) {
    let live = textureSampleLevel(frameTex, linearSampler, uv, 0.0).rgb;
    let source = textureSampleLevel(keyFrame, linearSampler, uv + shift, 0.0).rgb;
    return select(0.0, 1.0, distance(live, source) < 0.30);
  }
  let difference = abs(textureSampleLevel(currentLuma, linearSampler, uv, 0.0).r
    - textureSampleLevel(keyLuma, linearSampler, uv + shift, 0.0).r);
  return select(0.0, 1.0, difference < 0.22);
}
// The colour test is only for the silhouette. Interior skin, eyes and mouth stay.
// A leftover finger is thin, so a tap 8 or 16 pixels away lands on the background.
fn nearEdge(uv: vec2f) -> bool {
  let texel = 1.0 / vec2f(textureDimensions(frameTex));
  for (var ring = 1; ring <= 2; ring++) {
    let radius = f32(ring) * 8.0;
    for (var i = 0; i < 8; i++) {
      let angle = f32(i) * 0.78539816;
      let p = uv + vec2f(cos(angle), sin(angle)) * radius * texel;
      if (textureSampleLevel(maskTex, linearSampler, p + maskShift(p), 0.0).r < 0.5) { return true; }
    }
  }
  return false;
}
fn regionAgrees(uv: vec2f, shift: vec2f) -> f32 {
  if (params.motion == 0u || !nearEdge(uv)) { return 1.0; }
  return localAgrees(uv, shift);
}
fn maskAt(uv: vec2f, shift: vec2f) -> f32 {
  return textureSampleLevel(maskTex, linearSampler, uv + shift, 0.0).r;
}
fn refinedAlpha(uv: vec2f, centre: vec3f, shift: vec2f) -> f32 {
  var sum = 0.0;
  var total = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let offset = vec2f(f32(x), f32(y));
      let p = uv + offset * params.maskTexel;
      let d = textureSampleLevel(frameTex, linearSampler, p, 0.0).rgb - centre;
      let w = exp(-dot(offset, offset) / 4.5 - dot(d, d) / 0.02);
      sum += w * maskAt(p, shift);
      total += w;
    }
  }
  return smoothstep(0.15, 0.85, sum / total);
}
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  let fg = textureSampleLevel(frameTex, linearSampler, in.uv, 0.0).rgb;
  let shift = maskShift(in.uv);
  // The mask before the match test. The crack fill below runs only where this
  // coverage existed, so the background does not pay for it and the silhouette
  // does not grow outward.
  let copied = textureSampleLevel(maskTex, linearSampler, in.uv + shift, 0.0).r;
  var alpha = copied;
  if (params.refine != 0u) { alpha = refinedAlpha(in.uv, fg, shift); }
  let agree = regionAgrees(in.uv, shift);
  alpha *= agree;
  // A one-pixel crack would show the flat colour as a line. Fill it only inside a
  // patch that matched, so a removed ghost is not painted back.
  if (params.motion == 1u && agree == 1.0 && alpha == 0.0 && copied > 0.5) {
    let texel = 1.0 / vec2f(textureDimensions(frameTex));
    var kept = 0;
    var nearby = 0.0;
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        if (x == 0 && y == 0) { continue; }
        let p = in.uv + vec2f(f32(x), f32(y)) * texel;
        let a = maskAt(p, maskShift(p));
        if (a > 0.5) { kept += 1; nearby = max(nearby, a); }
      }
    }
    if (kept >= 5) { alpha = nearby; }
  }
  var bg = params.colour.rgb;
  if (params.mode == 1u) {
    bg = textureSampleLevel(blurTex, linearSampler, in.uv, 0.0).rgb;
  }
  if (params.mode == 2u) { return vec4f(vec3f(alpha), 1.0); }
  return vec4f(mix(bg, fg, alpha), 1.0);
}`;

const MODES={colour:0,blur:1,mask:2};
// Radius in camera pixels, then sigma. High is the wider of the two.
const BLUR_LEVELS={low:[5,2.5],high:[12,6]};
const RING=8;
const SAMPLED=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT;
const RING_USAGE=SAMPLED|GPUTextureUsage.COPY_SRC;

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
    this.blurred=[0,1].map(()=>device.createTexture({size:[this.width,this.height],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT}));
    const blurUniform=()=>device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.blurStep=blurUniform();
    this.blurDown=blurUniform();
    this.vertical=device.createBindGroup({layout:blur.getBindGroupLayout(0),entries:[
      {binding:0,resource:this.sampler},{binding:1,resource:this.blurred[0].createView()},{binding:2,resource:{buffer:this.blurDown}}]});
    this.blurCache=new WeakMap();
    this.compositeCache=new WeakMap();
    this.textureSerial=new WeakMap();
    this.nextTextureSerial=1;
    this.params=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.noFlow=device.createTexture({size:[1,1],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    device.queue.writeTexture({texture:this.noFlow},new Uint8Array(4),{bytesPerRow:4},[1,1]);
    this.colour=[0.27,0.48,0.46];
    this.setBlur('high');
    this.motion=null;
    this.maskFrameId=-1;
    this.inferenceFrameId=-1;
    this.heldFrameId=-1;
    this.heldSource=null;
    // Recent camera frames, plus the frame on screen and the frame inference is
    // running on. Those two are not recycled, so a slow mask still has its picture.
    this.ring = Array.from({length: RING}, () => ({frameId: -1, texture: this.device.createTexture({size: [this.width, this.height], format: 'rgba8unorm', usage: RING_USAGE})}));
  }

  async ensureMotion() {
    if (!this.motion) this.motion = await MotionWarp.create(this.device);
  }

  ringTexture(frameId) {
    return this.ring?.find(slot => slot.frameId === frameId)?.texture;
  }

  setColour(hex) {
    const value=parseInt(hex.slice(1),16);
    this.colour=[(value>>16&255)/255,(value>>8&255)/255,(value&255)/255];
    this.mode='colour';
  }

  setBlur(level = 'high') {
    this.mode='blur';
    const spec = BLUR_LEVELS[level] || BLUR_LEVELS.high;
    const write = (buffer, step) => this.device.queue.writeBuffer(buffer, 0, new Float32Array([...step, ...spec]));
    write(this.blurStep, [1 / this.width, 0]);
    write(this.blurDown, [0, 1 / this.height]);
  }

  // The frame inference has just started on. uploadFrame will not recycle it.
  setInferenceFrame(frameId) {
    this.inferenceFrameId = frameId;
  }

  uploadFrame(source, frameId = 0) {
    const keep = new Set([this.maskFrameId, this.inferenceFrameId].filter(id => id >= 0 && id !== frameId));
    const index = ringSlot(this.ring.map(slot => slot.frameId), frameId, keep);
    if (index < 0) return;
    const slot = this.ring[index];
    slot.frameId = frameId;
    this.device.queue.copyExternalImageToTexture({source}, {texture: slot.texture}, [this.width, this.height]);
  }

  // The frame an inference has just started on. Its pyramid is what later frames warp from.
  captureKey(frameId) {
    this.motion?.captureKey(frameId, this.ringTexture(frameId));
  }

  setMaskFrame(frameId) {
    this.maskFrameId = frameId;
    this.motion?.setMaskFrame(frameId);
  }

  textureId(texture) {
    let id = this.textureSerial.get(texture);
    if (id === undefined) this.textureSerial.set(texture, id = this.nextTextureSerial++);
    return id;
  }

  cached(map, key, make) {
    let value = map.get(key);
    if (!value) {
      value = make();
      map.set(key, value);
    }
    return value;
  }

  blurFrom(source) {
    let cache = this.blurCache.get(source);
    if (!cache) {
      cache = new Map();
      this.blurCache.set(source, cache);
    }
    return this.cached(cache, 'h', () => this.device.createBindGroup({layout: this.blur.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler}, {binding: 1, resource: source.createView()}, {binding: 2, resource: {buffer: this.blurStep}}]}));
  }

  compositeFrom(source, flow, currentLuma, keyLuma, keyFrame) {
    let cache = this.compositeCache.get(source);
    if (!cache) {
      cache = new Map();
      this.compositeCache.set(source, cache);
    }
    const key = `${this.textureId(flow)}:${this.textureId(currentLuma)}:${this.textureId(keyLuma)}:${this.textureId(keyFrame)}`;
    return this.cached(cache, key, () => this.device.createBindGroup({layout: this.composite.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler}, {binding: 1, resource: source.createView()}, {binding: 2, resource: this.maskTexture.createView()},
      {binding: 3, resource: this.blurred[1].createView()}, {binding: 4, resource: {buffer: this.params}}, {binding: 5, resource: flow.createView()},
      {binding: 6, resource: currentLuma.createView()}, {binding: 7, resource: keyLuma.createView()},
      {binding: 8, resource: keyFrame.createView()}]}));
  }

  draw(refine, view, {frameId = -1, motion = false} = {}) {
    const maskTexel=[1/this.maskTexture.width,1/this.maskTexture.height];
    const encoder=this.device.createCommandEncoder({label:'pixel-lab composite'});
    const live = this.ringTexture(frameId);
    const matched = this.maskFrameId >= 0 ? this.ringTexture(this.maskFrameId) : null;
    // Warp only when it can take the mask from its own frame onto this newer one.
    // Otherwise show the camera frame the mask was measured on. The newest frame
    // with an older mask is the lag, so a missing match repeats the last pair.
    const warped = Boolean(motion && live && this.motion?.encode(encoder, frameId, live));
    const picture = shownPicture({
      warped, frameId, maskFrameId: this.maskFrameId, matched: Boolean(matched), heldFrameId: this.heldFrameId,
    });
    const source = picture.live ? live : (matched && picture.id === this.maskFrameId ? matched : this.heldSource);
    if (!source) throw new Error('The camera frame for this composite is gone.');
    if (!picture.live && matched) {
      this.heldSource = matched;
      this.heldFrameId = this.maskFrameId;
    }
    const held = !picture.live && picture.id !== frameId;
    const synced = !warped && !held && this.maskFrameId === frameId;
    const keyLuma = warped ? this.motion.keyLuma(this.motion.maskFrameId) : null;
    const keyFrame = warped ? this.ringTexture(this.motion.maskFrameId) : null;
    const params=new ArrayBuffer(48);
    new Float32Array(params,0,6).set([...this.colour,1,...maskTexel]);
    new Uint32Array(params,24,6).set([MODES[view==='mask'?'mask':this.mode],refine?1:0,warped?1:0,keyFrame?1:0,0,0]);
    this.device.queue.writeBuffer(this.params,0,params);
    const flow = warped ? this.motion.flowTexture : this.noFlow;
    const currentLuma = keyLuma ? this.motion.currentLuma : this.noFlow;
    const boundKey = keyLuma ?? this.noFlow;
    const boundFrame = keyFrame ?? source;
    const pass=(attachment,pipeline,bind)=>{
      const render=encoder.beginRenderPass({colorAttachments:[{view:attachment,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
      render.setPipeline(pipeline);render.setBindGroup(0,bind);render.draw(3);render.end();
    };
    if(this.mode==='blur'&&view!=='mask') {
      pass(this.blurred[0].createView(),this.blur,this.blurFrom(source));
      pass(this.blurred[1].createView(),this.blur,this.vertical);
    }
    pass(this.context.getCurrentTexture().createView(),this.composite,this.compositeFrom(source, flow, currentLuma, boundKey, boundFrame));
    this.device.queue.submit([encoder.finish()]);
    return {warped, held, synced, presented: presentationKey(picture)};
  }
}
