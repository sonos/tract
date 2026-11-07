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

// Separable gaussian on the reduced frame. Low is radius 5 (sigma 2.5) after a
// half-size shrink. High is radius 12 (sigma 6) after an 8× shrink.
function separableBlur(radius, sigma) {
  const denom = 2 * sigma * sigma;
  return FULLSCREEN + `
struct Blur { step: vec2f, pad: vec2f };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur: Blur;
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  var sum = vec3f(0.0);
  var total = 0.0;
  for (var i = -${radius}; i <= ${radius}; i++) {
    let w = exp(-f32(i * i) / ${denom});
    sum += w * textureSampleLevel(source, linearSampler, in.uv + f32(i) * blur.step, 0.0).rgb;
    total += w;
  }
  return vec4f(sum / max(total, 1e-6), 1.0);
}`;
}
const BLUR_LOW = separableBlur(5, 2.5);
const BLUR_HIGH = separableBlur(12, 6);

// Average the camera frame into the smaller blur target. The window is the
// source pixels covered by one output pixel. The loop allows nine so a size
// that does not divide evenly still covers the whole frame.
const DOWNSAMPLE = FULLSCREEN + `
struct Down { window: vec2f, pad: vec2f };
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<uniform> down: Down;
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  let srcSize = vec2f(textureDimensions(source));
  let centre = in.uv * srcSize;
  let start = centre - down.window * 0.5;
  var sum = vec3f(0.0);
  var total = 0.0;
  for (var y = 0; y < 9; y++) {
    if (f32(y) + 0.5 >= down.window.y) { continue; }
    for (var x = 0; x < 9; x++) {
      if (f32(x) + 0.5 >= down.window.x) { continue; }
      let p = (start + vec2f(f32(x), f32(y)) + 0.5) / srcSize;
      sum += textureSampleLevel(source, linearSampler, p, 0.0).rgb;
      total += 1.0;
    }
  }
  return vec4f(sum / max(total, 1e-6), 1.0);
}`;

// The mask arrives at model resolution. Refinement resamples it on a 5x5
// mask-texel neighbourhood, weighting each tap by how close the frame colour
// there is to the frame colour at the output pixel, so the edge follows the
// image rather than the low-resolution mask grid.
const COMPOSITE = FULLSCREEN + `
struct Params { colour: vec4f, maskTexel: vec2f, mode: u32, refine: u32, motion: u32, haveKey: u32, pad0: u32, pad1: u32, harmon: vec4f };
fn lumaOf(color: vec3f) -> f32 { return dot(color, vec3f(0.299, 0.587, 0.114)); }
// Shift chroma toward the background colour and put the original brightness back.
fn harmonized(color: vec3f) -> vec3f {
  let strength = params.harmon.w;
  if (strength <= 0.0) { return color; }
  let ambient = params.harmon.rgb;
  let fgLuma = lumaOf(color);
  let ambientLuma = lumaOf(ambient);
  var shifted = vec3f(fgLuma) + mix(color - vec3f(fgLuma), ambient - vec3f(ambientLuma), strength);
  return clamp(shifted + vec3f(fgLuma - lumaOf(shifted)), vec3f(0.0), vec3f(1.0));
}
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
  // Blur and a still image both come through blurTex. The bind group points that
  // binding at the blurred frame or at the chosen background.
  if (params.mode == 1u || params.mode == 3u) {
    bg = textureSampleLevel(blurTex, linearSampler, in.uv, 0.0).rgb;
  }
  if (params.mode == 2u) { return vec4f(vec3f(alpha), 1.0); }
  return vec4f(mix(bg, harmonized(fg), alpha), 1.0);
}`;

const MODES={colour:0,blur:1,mask:2,image:3};
const LOW_SCALE=2;
const HIGH_SCALE=8;
// Medium strength from the replacement harmonization pass: chroma only.
const HARMONIZE_STRENGTH=0.14;
const RING=8;
const SAMPLED=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT;
const RING_USAGE=SAMPLED|GPUTextureUsage.COPY_SRC;

function averageColour(source) {
  const sample = new OffscreenCanvas(4, 4);
  const ctx = sample.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(source, 0, 0, 4, 4);
  const data = ctx.getImageData(0, 0, 4, 4).data;
  let red = 0, green = 0, blue = 0;
  for (let i = 0; i < data.length; i += 4) {
    red += data[i];
    green += data[i + 1];
    blue += data[i + 2];
  }
  const scale = 1 / (16 * 255);
  return [red * scale, green * scale, blue * scale];
}

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
    const [composite,blurLow,blurHigh,downsample]=await Promise.all([
      pipeline(COMPOSITE,format),pipeline(BLUR_LOW,'rgba8unorm'),pipeline(BLUR_HIGH,'rgba8unorm'),pipeline(DOWNSAMPLE,'rgba8unorm'),
    ]);
    return new Compositor(device,maskTexture,canvas,context,composite,blurLow,blurHigh,downsample);
  }

  constructor(device,maskTexture,canvas,context,composite,blurLow,blurHigh,downsample) {
    Object.assign(this,{device,maskTexture,context,composite,blurLow,blurHigh,downsample});
    this.width=canvas.width;this.height=canvas.height;
    this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'clamp-to-edge',addressModeV:'clamp-to-edge'});
    this.frame=device.createTexture({size:[this.width,this.height],format:'rgba8unorm',usage:SAMPLED});
    this.downCache=new WeakMap();
    this.reducedCache=new WeakMap();
    this.compositeCache=new WeakMap();
    this.textureSerial=new WeakMap();
    this.nextTextureSerial=1;
    this.params=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.harmonize=false;
    this.imageAmbient=[0.5,0.5,0.5];
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

  setImage(bitmap) {
    this.mode = 'image';
    if (!this.imageCanvas) this.imageCanvas = new OffscreenCanvas(this.width, this.height);
    const ctx = this.imageCanvas.getContext('2d');
    const scale = Math.max(this.width / bitmap.width, this.height / bitmap.height);
    const width = this.width / scale;
    const height = this.height / scale;
    ctx.drawImage(bitmap, (bitmap.width - width) / 2, (bitmap.height - height) / 2, width, height, 0, 0, this.width, this.height);
    if (!this.imageTexture) {
      this.imageTexture = this.device.createTexture({
        size: [this.width, this.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    this.device.queue.copyExternalImageToTexture(
      {source: this.imageCanvas},
      {texture: this.imageTexture},
      [this.width, this.height],
    );
    this.imageAmbient = averageColour(this.imageCanvas);
  }

  setHarmonize(enabled) {
    this.harmonize = Boolean(enabled);
  }

  setBlur(level = 'high') {
    this.mode='blur';
    this.blurLevel = level === 'low' ? 'low' : 'high';
    const scale = this.blurLevel === 'low' ? LOW_SCALE : HIGH_SCALE;
    this.activeBlur = this.blurLevel === 'low' ? this.blurLow : this.blurHigh;
    const [width, height] = this.ensureReduced(scale);
    const write = (buffer, step) => this.device.queue.writeBuffer(buffer, 0, new Float32Array([step[0], step[1], 0, 0]));
    write(this.reducedStep, [1 / width, 0]);
    write(this.reducedDown, [0, 1 / height]);
  }

  // Both levels blur a shrunk frame. The two textures trade places: the shrink
  // fills the first, the horizontal pass the second, the vertical pass the first again.
  ensureReduced(scale) {
    const width = Math.max(1, Math.floor(this.width / scale));
    const height = Math.max(1, Math.floor(this.height / scale));
    if (!this.reducedStep) {
      const uniform = () => this.device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      this.reducedStep = uniform();
      this.reducedDown = uniform();
      this.downWindow = uniform();
    }
    if (!this.reduced || this.reducedScale !== scale) {
      this.reduced?.forEach(texture => texture.destroy());
      const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
      this.reduced = [0, 1].map(() => this.device.createTexture({size: [width, height], format: 'rgba8unorm', usage}));
      this.reducedCache = new WeakMap();
      this.reducedScale = scale;
    }
    this.device.queue.writeBuffer(this.downWindow, 0, new Float32Array([this.width / width, this.height / height, 0, 0]));
    return [width, height];
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

  downFrom(source) {
    let cache = this.downCache.get(source);
    if (!cache) {
      cache = new Map();
      this.downCache.set(source, cache);
    }
    return this.cached(cache, 'down', () => this.device.createBindGroup({layout: this.downsample.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler}, {binding: 1, resource: source.createView()}, {binding: 2, resource: {buffer: this.downWindow}}]}));
  }

  reducedFrom(texture, axis) {
    let cache = this.reducedCache.get(texture);
    if (!cache) {
      cache = new Map();
      this.reducedCache.set(texture, cache);
    }
    const buffer = axis === 'h' ? this.reducedStep : this.reducedDown;
    return this.cached(cache, axis, () => this.device.createBindGroup({layout: this.activeBlur.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler}, {binding: 1, resource: texture.createView()}, {binding: 2, resource: {buffer}}]}));
  }

  compositeFrom(source, flow, currentLuma, keyLuma, keyFrame, blurTexture) {
    let cache = this.compositeCache.get(source);
    if (!cache) {
      cache = new Map();
      this.compositeCache.set(source, cache);
    }
    const background = this.mode === 'image' && this.imageTexture ? this.imageTexture : blurTexture;
    const key = `${this.textureId(background)}:${this.textureId(flow)}:${this.textureId(currentLuma)}:${this.textureId(keyLuma)}:${this.textureId(keyFrame)}`;
    return this.cached(cache, key, () => this.device.createBindGroup({layout: this.composite.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler}, {binding: 1, resource: source.createView()}, {binding: 2, resource: this.maskTexture.createView()},
      {binding: 3, resource: background.createView()}, {binding: 4, resource: {buffer: this.params}}, {binding: 5, resource: flow.createView()},
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
    const params=new ArrayBuffer(64);
    new Float32Array(params,0,6).set([...this.colour,1,...maskTexel]);
    new Uint32Array(params,24,6).set([MODES[view==='mask'?'mask':this.mode],refine?1:0,warped?1:0,keyFrame?1:0,0,0]);
    const replacement = this.harmonize && view !== 'mask' && (this.mode === 'colour' || this.mode === 'image');
    const ambient = this.mode === 'colour' ? this.colour : this.imageAmbient;
    new Float32Array(params,48,4).set([...ambient, replacement ? HARMONIZE_STRENGTH : 0]);
    this.device.queue.writeBuffer(this.params,0,params);
    const flow = warped ? this.motion.flowTexture : this.noFlow;
    const currentLuma = keyLuma ? this.motion.currentLuma : this.noFlow;
    const boundKey = keyLuma ?? this.noFlow;
    const boundFrame = keyFrame ?? source;
    const pass=(attachment,pipeline,bind)=>{
      const render=encoder.beginRenderPass({colorAttachments:[{view:attachment,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
      render.setPipeline(pipeline);render.setBindGroup(0,bind);render.draw(3);render.end();
    };
    let blurTexture = this.noFlow;
    if(this.mode==='blur'&&view!=='mask') {
      const [shrunk, horizontal] = this.reduced;
      pass(shrunk.createView(), this.downsample, this.downFrom(source));
      pass(horizontal.createView(), this.activeBlur, this.reducedFrom(shrunk, 'h'));
      pass(shrunk.createView(), this.activeBlur, this.reducedFrom(horizontal, 'v'));
      blurTexture = shrunk;
    }
    pass(this.context.getCurrentTexture().createView(),this.composite,this.compositeFrom(source, flow, currentLuma, boundKey, boundFrame, blurTexture));
    this.device.queue.submit([encoder.finish()]);
    return {warped, held, synced, presented: presentationKey(picture)};
  }
}
