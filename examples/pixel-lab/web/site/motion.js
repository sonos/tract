// GPU mask motion for a segmenter that is behind the camera.
//
// The mask was inferred on frame k. Frame t is on screen. A luma pyramid of frame k is kept
// from the moment that inference starts. The compositor reads the mask at x + flow(x).
// Flow stays on this GPU. Nothing is read back, and a missing key frame holds the mask in place.
//
// The motion is one similarity for the whole person, fit to pixels that have texture in two
// directions (eyes, hairline, shoulders, fingers). Bare skin inherits that motion, so the face
// cannot tear. A coarse grid may depart from the similarity only where the texture agrees, and
// a few smoothness passes keep a hand attached to the wrist. Level-zero flow is in pixels of
// the 320-wide pyramid; divide by that texture's size to get the normalized UV shift.

const LUMA = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
fn centre(id: vec3u) -> vec2f { return (vec2f(id.xy) + 0.5) / vec2f(textureDimensions(dst)); }
@compute @workgroup_size(8, 8) fn luma(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  let c = textureSampleLevel(src, samp, centre(id), 0.0).rgb;
  textureStore(dst, id.xy, vec4f(dot(c, vec3f(0.299, 0.587, 0.114)), 0.0, 0.0, 1.0));
}
@compute @workgroup_size(8, 8) fn down(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  textureStore(dst, id.xy, vec4f(textureSampleLevel(src, samp, centre(id), 0.0).r, 0.0, 0.0, 1.0));
}`;

const LK = `
struct Step { scale: f32, lambda: f32, maxStep: f32, pad: f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tmpl: texture_2d<f32>;
@group(0) @binding(2) var key: texture_2d<f32>;
@group(0) @binding(3) var flowIn: texture_2d<f32>;
@group(0) @binding(4) var flowOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> step: Step;
fn at(q: vec2i, size: vec2i) -> f32 { return textureLoad(tmpl, clamp(q, vec2i(0), size - 1), 0).r; }
@compute @workgroup_size(8, 8) fn lk(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(flowOut));
  let pix = vec2i(id.xy);
  if (any(pix >= size)) { return; }
  let fsize = vec2f(size);
  let d = textureSampleLevel(flowIn, samp, (vec2f(pix) + 0.5) / fsize, 0.0).xy * step.scale;
  var h = vec3f(0.0);
  var b = vec2f(0.0);
  var err = 0.0;
  for (var y = -3; y <= 3; y++) {
    for (var x = -3; x <= 3; x++) {
      let q = pix + vec2i(x, y);
      let g = 0.5 * vec2f(at(q + vec2i(1, 0), size) - at(q - vec2i(1, 0), size),
                          at(q + vec2i(0, 1), size) - at(q - vec2i(0, 1), size));
      let e = at(q, size) - textureSampleLevel(key, samp, (vec2f(q) + 0.5 + d) / fsize, 0.0).r;
      h += vec3f(g.x * g.x, g.x * g.y, g.y * g.y);
      b += g * e;
      err += abs(e);
    }
  }
  // Smallest eigenvalue of the gradient outer product, before the damper. Flat skin is near 0.
  let det = h.x * h.z - h.y * h.y;
  let trace = h.x + h.z;
  let minEig = 0.5 * trace - sqrt(max(0.25 * trace * trace - det, 0.0));
  h.x += step.lambda;
  h.z += step.lambda;
  var delta = vec2f(h.z * b.x - h.y * b.y, h.x * b.y - h.y * b.x) / (h.x * h.z - h.y * h.y);
  let len = length(delta);
  if (len > step.maxStep) { delta *= step.maxStep / len; }
  // Mean absolute luma error in the window. Under 0.06 the shift is trusted; over 0.22 it is
  // dropped. A fast hand is motion-blurred and sits between those, so it still has to move.
  let confidence = 1.0 - smoothstep(0.06, 0.22, err / 49.0);
  // No texture in both directions: do not invent a step. A straight edge and bare skin keep
  // the coarser motion when the colours still match, which is what stops the face tearing.
  if (minEig < 0.02) { delta = vec2f(0.0); }
  // .w is the corner strength. The body fit uses it and ignores flat skin and straight edges.
  textureStore(flowOut, id.xy, vec4f(d + delta, confidence, minEig));
}`;

const SIM = `
struct Sim { a: f32, b: f32, tx: f32, ty: f32, confidence: f32, pad0: f32, pad1: f32, pad2: f32 };
struct Grid { size: vec2f, pad: vec2f };
fn body(sim: Sim, p: vec2f) -> vec2f {
  let k = vec2f(sim.a * p.x - sim.b * p.y + sim.tx, sim.b * p.x + sim.a * p.y + sim.ty);
  return k - p;
}`;

// Previous similarity, expressed in the coarse grid, becomes the search start.
const SEED = SIM + `
@group(0) @binding(0) var<storage, read> sim: Sim;
@group(0) @binding(1) var<uniform> grid: Grid;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let coarse = vec2f(textureDimensions(dst));
  if (any(id.xy >= vec2u(textureDimensions(dst)))) { return; }
  let p = (vec2f(id.xy) + 0.5) / coarse * grid.size;
  let flow = body(sim, p) * coarse / grid.size;
  textureStore(dst, id.xy, vec4f(flow, 0.0, 1.0));
}`;

// Weighted similarity from textured grid cells. Maps the current cell to the key cell.
const FIT = `
struct Sim { a: f32, b: f32, tx: f32, ty: f32, confidence: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var measured: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> sim: Sim;
var<workgroup> part: array<f32, 768>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3u) {
  let size = vec2i(textureDimensions(measured));
  let n = size.x * size.y;
  var sw = 0.0; var sx = 0.0; var sy = 0.0; var su = 0.0; var sv = 0.0;
  var sxx = 0.0; var syy = 0.0; var sxu = 0.0; var syv = 0.0; var sxv = 0.0; var syu = 0.0;
  for (var i = i32(lid.x); i < n; i = i + 64) {
    let pix = vec2i(i % size.x, i / size.x);
    let s = textureLoad(measured, pix, 0);
    let w = select(0.0, s.w, s.z >= 0.35 && s.w >= 0.02);
    if (w == 0.0) { continue; }
    let c = vec2f(pix) + 0.5;
    let k = c + s.xy;
    sw += w; sx += w * c.x; sy += w * c.y; su += w * k.x; sv += w * k.y;
    sxx += w * c.x * c.x; syy += w * c.y * c.y;
    sxu += w * c.x * k.x; syv += w * c.y * k.y; sxv += w * c.x * k.y; syu += w * c.y * k.x;
  }
  let base = lid.x * 12u;
  part[base] = sw; part[base + 1u] = sx; part[base + 2u] = sy; part[base + 3u] = su;
  part[base + 4u] = sv; part[base + 5u] = sxx; part[base + 6u] = syy; part[base + 7u] = sxu;
  part[base + 8u] = syv; part[base + 9u] = sxv; part[base + 10u] = syu;
  workgroupBarrier();
  if (lid.x != 0u) { return; }
  sw = 0.0; sx = 0.0; sy = 0.0; su = 0.0; sv = 0.0;
  sxx = 0.0; syy = 0.0; sxu = 0.0; syv = 0.0; sxv = 0.0; syu = 0.0;
  for (var t = 0u; t < 64u; t = t + 1u) {
    let o = t * 12u;
    sw += part[o]; sx += part[o + 1u]; sy += part[o + 2u]; su += part[o + 3u];
    sv += part[o + 4u]; sxx += part[o + 5u]; syy += part[o + 6u]; sxu += part[o + 7u];
    syv += part[o + 8u]; sxv += part[o + 9u]; syu += part[o + 10u];
  }
  var a = 1.0; var b = 0.0; var tx = 0.0; var ty = 0.0; var confidence = 0.0;
  if (sw >= 6.0) {
    let mx = sx / sw; let my = sy / sw; let mu = su / sw; let mv = sv / sw;
    let den = sxx - sw * mx * mx + syy - sw * my * my;
    if (den >= 4.0) {
      a = (sxu + syv - sw * (mx * mu + my * mv)) / den;
      b = (sxv - syu - sw * (mx * mv - my * mu)) / den;
      tx = mu - a * mx + b * my;
      ty = mv - b * mx - a * my;
      let scale = length(vec2f(a, b));
      let shift = length(vec2f(tx, ty));
      if (scale > 0.8 && scale < 1.25 && shift < 40.0) { confidence = 1.0; }
    }
  }
  if (confidence == 0.0) { a = 1.0; b = 0.0; tx = 0.0; ty = 0.0; }
  sim.a = a; sim.b = b; sim.tx = tx; sim.ty = ty; sim.confidence = confidence;
}`;

const FIELD = SIM + `
@group(0) @binding(0) var<storage, read> sim: Sim;
@group(0) @binding(1) var measured: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn data(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  let p = vec2f(id.xy) + 0.5;
  let m = textureLoad(measured, vec2i(id.xy), 0);
  let motion = body(sim, p);
  let local = sim.confidence > 0.5 && m.z >= 0.35 && m.w >= 0.02 && length(m.xy - motion) < 16.0;
  textureStore(dst, id.xy, vec4f(select(motion, m.xy, local), select(0.0, 1.0, local), 1.0));
}`;

const GUESS = SIM + `
@group(0) @binding(0) var<storage, read> sim: Sim;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  textureStore(dst, id.xy, vec4f(body(sim, vec2f(id.xy) + 0.5), 0.0, 1.0));
}`;

const JACOBI = `
@group(0) @binding(0) var dataTex: texture_2d<f32>;
@group(0) @binding(1) var est: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  let pix = vec2i(id.xy);
  if (any(pix >= size)) { return; }
  let sample = textureLoad(dataTex, pix, 0);
  var sum = vec2f(0.0);
  var n = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      if (x == 0 && y == 0) { continue; }
      sum += textureLoad(est, clamp(pix + vec2i(x, y), vec2i(0), size - 1), 0).xy;
      n += 1.0;
    }
  }
  // A textured cell keeps its own measurement. Bare skin takes the cells around it,
  // which is the body motion the field was filled with.
  let w = sample.z;
  let flow = (w * sample.xy + sum) / (w + n);
  textureStore(dst, pix, vec4f(flow, w, 1.0));
}`;

const EXPAND = `
struct Sim { a: f32, b: f32, tx: f32, ty: f32, confidence: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> sim: Sim;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  let dstSize = vec2f(textureDimensions(dst));
  let scale = dstSize / vec2f(textureDimensions(src));
  let uv = (vec2f(id.xy) + 0.5) / dstSize;
  let flow = textureSampleLevel(src, samp, uv, 0.0).xy * scale;
  textureStore(dst, id.xy, vec4f(flow, sim.confidence, 1.0));
}`;

const PYRAMID_USAGE = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

export class MotionWarp {
  static async create(device, {width = 320, height = 180, levels = 5, iterations = 4, keySlots = 3, lambda = 1e-3, maxStep = 2, gridLevel = 2} = {}) {
    const compute = async (code, entryPoint) => {
      const module = device.createShaderModule({code});
      const info = await module.getCompilationInfo();
      const messages = info.messages.filter(item => item.type === 'error');
      if (messages.length) throw new Error(`${entryPoint}: ${messages.map(item => `${item.lineNum}:${item.message}`).join(' | ')}`);
      return device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint}});
    };
    const [luma, down, lk, seed, fit, data, guess, jacobi, expand] = await Promise.all([
      compute(LUMA, 'luma'), compute(LUMA, 'down'), compute(LK, 'lk'), compute(SEED, 'main'),
      compute(FIT, 'main'), compute(FIELD, 'data'), compute(GUESS, 'main'), compute(JACOBI, 'main'), compute(EXPAND, 'main'),
    ]);
    return new MotionWarp(device, {width, height, levels, iterations, keySlots, lambda, maxStep, gridLevel, luma, down, lk, seed, fit, data, guess, jacobi, expand});
  }

  constructor(device, o) {
    Object.assign(this, {device, ...o});
    this.sampler = device.createSampler({magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
    this.sizes = [];
    for (let level = 0, w = o.width, h = o.height; level < o.levels; level++, w = Math.ceil(w / 2), h = Math.ceil(h / 2)) this.sizes.push([w, h]);
    const pyramid = () => this.sizes.map(size => device.createTexture({size, format: 'rgba16float', usage: PYRAMID_USAGE}));
    this.current = pyramid();
    this.keys = Array.from({length: o.keySlots}, () => ({frameId: -1, levels: pyramid(), groups: new Map()}));
    this.flows = this.sizes.map(size => [0, 1].map(() => device.createTexture({size, format: 'rgba16float', usage: PYRAMID_USAGE | GPUTextureUsage.COPY_SRC})));
    this.gridLevel = o.gridLevel;
    const grid = this.sizes[o.gridLevel];
    const coarse = this.sizes[o.levels - 1];
    const storage = () => device.createBuffer({size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    this.sim = storage();
    this.prevSim = storage();
    // Identity until the first pair is measured. a = 1, and the rest is zero.
    const identity = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    device.queue.writeBuffer(this.sim, 0, identity);
    device.queue.writeBuffer(this.prevSim, 0, identity);
    this.gridSize = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(this.gridSize, 0, new Float32Array([grid[0], grid[1], 0, 0]));
    this.seedFlow = device.createTexture({size: coarse, format: 'rgba16float', usage: PYRAMID_USAGE});
    this.fieldData = device.createTexture({size: grid, format: 'rgba16float', usage: PYRAMID_USAGE});
    this.field = [0, 1].map(() => device.createTexture({size: grid, format: 'rgba16float', usage: PYRAMID_USAGE}));
    this.expanded = device.createTexture({size: [o.width, o.height], format: 'rgba16float', usage: PYRAMID_USAGE | GPUTextureUsage.COPY_SRC});
    this.currentGroups = new Map();
    const stepBuffer = scale => {
      const buffer = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(buffer, 0, new Float32Array([scale, o.lambda, o.maxStep, 0]));
      return buffer;
    };
    this.steps = {hold: stepBuffer(1), up: stepBuffer(2)};
    for (const key of this.keys) key.flowGroups = this.flowGroups(key.levels);
    const measured = this.flows[o.gridLevel][(o.iterations - 1) % 2];
    const view = texture => texture.createView();
    this.seedGroup = device.createBindGroup({layout: o.seed.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: this.prevSim}},
      {binding: 1, resource: {buffer: this.gridSize}},
      {binding: 2, resource: view(this.seedFlow)},
    ]});
    this.fitGroup = device.createBindGroup({layout: o.fit.getBindGroupLayout(0), entries: [
      {binding: 0, resource: view(measured)},
      {binding: 1, resource: {buffer: this.sim}},
    ]});
    this.initDataGroup = device.createBindGroup({layout: o.data.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: this.sim}},
      {binding: 1, resource: view(measured)},
      {binding: 2, resource: view(this.fieldData)},
    ]});
    this.initEstGroup = device.createBindGroup({layout: o.guess.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: this.sim}},
      {binding: 1, resource: view(this.field[0])},
    ]});
    this.jacobiGroups = [];
    for (let pass = 0; pass < 6; pass++) {
      this.jacobiGroups.push(device.createBindGroup({layout: o.jacobi.getBindGroupLayout(0), entries: [
        {binding: 0, resource: view(this.fieldData)},
        {binding: 1, resource: view(this.field[pass % 2])},
        {binding: 2, resource: view(this.field[(pass + 1) % 2])},
      ]}));
    }
    this.expandGroup = device.createBindGroup({layout: o.expand.getBindGroupLayout(0), entries: [
      {binding: 0, resource: this.sampler},
      {binding: 1, resource: view(this.field[0])},
      {binding: 2, resource: {buffer: this.sim}},
      {binding: 3, resource: view(this.expanded)},
    ]});
    this.maskFrameId = -1;
  }

  pyramidGroups(levels, source) {
    return levels.map((dst, level) => this.device.createBindGroup({
      layout: (level ? this.down : this.luma).getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: this.sampler},
        {binding: 1, resource: (level ? levels[level - 1] : source).createView()},
        {binding: 2, resource: dst.createView()},
      ]}));
  }

  cachedGroups(cache, levels, source) {
    let groups = cache.get(source);
    if (!groups) {
      groups = this.pyramidGroups(levels, source);
      cache.set(source, groups);
    }
    return groups;
  }

  // Coarse to the grid. Finer than the grid, skin would get its own vector.
  // The grid result lands in flows[gridLevel][(iterations - 1) % 2].
  flowGroups(keyLevels) {
    const groups = [];
    for (let level = this.levels - 1; level >= this.gridLevel; level--) {
      for (let i = 0; i < this.iterations; i++) {
        const out = this.flows[level][i % 2];
        const coarsest = level === this.levels - 1;
        const coarser = coarsest ? this.seedFlow : this.flows[level + 1][(this.iterations - 1) % 2];
        const input = i ? this.flows[level][(i - 1) % 2] : coarser;
        const step = i === 0 && !coarsest ? this.steps.up : this.steps.hold;
        groups.push({size: this.sizes[level], group: this.device.createBindGroup({layout: this.lk.getBindGroupLayout(0), entries: [
          {binding: 0, resource: this.sampler},
          {binding: 1, resource: this.current[level].createView()},
          {binding: 2, resource: keyLevels[level].createView()},
          {binding: 3, resource: input.createView()},
          {binding: 4, resource: out.createView()},
          {binding: 5, resource: {buffer: step}},
        ]})});
      }
    }
    return groups;
  }

  get flowTexture() {
    return this.expanded;
  }

  // Level-0 luma of the frame most recently passed to encode().
  get currentLuma() {
    return this.current[0];
  }

  pyramid(pass, groups) {
    groups.forEach((group, level) => {
      pass.setPipeline(level ? this.down : this.luma);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(this.sizes[level][0] / 8), Math.ceil(this.sizes[level][1] / 8));
    });
  }

  // Keep a pyramid of `source` under `frameId`, the frame an inference is starting on.
  // The pyramid already on screen is not evicted.
  captureKey(frameId, source) {
    if (!source) return;
    const shown = this.maskFrameId >= 0 ? this.maskFrameId : null;
    const free = this.keys.filter(key => key.frameId !== shown).sort((a, b) => a.frameId - b.frameId)[0];
    free.frameId = frameId;
    const encoder = this.device.createCommandEncoder({label: 'motion key pyramid'});
    const pass = encoder.beginComputePass();
    this.pyramid(pass, this.cachedGroups(free.groups, free.levels, source));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  keyLuma(frameId) {
    return this.keys.find(key => key.frameId === frameId)?.levels[0];
  }

  setMaskFrame(frameId) {
    this.maskFrameId = frameId;
  }

  // Encode flow from `source` (frame `currentFrameId`) back to the mask's frame.
  // Returns false when that pyramid is gone or the mask is already for this frame.
  encode(encoder, currentFrameId, source) {
    if (this.maskFrameId < 0 || !source || this.maskFrameId === currentFrameId) return false;
    const key = this.keys.find(key => key.frameId === this.maskFrameId);
    if (!key) return false;
    const dispatch = (pass, pipeline, group, size) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8));
    };
    const prep = encoder.beginComputePass();
    this.pyramid(prep, this.cachedGroups(this.currentGroups, this.current, source));
    dispatch(prep, this.seed, this.seedGroup, this.sizes[this.levels - 1]);
    prep.end();
    const search = encoder.beginComputePass();
    search.setPipeline(this.lk);
    for (const {size, group} of key.flowGroups) {
      search.setBindGroup(0, group);
      search.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8));
    }
    search.end();
    const fit = encoder.beginComputePass();
    fit.setPipeline(this.fit);
    fit.setBindGroup(0, this.fitGroup);
    fit.dispatchWorkgroups(1);
    fit.end();
    const fill = encoder.beginComputePass();
    dispatch(fill, this.data, this.initDataGroup, this.sizes[this.gridLevel]);
    dispatch(fill, this.guess, this.initEstGroup, this.sizes[this.gridLevel]);
    fill.end();
    for (const group of this.jacobiGroups) {
      const blur = encoder.beginComputePass();
      dispatch(blur, this.jacobi, group, this.sizes[this.gridLevel]);
      blur.end();
    }
    const expand = encoder.beginComputePass();
    dispatch(expand, this.expand, this.expandGroup, [this.width, this.height]);
    expand.end();
    encoder.copyBufferToBuffer(this.sim, 0, this.prevSim, 0, 32);
    return true;
  }
}
