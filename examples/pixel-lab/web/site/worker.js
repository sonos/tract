// Runs the segmenter and the compositor off the page's thread, each on a WebGPU device of its own.
import {model, modelUrl} from './models.mjs';

let modulePromise, segmenter, tractDevice, composer, device, frameCanvas, frameContext, modelCanvas, modelContext, mask, gpuFrames = 0, framesSeen = 0, inferredFrame = 0;
// Composited frames submitted and not finished yet. Past this a new frame is dropped rather than
// queued, so latency stays bounded; three still covers Firefox's ~100 ms completion callback at 30 fps.
const MAX_GPU_FRAMES = 3;
const post = message => postMessage(message);

async function wasm() {
  if (!modulePromise) modulePromise = (async () => {
    if (!self.navigator.gpu) throw new Error('This browser has no WebGPU in workers.');
    post({status: 'Loading Rust / Wasm…'});
    const loaded = await import('./pkg/pixel_lab_web.js');
    await loaded.default({module_or_path: new URL('./pkg/pixel_lab_web_bg.wasm', import.meta.url)});
    return loaded;
  })();
  return modulePromise;
}

// The model file, from this browser's cache when an earlier visit stored it.
async function download(m) {
  const url = modelUrl(m);
  const cache = await caches.open('pixel-lab-models').catch(() => null);
  const cached = await cache?.match(url);
  if (cached) return new Uint8Array(await cached.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(m.bytes);
  const reader = response.body.getReader();
  let loaded = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) throw new Error(`${m.name}: more bytes than expected`);
    bytes.set(value, loaded);
    loaded += value.length;
    post({download: loaded / m.bytes, status: `Downloading ${m.name}: ${(loaded / 1e6).toFixed(1)} / ${(m.bytes / 1e6).toFixed(1)} MB`});
  }
  if (loaded !== m.bytes) throw new Error(`${m.name}: got ${loaded} bytes, expected ${m.bytes}`);
  await cache?.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

function audit(runner) {
  const nodes = JSON.parse(runner.graph()).nodes;
  const glue = new Set(['Source', 'Const', 'DeviceSyncToDevice', 'DeviceSyncToHost']);
  const gpu = op => op.startsWith('Wgpu') || op.startsWith('Gpu');
  const fallback = nodes.filter(node => !gpu(node.op_name) && !glue.has(node.op_name));
  if (fallback.length) throw new Error(`CPU compute fallback: ${fallback.map(node => node.op_name).join(', ')}`);
  return nodes.filter(node => gpu(node.op_name)).length;
}

// The compositor has a GPU device of its own. A device's queue runs in order, so on tract's device
// every composite would wait behind the inference before it; here the camera is composited at its
// own rate with the latest mask, and each finished mask is copied across, once per inference.
class MaskRelay {
  constructor(source, target, width, height) {
    Object.assign(this, {source, target, width, height, busy: false});
    this.bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    this.staging = tractDevice.createBuffer({size: this.bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  }

  // Run tract on the latest frame; resolves once the mask is copied.
  async infer() {
    this.busy = true;
    try {
      const started = performance.now();
      segmenter.run(modelCanvas);
      const encoder = tractDevice.createCommandEncoder({label: 'mask readback'});
      encoder.copyTextureToBuffer({texture: this.source}, {buffer: this.staging, bytesPerRow: this.bytesPerRow}, [this.width, this.height]);
      tractDevice.queue.submit([encoder.finish()]);
      await this.staging.mapAsync(GPUMapMode.READ);
      device.queue.writeTexture({texture: this.target}, this.staging.getMappedRange(), {bytesPerRow: this.bytesPerRow}, [this.width, this.height]);
      this.staging.unmap();
      post({mask: performance.now() - started});
    } finally {
      this.busy = false;
    }
  }
}

// Start an inference on the latest camera frame if none is running. When a mask lands and a newer
// frame is already there, the next inference starts at once instead of waiting for the next frame.
function inferLatest() {
  if (mask.busy || inferredFrame === framesSeen) return;
  inferredFrame = framesSeen;
  modelContext.drawImage(frameCanvas, 0, 0, modelCanvas.width, modelCanvas.height);
  mask.infer().then(inferLatest, error => post({fatal: `Inference failed: ${error?.message || error}`}));
}

async function command(type, payload) {
  if (type === 'init') {
    const m = model(payload.model);
    const [{Segmenter}, bytes, {Compositor}] = await Promise.all([wasm(), download(m), import('./compositor.js')]);
    post({status: `Preparing ${m.name} on WebGPU…`});
    segmenter = await Segmenter.create(bytes, m.width, m.height, m.mean, m.std);
    const ops = audit(segmenter);
    tractDevice = segmenter.device();
    device = await (await navigator.gpu.requestAdapter({powerPreference: 'high-performance'})).requestDevice();
    for (const d of [tractDevice, device]) {
      d.addEventListener('uncapturederror', event => post({fatal: `WebGPU: ${event.error.message}`}));
      d.lost.then(info => post({fatal: `WebGPU device lost: ${info.message}`}));
    }
    const target = device.createTexture({size: [m.width, m.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST});
    mask = new MaskRelay(segmenter.mask_texture(), target, m.width, m.height);
    composer = await Compositor.create(device, target, payload.canvas);
    frameCanvas = new OffscreenCanvas(payload.canvas.width, payload.canvas.height);
    frameContext = frameCanvas.getContext('2d');
    modelCanvas = new OffscreenCanvas(m.width, m.height);
    modelContext = modelCanvas.getContext('2d');
    framesSeen = inferredFrame = 0;
    return {ops};
  }
  if (type === 'background') {
    if (payload.mode === 'colour') composer.setColour(payload.colour);
    else composer.setBlur();
    return {};
  }
  if (type === 'frame') {
    const bitmap = payload.bitmap;
    if (gpuFrames >= MAX_GPU_FRAMES) {
      bitmap.close();
      return {dropped: true};
    }
    try {
      const scale = Math.max(frameCanvas.width / bitmap.width, frameCanvas.height / bitmap.height);
      const width = frameCanvas.width / scale, height = frameCanvas.height / scale;
      frameContext.drawImage(bitmap, (bitmap.width - width) / 2, (bitmap.height - height) / 2, width, height, 0, 0, frameCanvas.width, frameCanvas.height);
      composer.uploadFrame(frameCanvas);
      framesSeen++;
      inferLatest();
    } finally { bitmap.close(); }
    composer.draw(payload.refine, payload.view);
    gpuFrames++;
    device.queue.onSubmittedWorkDone().then(() => {
      gpuFrames--;
      post({completed: true});
    });
    return {};
  }
  throw new Error(`Unknown command: ${type}`);
}

let pending = Promise.resolve();
self.onmessage = ({data}) => {
  pending = pending.then(async () => {
    try {
      postMessage({id: data.id, result: await command(data.type, data.payload)});
    } catch (error) {
      postMessage({id: data.id, error: `${error?.name || 'Error'}: ${error?.message || error}`});
    }
  });
};
