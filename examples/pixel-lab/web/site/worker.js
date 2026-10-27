// Runs the segmenter and the compositor on one WebGPU device, off the page's thread.
import {model, modelUrl} from './models.mjs';

let modulePromise, segmenter, composer, device, frameCanvas, frameContext, modelCanvas, modelContext, frames = 0, profileEvery = 15;
const post = message => postMessage(message);
const absoluteNow = () => performance.timeOrigin + performance.now();

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

async function command(type, payload) {
  if (type === 'init') {
    const m = model(payload.model);
    const [{Segmenter}, bytes, {Compositor}] = await Promise.all([wasm(), download(m), import('./compositor.js')]);
    post({status: `Preparing ${m.name} on WebGPU…`});
    segmenter = await Segmenter.create(bytes, m.width, m.height, m.mean, m.std);
    const ops = audit(segmenter);
    device = segmenter.device();
    device.addEventListener('uncapturederror', event => post({fatal: `WebGPU: ${event.error.message}`}));
    device.lost.then(info => post({fatal: `WebGPU device lost: ${info.message}`}));
    composer = await Compositor.create(device, segmenter.mask_texture(), payload.canvas);
    frames = 0;
    // Stage timing waits for GPU completion between stages. Where that
    // round-trip has a ~100 ms floor it would measure the floor, not the work,
    // and stall the frame it samples, so stage timing is skipped there.
    const roundTrips = [];
    for (let i = 0; i < 3; i++) {
      const started = performance.now();
      device.queue.submit([device.createCommandEncoder().finish()]);
      await device.queue.onSubmittedWorkDone();
      roundTrips.push(performance.now() - started);
    }
    const stageTiming = roundTrips.sort((a, b) => a - b)[1] <= 25;
    profileEvery = stageTiming ? 15 : Infinity;
    frameCanvas = new OffscreenCanvas(payload.canvas.width, payload.canvas.height);
    frameContext = frameCanvas.getContext('2d');
    modelCanvas = new OffscreenCanvas(m.width, m.height);
    modelContext = modelCanvas.getContext('2d');
    return {ops, stageTiming};
  }
  if (type === 'background') {
    if (payload.mode === 'colour') composer.setColour(payload.colour);
    else composer.setBlur();
    return {};
  }
  if (type === 'frame') {
    const profile = ++frames % profileEvery === 0;
    const bitmap = payload.bitmap;
    try {
      const scale = Math.max(frameCanvas.width / bitmap.width, frameCanvas.height / bitmap.height);
      const width = frameCanvas.width / scale, height = frameCanvas.height / scale;
      frameContext.drawImage(bitmap, (bitmap.width - width) / 2, (bitmap.height - height) / 2, width, height, 0, 0, frameCanvas.width, frameCanvas.height);
      modelContext.drawImage(frameCanvas, 0, 0, modelCanvas.width, modelCanvas.height);
      composer.uploadFrame(frameCanvas);
    } finally { bitmap.close(); }
    if (profile) await device.queue.onSubmittedWorkDone();
    const captured = absoluteNow();
    segmenter.run(modelCanvas);
    if (profile) await device.queue.onSubmittedWorkDone();
    const inferred = absoluteNow();
    composer.draw(payload.refine);
    if (profile) await device.queue.onSubmittedWorkDone();
    const composited = absoluteNow();
    return {stages: profile ? {capture: captured - payload.captureStart, inference: inferred - captured, compositing: composited - inferred} : null, total: composited - payload.captureStart};
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
