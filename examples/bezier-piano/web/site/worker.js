// Hand tracking off the page's thread: tract runs MediaPipe's palm detector and
// hand landmark model, this file crops the camera frame for them and keeps the
// hands from one frame to the next.
const stamp = new URL(import.meta.url).searchParams.get('stamp') || '';
const stamped = file => import(`./${file}${stamp ? `?stamp=${stamp}` : ''}`);

// The float16 Hand Landmarker bundle Google publishes for MediaPipe Tasks: a
// stored zip of hand_detector.tflite and hand_landmarks_detector.tflite.
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1';
const MAX_HANDS = 2;
const MIN_PRESENCE = 0.5;
// With one hand tracked, look for a second one every this many frames.
const SEARCH_EVERY = 4;

let geometry, wasm, task, loading, hands, backend;
// Both backends stay built once used, so switching back is instant.
const engines = {};
const LABELS = {wgpu: 'tract · WebGPU', cpu: 'tract · wasm CPU'};
let tracks = [];
let frames = 0;
const post = message => postMessage(message);
const progress = (text, fraction) => post({progress: {text, fraction}});

async function download(url, onFraction) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`The hand model did not download (HTTP ${response.status}).`);
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total) onFraction(received / total);
  }
  const bytes = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.byteLength;
  }
  return bytes;
}

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
}

async function load() {
  progress('Loading Rust / Wasm…', 0.02);
  geometry = await stamped('tracking.mjs');
  wasm = await import('./pkg/bezier_piano_web.js');
  await wasm.default({module_or_path: new URL('./pkg/bezier_piano_web_bg.wasm', import.meta.url)});
  progress('Downloading the hand models…', 0.08);
  task = await download(MODEL_URL, f => progress('Downloading the hand models…', 0.08 + f * 0.8));
  if (await sha256(task) !== MODEL_SHA256) throw new Error('The hand model download is not the expected file.');
}

// Builds the models on one backend, once. Returns what the page shows.
async function use(wanted) {
  if (wanted === 'wgpu' && !self.navigator.gpu) throw new Error('This browser has no WebGPU in workers.');
  if (!engines[wanted]) {
    progress(wanted === 'wgpu' ? 'Preparing the models for WebGPU…' : 'Optimizing the models for the CPU…', 0.9);
    engines[wanted] = wasm.Hands.create(task, wanted);
    engines[wanted].catch(() => { delete engines[wanted]; });
  }
  hands = await engines[wanted];
  backend = wanted;
  tracks = [];
  return {backend: LABELS[wanted], name: wanted};
}

const canvases = {};
function surface(side) {
  if (!canvases[side]) {
    const canvas = new OffscreenCanvas(side, side);
    const ctx = canvas.getContext('2d', {willReadFrequently: true, alpha: false});
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    canvases[side] = {ctx, rgb: new Float32Array(side * side * 3)};
  }
  return canvases[side];
}

function pixels(side, draw) {
  const {ctx, rgb} = surface(side);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, side, side);
  draw(ctx);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const rgba = ctx.getImageData(0, 0, side, side).data;
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i] / 255;
    rgb[j + 1] = rgba[i + 1] / 255;
    rgb[j + 2] = rgba[i + 2] / 255;
  }
  return rgb;
}

async function landmarksIn(bitmap, roi) {
  const side = geometry.LANDMARKS_SIZE;
  const rgb = pixels(side, ctx => {
    ctx.setTransform(...geometry.cropTransform(roi, side));
    ctx.drawImage(bitmap, 0, 0);
  });
  const out = await hands.landmarks(rgb);
  if (out[63] < MIN_PRESENCE) return null;
  return geometry.projectLandmarks(out, roi, side);
}

async function palmsIn(bitmap) {
  const side = geometry.DETECTOR_SIZE;
  const box = geometry.letterbox(bitmap.width, bitmap.height, side);
  const rgb = pixels(side, ctx => {
    ctx.drawImage(bitmap, box.padX, box.padY, bitmap.width * box.scale, bitmap.height * box.scale);
  });
  return geometry.weightedNms(geometry.decodePalms(await hands.detect(rgb), box), 0.3, MAX_HANDS);
}

async function track(bitmap, time) {
  frames++;
  const timing = {detect: 0, landmarks: 0};
  const timed = async (key, run) => {
    const start = performance.now();
    const value = await run();
    timing[key] += performance.now() - start;
    return value;
  };
  const kept = [];
  for (const old of tracks) {
    const points = await timed('landmarks', () => landmarksIn(bitmap, old.roi));
    if (points) kept.push({...old, points});
  }
  const search = kept.length === 0 || (kept.length < MAX_HANDS && frames % SEARCH_EVERY === 0);
  if (search) {
    for (const palm of await timed('detect', () => palmsIn(bitmap))) {
      if (kept.length >= MAX_HANDS) break;
      const roi = geometry.palmRoi(palm);
      const box = geometry.roiBox(roi);
      if (kept.some(hand => geometry.boxIou(box, geometry.roiBox(hand.roi)) > 0.5)) continue;
      const points = await timed('landmarks', () => landmarksIn(bitmap, roi));
      if (points) kept.push({roi, points, smoother: new geometry.LandmarkSmoother()});
    }
  }
  tracks = kept.map(hand => ({...hand, roi: geometry.landmarksRoi(hand.points)}));
  const landmarks = kept.map(hand => hand.smoother.apply(hand.points, time)
    .map(p => ({x: p.x / bitmap.width, y: p.y / bitmap.height})));
  return {landmarks, timing, searched: search};
}

onmessage = async ({data}) => {
  const {id, type, bitmap} = data;
  try {
    if (type === 'init') {
      loading ??= load();
      await loading;
      let result;
      try {
        result = await use(data.backend || 'wgpu');
      } catch (error) {
        if ((data.backend || 'wgpu') !== 'wgpu') throw error;
        console.warn('WebGPU unavailable, using the CPU.', error);
        result = {...await use('cpu'), fallback: `${error.message || error}`};
      }
      post({id, result: {...result, connections: geometry.HAND_CONNECTIONS}});
      return;
    }
    if (type === 'backend') {
      post({id, result: await use(data.backend)});
      return;
    }
    if (type === 'reset') {
      tracks = [];
      post({id, result: true});
      return;
    }
    if (type === 'frame') {
      const started = performance.now();
      const result = await track(bitmap, started);
      bitmap.close();
      post({id, result: {...result, ms: performance.now() - started, backend: LABELS[backend]}});
      return;
    }
    post({id, error: `Unknown request ${type}`});
  } catch (error) {
    try { bitmap?.close(); } catch { /* already closed */ }
    if (type === 'init') loading = null;
    post({id, error: `${error.message || error}`});
  }
};
