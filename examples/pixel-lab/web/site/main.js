import {MODELS, CAPTURES, model, capture} from './models.mjs';
import {GpuClient} from './gpu-client.js';
import {FrameRate} from './frame-rate.mjs';

const $ = id => document.getElementById(id);
const status = text => ($('status').textContent = text);
const bar = fraction => ($('bar').style.width = `${Math.round(fraction * 100)}%`);
const go = $('go');
const video = $('camera-feed');
const state = {client: null, stream: null, running: false, starting: false, generation: 0, callback: null, inFlight: null};
// Past this without a composited frame or a mask, the camera is stopped with what was pending.
const STALL_MS = 15000;
const WORKER_SILENCE_MS = 10000;

for (const m of MODELS) $('model').append(new Option(m.name, m.id));
for (const c of CAPTURES) $('capture').append(new Option(`${c.width} × ${c.height}`, c.id));

function idle(label = 'Start camera') {
  go.disabled = false;
  go.textContent = label;
}

function resetStats() {
  Object.assign(state, {rate: new FrameRate(), cameraRate: new FrameRate(), maskRate: new FrameRate(), frames: 0, masks: 0, latencies: [], lastStats: 0});
  $('camera-fps').textContent = $('out-fps').textContent = $('stats').textContent = '';
}

const median = values => [...values].sort((a, b) => a - b)[values.length >> 1];

// Composited frames are counted when the GPU has finished them, not when they were submitted.
function frameCompleted() {
  state.lastCompleted = performance.now();
  state.rate.add(state.lastCompleted, ++state.frames);
  showStats();
}

function maskReady(ms) {
  state.lastMask = performance.now();
  state.maskRate.add(state.lastMask, ++state.masks);
  state.latencies.push(ms);
  if (state.latencies.length > 60) state.latencies.shift();
  showStats();
}

function showStats() {
  const now = performance.now();
  if (now - state.lastStats < 250) return;
  state.lastStats = now;
  const rate = r => (r.fps() === null ? '' : `${r.fps().toFixed(1)} fps`);
  $('out-fps').textContent = rate(state.rate);
  $('camera-fps').textContent = rate(state.cameraRate);
  const maskFps = state.maskRate.fps();
  $('stats').textContent = maskFps === null ? '' : `mask ${maskFps.toFixed(1)} fps · ${median(state.latencies).toFixed(0)} ms from frame to mask`;
}

async function applyBackground() {
  const mode = $('background').value;
  $('colour').disabled = mode !== 'colour';
  if (state.client) await state.client.request('background', {mode, colour: $('colour').value});
}
$('background').onchange = () => applyBackground().catch(error => status(error.message));
$('colour').oninput = () => applyBackground().catch(error => status(error.message));
$('refine').onchange = resetStats;

// What the frame loop is stuck on, or '' while frames and masks keep coming.
function stallReport(now = performance.now()) {
  const ago = t => `${((now - t) / 1000).toFixed(0)} s`;
  const timers = state.tick !== undefined && now - state.tick > 3000 ? [`page timers last ran ${ago(state.tick)} ago`] : [];
  if (!state.running) {
    const worker = state.starting && state.lastWorker && now - state.lastWorker.at > WORKER_SILENCE_MS ? [`starting, last word from the GPU worker ${ago(state.lastWorker.at)} ago: ${state.lastWorker.status}`] : [];
    return [...worker, ...timers].join('; ');
  }
  if (document.hidden) return timers.join('; ');
  const since = t => Math.max(t ?? 0, state.liveSince);
  const stalled = [];
  if (now - since(state.lastCompleted) > STALL_MS) stalled.push(`no composited frame for ${ago(since(state.lastCompleted))}`);
  if (now - since(state.lastMask) > STALL_MS) stalled.push(`no mask for ${ago(since(state.lastMask))}`);
  if (!stalled.length) return timers.join('; ');
  const reply = state.lastReply;
  const cause = state.waiting ? `${state.waiting.stage} pending for ${ago(state.waiting.since)}`
    : reply ? `last frame reply ${ago(reply.at)} ago: ${reply.dropped ? 'dropped, ' : ''}${reply.gpuFrames} frames on the GPU, mask inference ${reply.inferring ? 'running' : 'idle'}`
    : 'no frame sent yet';
  return [...stalled, cause, ...timers].join('; ');
}
window.pixelLabStall = () => stallReport();

function watch() {
  const report = state.running && stallReport();
  if (report) stop(`Stopped: ${report}`);
}
setInterval(() => ((state.tick = performance.now()), watch()), 1000);

function stop(message = 'Camera stopped.') {
  state.generation++;
  state.running = false;
  state.inFlight = state.waiting = null;
  if (state.callback !== null) video.cancelVideoFrameCallback(state.callback);
  state.callback = null;
  state.client?.close();
  state.client = null;
  state.stream?.getTracks().forEach(track => track.stop());
  state.stream = null;
  video.srcObject = null;
  $('camera-fps').textContent = $('out-fps').textContent = '';
  for (const id of ['model', 'camera', 'capture']) $(id).disabled = false;
  status(message);
  idle();
}

async function listCameras(selected) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
  $('camera').replaceChildren(...devices.map((device, i) => new Option(device.label || `Camera ${i + 1}`, device.deviceId)));
  if (selected) $('camera').value = selected;
}

async function start() {
  if (state.running || state.starting) return;
  state.starting = true;
  const generation = ++state.generation;
  go.disabled = true;
  for (const id of ['model', 'camera', 'capture']) $(id).disabled = true;
  try {
    const size = capture($('capture').value);
    const m = model($('model').value);
    status('Asking for the camera…');
    const deviceId = $('camera').value;
    const stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {exact: size.width}, height: {exact: size.height}, frameRate: {ideal: 30}, ...(deviceId ? {deviceId: {exact: deviceId}} : {facingMode: 'user'})}});
    if (generation !== state.generation) return stream.getTracks().forEach(track => track.stop());
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    stream.getVideoTracks()[0].addEventListener('ended', () => generation === state.generation && stop('The camera disconnected.'), {once: true});
    const settings = stream.getVideoTracks()[0].getSettings();
    listCameras(settings.deviceId).catch(() => {});
    const previous = $('out');
    const output = previous.cloneNode();
    output.width = settings.width || video.videoWidth;
    output.height = settings.height || video.videoHeight;
    previous.replaceWith(output);
    const canvas = output.transferControlToOffscreen();
    bar(0);
    state.client = new GpuClient(message => {
      if (generation !== state.generation) return;
      if (message.status) state.lastWorker = {at: performance.now(), status: message.status};
      if (message.download !== undefined) bar(message.download);
      if (message.status) status(message.status);
      if (message.fatal) stop(message.fatal);
      if (message.completed && state.running) frameCompleted();
      if (message.mask !== undefined && state.running) maskReady(message.mask);
    });
    const prepared = await state.client.request('init', {canvas, model: m.id}, [canvas], 180000);
    if (generation !== state.generation) return;
    await applyBackground();
    bar(1);
    resetStats();
    Object.assign(state, {liveSince: performance.now(), lastCompleted: undefined, lastMask: undefined, lastReply: null, waiting: null, lastWorker: null});
    state.running = true;
    go.disabled = false;
    go.textContent = 'Stop camera';
    status(`Live: ${m.name}, ${m.note}, ${prepared.ops} GPU ops.`);
    const frame = (_now, metadata) => {
      if (!state.running || generation !== state.generation) return;
      state.callback = video.requestVideoFrameCallback(frame);
      state.cameraRate.add(performance.now(), metadata.presentedFrames);
      watch();
      if (!state.running || state.inFlight || document.hidden) return;
      state.inFlight = (async () => {
        try {
          state.waiting = {stage: 'createImageBitmap', since: performance.now()};
          const bitmap = await createImageBitmap(video);
          if (!state.running || generation !== state.generation) return bitmap.close();
          state.waiting = {stage: 'frame request', since: performance.now()};
          const reply = await state.client.request('frame', {bitmap, refine: $('refine').checked, view: $('view').value}, [bitmap], 15000);
          state.lastReply = {...reply, at: performance.now()};
        } catch (error) {
          if (generation === state.generation) stop(`Stopped: ${error.message || error}`);
        } finally {
          if (generation === state.generation) state.inFlight = state.waiting = null;
        }
      })();
    };
    state.callback = video.requestVideoFrameCallback(frame);
  } catch (error) {
    if (generation !== state.generation) return;
    const reasons = {
      NotAllowedError: 'Camera access was denied. Allow it in the site settings, then retry.',
      NotFoundError: 'No camera found.',
      NotReadableError: 'The camera is busy in another application.',
      OverconstrainedError: 'This camera cannot capture at that size. Pick another one.',
    };
    stop(reasons[error.name] || `Error: ${error.message || error}`);
  } finally {
    state.starting = false;
  }
}

go.onclick = () => (state.running ? stop() : start());
$('camera').onchange = () => state.running && (stop(), start());
document.addEventListener('visibilitychange', () => !document.hidden && state.running && (resetStats(), (state.liveSince = performance.now())));
window.addEventListener('pagehide', () => stop());

// Which commit and day this page was built from, when the build says so.
async function buildInfo() {
  try {
    const {commit, dirty, date} = await (await fetch('build.json')).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/pixel-lab/web`;
    $('build').textContent = `${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
  } catch {}
}

function check() {
  if (!window.isSecureContext) return 'The camera needs a secure context: HTTPS, or localhost.';
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser exposes no camera here.';
  if (!navigator.gpu) return 'This browser has no WebGPU. Try a recent Chrome, Edge, Firefox or Safari.';
  if (!window.Worker || !HTMLCanvasElement.prototype.transferControlToOffscreen) return 'This browser lacks OffscreenCanvas in workers.';
  if (!HTMLVideoElement.prototype.requestVideoFrameCallback) return 'This browser lacks requestVideoFrameCallback.';
  return null;
}

buildInfo();
const problem = check();
if (problem) {
  status(problem);
  go.textContent = 'Unsupported';
} else {
  status('Ready. The first start downloads the chosen model.');
  idle();
}
