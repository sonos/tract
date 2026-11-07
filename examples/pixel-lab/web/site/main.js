const stamp = new URL(import.meta.url).searchParams.get('stamp') || Date.now();
const stamped = file => import(`./${file}?stamp=${stamp}`);
const {CAPTURES, model, capture, rvmNetworkSize, rvmScale, formatR, modelsFor} = await stamped('models.mjs');
const {GpuClient} = await stamped('gpu-client.js');
const {FrameRate, isNewSourceFrame} = await stamped('frame-rate.mjs');
const {inspectorView, textureBytes} = await stamped('inspector.mjs');

const $ = id => document.getElementById(id);
const status = text => ($('status').textContent = text);
const bar = fraction => ($('bar').style.width = `${Math.round(fraction * 100)}%`);
const go = $('go');
const video = $('camera-feed');
const state = {client: null, stream: null, running: false, starting: false, generation: 0, callback: null, inFlight: null};
// Past this without a composited frame or a mask, the camera is stopped with what was pending.
const STALL_MS = 15000;
const WORKER_SILENCE_MS = 10000;

for (const c of CAPTURES) $('capture').append(new Option(`${c.width} × ${c.height}`, c.id));
$('capture').value = '1280x720';

function idle(label = 'Start camera') {
  go.disabled = false;
  go.textContent = label;
}

function resetStats() {
  Object.assign(state, {
    rate: new FrameRate(), cameraRate: new FrameRate(), maskRate: new FrameRate(),
    frames: 0, masks: 0, latencies: [], lastStats: 0,
    cameraTimes: [], outputTimes: [], compositeMs: [], sent: 0, drops: 0, lastPresented: undefined,
  });
  $('camera-fps').textContent = $('out-fps').textContent = $('stats').textContent = '';
  showInspector();
}

const median = values => [...values].sort((a, b) => a - b)[values.length >> 1];

// Composited frames are counted when the GPU has finished them, not when they were submitted.
function frameCompleted() {
  state.lastCompleted = performance.now();
  state.outputTimes.push(state.lastCompleted);
  if (state.outputTimes.length > 60) state.outputTimes.shift();
  state.rate.add(state.lastCompleted, ++state.frames);
  showStats();
}

function noteComposite(ms, advanced) {
  state.compositeMs.push(ms);
  if (state.compositeMs.length > 60) state.compositeMs.shift();
  if (advanced) frameCompleted();
  else showStats();
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
  $('stats').textContent = maskFps === null ? '' : `mask ${maskFps.toFixed(1)} fps · ${median(state.latencies).toFixed(0)} ms to compute the mask${motionSuffix()}`;
  showInspector();
}

function tensorSize() {
  const size = capture($('capture').value);
  const chosen = model($('model').value);
  if (chosen.kind === 'rvm') {
    const network = rvmNetworkSize(size.width, size.height, rvmScale(size.width, size.height, $('r-factor').value));
    return {width: network.width, height: network.height, label: `${network.width}×${network.height}`};
  }
  return {width: chosen.width, height: chosen.height, label: `${chosen.width}×${chosen.height}`};
}

function showInspector() {
  const tensor = tensorSize();
  const mask = model($('model').value).kind === 'rvm'
    ? {width: state.captureWidth || 0, height: state.captureHeight || 0}
    : tensor;
  const view = inspectorView({
    cameraFps: state.cameraRate?.fps() ?? null,
    cameraTimes: state.cameraTimes || [],
    captureWidth: state.captureWidth || 0,
    captureHeight: state.captureHeight || 0,
    latencies: state.latencies || [],
    tensor: tensor.label,
    compositeMs: state.compositeMs || [],
    bytes: textureBytes({
      width: state.captureWidth || 0,
      height: state.captureHeight || 0,
      maskWidth: mask.width,
      maskHeight: mask.height,
      motion: state.motionAllocated,
    }),
    outputFps: state.rate?.fps() ?? null,
    outputTimes: state.outputTimes || [],
    sent: state.sent || 0,
    drops: state.drops || 0,
  });
  for (const name of ['ingest', 'inference', 'composite', 'output']) {
    const meter = view[name];
    $(`${name}-value`).textContent = meter.value;
    $(`${name}-detail`).textContent = meter.detail;
    $(`${name}-spark`).setAttribute('points', meter.points);
  }
  $('budget-value').textContent = view.budget.text;
  $('budget-fill').style.width = `${Math.round(view.budget.fill * 100)}%`;
  $('budget-track').classList.toggle('over', view.budget.over);
}

// What the last composited frame actually did. The line still starts with "mask " so the
// frame-rate check can read the number.
function motionSuffix() {
  const reply = state.lastReply;
  if (!reply) return '';
  if (reply.warped) return ' · mask moving';
  if (reply.held) return ' · frame matched to the mask';
  if (reply.synced) return ' · in sync';
  return '';
}

function showBlur() {
  const blur = $('background').value === 'blur';
  $('blur-label').hidden = $('blur').hidden = !blur;
}

async function applyBackground() {
  showBlur();
  const mode = $('background').value;
  $('colour').disabled = mode !== 'colour';
  if (state.client) await state.client.request('background', {mode, colour: $('colour').value, blur: $('blur').value});
}
$('background').onchange = () => applyBackground().catch(error => status(error.message));
$('colour').oninput = () => applyBackground().catch(error => status(error.message));
$('blur').onchange = () => applyBackground().catch(error => status(error.message));
showBlur();
$('refine').onchange = resetStats;

const LOAD_CONTROLS = ['model', 'camera', 'capture'];

function liveStatus(ratio, ops) {
  const scale = ratio !== undefined && ratio !== null ? ` (R=${formatR(Number(ratio))})` : '';
  status(`Live: ${state.modelName}, ${state.modelNote}${scale}, ${ops} GPU ops.`);
}

// Keep the chosen model when it still matches. Otherwise use RVM, which fits every shape.
function showModels() {
  const size = capture($('capture').value);
  const fitting = modelsFor(size.width, size.height);
  const previous = $('model').value;
  const keep = fitting.some(entry => entry.id === previous) ? previous : (fitting.some(entry => entry.id === 'rvm') ? 'rvm' : fitting[0].id);
  $('model').replaceChildren(...fitting.map(entry => new Option(entry.name, entry.id)));
  $('model').value = keep;
  showScale();
}

// Auto prints the official factor for this capture size. A change while live swaps the network.
function showScale() {
  const rvm = model($('model').value).kind === 'rvm';
  $('r-label').hidden = $('r-factor').hidden = $('scale').hidden = !rvm;
  if (!rvm) return;
  const size = capture($('capture').value);
  const auto = rvmNetworkSize(size.width, size.height);
  const factor = $('r-factor');
  factor.options[0].textContent = `Auto (R=${formatR(auto.ratio)})`;
  const network = rvmNetworkSize(size.width, size.height, rvmScale(size.width, size.height, factor.value));
  $('scale').textContent = `Main network ${network.width} × ${network.height}. You can change it while the camera is running.`;
}

// One change at a time. A newer pick waits and then replaces whatever just finished.
async function pushFactor() {
  showScale();
  if (!state.running || !state.client) return;
  if (state.factorBusy) {
    state.factorDirty = true;
    return;
  }
  state.factorBusy = true;
  try {
    do {
      state.factorDirty = false;
      const choice = $('r-factor').value;
      state.ratioPending = true;
      let updated;
      try {
        updated = await state.client.request('ratio', {ratio: choice}, [], 180000);
      } finally {
        state.ratioPending = false;
        if (state.running) state.liveSince = performance.now();
      }
      if (!state.running || state.factorDirty) continue;
      state.ops = updated.ops;
      // The previous rate was the smaller network. Start the mask and output
      // clocks again so they show this network instead of averaging the old one in.
      state.rate = new FrameRate();
      state.maskRate = new FrameRate();
      state.frames = 0;
      state.masks = 0;
      state.outputTimes = [];
      state.latencies = [];
      state.compositeMs = [];
      state.lastStats = 0;
      liveStatus(updated.ratio, updated.ops);
    } while (state.factorDirty && state.running && state.client);
  } catch (error) {
    if (state.running) status(error.message);
  } finally {
    state.factorBusy = false;
    state.ratioPending = false;
  }
}
$('model').onchange = showScale;
$('capture').onchange = showModels;
$('r-factor').onchange = () => pushFactor().catch(error => status(error.message));
showModels();

function applyLayout() {
  $('tiles').classList.toggle('output', $('layout').value === 'output');
}
$('layout').onchange = applyLayout;
applyLayout();

// What the frame loop is stuck on, or '' while frames and masks keep coming.
function stallReport(now = performance.now()) {
  const ago = t => `${((now - t) / 1000).toFixed(0)} s`;
  const timers = state.tick !== undefined && now - state.tick > 3000 ? [`page timers last ran ${ago(state.tick)} ago`] : [];
  if (state.ratioPending) return timers.join('; ');
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
  // The composited canvas was given to the worker, so the last frame stays painted
  // there. A new canvas is empty, which is the blank tile from before the camera started.
  const previous = $('out');
  const output = document.createElement('canvas');
  output.id = 'out';
  output.width = previous.width;
  output.height = previous.height;
  previous.replaceWith(output);
  resetStats();
  $('capture-chip').textContent = '';
  state.captureWidth = state.captureHeight = 0;
  state.motionAllocated = false;
  showInspector();
  for (const id of LOAD_CONTROLS) $(id).disabled = false;
  $('r-factor').disabled = false;
  status(message);
  idle();
}

// Input names the camera once one is chosen. Output follows the view.
function showOutput() {
  $('output-caption').textContent = $('view').value === 'mask' ? 'Output: Alpha Mask' : 'Output: Composited';
}
$('view').onchange = showOutput;
showOutput();

function showInput(name) {
  const option = $('camera').selectedOptions[0];
  const listed = option && option.value ? option.textContent.trim() : '';
  const real = text => Boolean(text) && !/^Camera \d+$/.test(text);
  const label = real(listed) ? listed : (name || '').trim();
  $('input-caption').textContent = real(label) ? `Input: ${label}` : 'Input';
}

async function listCameras(selected) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
  $('camera').replaceChildren(...devices.map((device, i) => new Option(device.label || `Camera ${i + 1}`, device.deviceId)));
  if (selected) $('camera').value = selected;
  showInput();
}

async function start() {
  if (state.running || state.starting) return;
  state.starting = true;
  const generation = ++state.generation;
  go.disabled = true;
  for (const id of LOAD_CONTROLS) $(id).disabled = true;
  $('r-factor').disabled = true;
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
    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => generation === state.generation && stop('The camera disconnected.'), {once: true});
    showInput(track.label);
    const settings = stream.getVideoTracks()[0].getSettings();
    const capturedWidth = settings.width || video.videoWidth;
    const capturedHeight = settings.height || video.videoHeight;
    $('capture-chip').textContent = capturedWidth && capturedHeight ? `${capturedWidth} × ${capturedHeight}` : '';
    listCameras(settings.deviceId).then(() => showInput(track.label)).catch(() => {});
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
      if (message.compositeMs !== undefined && state.running) noteComposite(message.compositeMs, message.advanced);
      if (message.mask !== undefined && state.running) maskReady(message.mask);
    });
    const choice = $('r-factor').value;
    const prepared = await state.client.request('init', {canvas, model: m.id, ratio: choice}, [canvas], 180000);
    if (generation !== state.generation) return;
    await applyBackground();
    bar(1);
    resetStats();
    state.captureWidth = output.width;
    state.captureHeight = output.height;
    state.motionAllocated = false;
    Object.assign(state, {liveSince: performance.now(), lastCompleted: undefined, lastMask: undefined, lastReply: null, waiting: null, lastWorker: null});
    state.running = true;
    $('r-factor').disabled = false;
    go.disabled = false;
    go.textContent = 'Stop camera';
    Object.assign(state, {modelName: m.name, modelNote: m.note, ops: prepared.ops});
    liveStatus(prepared.ratio, prepared.ops);
    if (m.kind === 'rvm' && $('r-factor').value !== choice) pushFactor().catch(error => status(error.message));
    const frame = (_now, metadata) => {
      if (!state.running || generation !== state.generation) return;
      state.callback = video.requestVideoFrameCallback(frame);
      // The display may call this at 60 Hz for a 30 fps camera. Motion moves the
      // mask onto each new camera frame, and does not invent frames the camera never sent.
      if (!isNewSourceFrame(state.lastPresented, metadata.presentedFrames)) return;
      state.lastPresented = metadata.presentedFrames;
      const now = performance.now();
      state.cameraRate.add(now, metadata.presentedFrames);
      state.cameraTimes.push(now);
      if (state.cameraTimes.length > 60) state.cameraTimes.shift();
      watch();
      if (!state.running || state.inFlight || document.hidden) return;
      state.inFlight = (async () => {
        try {
          state.waiting = {stage: 'createImageBitmap', since: performance.now()};
          const bitmap = await createImageBitmap(video);
          if (!state.running || generation !== state.generation) return bitmap.close();
          state.waiting = {stage: 'frame request', since: performance.now()};
          const moving = $('motion').checked;
          if (moving) state.motionAllocated = true;
          const reply = await state.client.request('frame', {
            bitmap,
            refine: $('refine').checked,
            view: $('view').value,
            motion: moving,
          }, [bitmap], 15000);
          state.lastReply = {...reply, at: performance.now()};
          state.sent++;
          if (reply.dropped) state.drops++;
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
$('camera').onchange = () => (showInput(), state.running && (stop(), start()));
document.addEventListener('visibilitychange', () => !document.hidden && state.running && (resetStats(), (state.liveSince = performance.now())));
window.addEventListener('pagehide', () => stop());

// Which commit and day this page was built from, when the build says so.
async function buildInfo() {
  try {
    const {version, commit, dirty, date} = await (await fetch('build.json', {cache: 'no-store'})).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/pixel-lab/web`;
    $('build').textContent = `${version ? `tract ${version} · ` : ''}${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
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
