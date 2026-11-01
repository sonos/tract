import { TIERS, megabytes, weightBytes } from './models.mjs';

const $ = (id) => document.getElementById(id);
const status = (text, warn = false) => {
  $('status').textContent = text;
  $('status').classList.toggle('warn', warn);
};
const bar = (fraction) => {
  const span = $('bar').firstElementChild;
  $('bar').classList.remove('busy');
  span.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
};
const barBusy = () => {
  $('bar').classList.add('busy');
  $('bar').firstElementChild.style.width = '';
};

// Download has bytes, detection has none, and recognition has lines. Each
// phase gets its own clock so a one-time graph prepare is not reported as
// zero lines per second.
let pace = null;
let paceTimer = 0;
let jobStarted = 0;

function seconds(value) {
  const whole = Math.max(0, Math.round(value));
  if (whole < 60) return `${whole} s`;
  return `${Math.floor(whole / 60)} m ${String(whole % 60).padStart(2, '0')} s`;
}

function stopPace() {
  if (paceTimer) clearInterval(paceTimer);
  paceTimer = 0;
  pace = null;
  jobStarted = 0;
}

function paceSuffix() {
  if (!pace) return '';
  const elapsed = (performance.now() - pace.started) / 1000;
  if (pace.kind === 'bytes' && pace.got > 0 && elapsed >= 0.4) {
    const rate = pace.got / (1024 * 1024) / elapsed;
    return rate >= 0.1 ? ` · ${rate.toFixed(1)} MB/s` : '';
  }
  if (pace.kind === 'wait' && elapsed >= 1) return ` · ${seconds(elapsed)}`;
  if (pace.kind === 'lines' && pace.done > 0 && elapsed >= 0.4) {
    const rate = pace.done / elapsed;
    const left = Math.max(0, pace.total - pace.done) / rate;
    const remain = pace.done < pace.total ? ` · ~${seconds(left)} left` : '';
    return ` · ${rate.toFixed(1)} lines/s${remain}`;
  }
  return '';
}

function showPace(base) {
  pace.base = base;
  status(`${base}${paceSuffix()}`);
  if (!paceTimer) paceTimer = setInterval(() => pace && status(`${pace.base}${paceSuffix()}`), 500);
}

function beginPace(kind, base) {
  pace = { kind, label: base, base, started: performance.now(), got: 0, done: 0, total: 0 };
  showPace(base);
}

const modelSelect = $('model');
for (const candidate of TIERS) {
  const option = new Option(`${candidate.name} · ${megabytes(weightBytes(candidate))} MB`, candidate.id);
  option.title = candidate.note;
  modelSelect.add(option);
}

const view = $('view');
const video = $('video');
const ctx = view.getContext('2d');
const state = { rgb: null, width: 0, height: 0, lines: [], stream: null, job: 0, hover: -1, phase: 'image' };
let request = 0;

let worker;
try {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
} catch {
  status('This browser cannot start the reader.', true);
}

worker?.addEventListener('error', (event) => {
  state.phase = 'image';
  stopPace();
  status(event.message || 'The reader stopped.', true);
});

worker?.addEventListener('message', ({ data }) => {
  if (data.model) {
    if (data.model !== modelSelect.value) return;
    state.phase = 'model';
    if (data.progress) {
      const { label, got, total } = data.progress;
      const amount = total ? `${megabytes(got)} / ${megabytes(total)} MB` : `${megabytes(got)} MB`;
      const text = `Downloading ${label}, ${amount}`;
      if (!pace || pace.kind !== 'bytes' || pace.label !== label || got < pace.got) beginPace('bytes', label);
      pace.label = label;
      pace.got = got;
      if (total) bar(got / total);
      showPace(text);
    } else if (data.status) {
      beginPace('wait', data.status);
    }
    return;
  }
  if (data.accepted) {
    state.job = data.job;
    return;
  }
  if (data.job !== state.job) return;
  state.phase = 'image';
  if (data.error) {
    stopPace();
    status(data.error, true);
    bar(0);
    return;
  }
  if (data.read) {
    bar(data.read.done / Math.max(1, data.read.total));
    if (!pace || pace.kind !== 'lines') beginPace('lines', data.status || 'Reading');
    pace.done = data.read.done;
    pace.total = data.read.total;
    showPace(data.status || pace.base);
  } else if (data.status?.startsWith('Finding') || data.status?.startsWith('Preparing')) {
    barBusy();
    if (!jobStarted) jobStarted = performance.now();
    beginPace('wait', data.status);
  } else if (data.status) {
    status(data.status);
  }
  if (data.lines) {
    const elapsed = jobStarted ? (performance.now() - jobStarted) / 1000 : 0;
    stopPace();
    if (data.status && elapsed >= 1) status(`${data.status.replace(/\.$/, '')} · ${seconds(elapsed)}.`);
    state.lines = data.lines;
    state.hover = -1;
    bar(1);
    draw();
    showLines();
  }
});

function showLines() {
  const list = $('lines');
  list.replaceChildren();
  $('copy').disabled = state.lines.length === 0;
  if (!state.lines.length) {
    const item = document.createElement('li');
    item.className = 'empty';
    item.textContent = 'Nothing read yet.';
    list.append(item);
    return;
  }
  state.lines.forEach((line, index) => {
    const item = document.createElement('li');
    item.append(document.createTextNode(line.text));
    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = `${Math.round(line.score * 100)}%`;
    item.append(score);
    item.addEventListener('pointerenter', () => {
      state.hover = index;
      draw();
    });
    item.addEventListener('pointerleave', () => {
      state.hover = -1;
      draw();
    });
    list.append(item);
  });
}

function draw() {
  if (!state.width || !state.rgb) return;
  const pixels = new Uint8ClampedArray(state.width * state.height * 4);
  for (let source = 0, dest = 0; source < state.rgb.length; source += 3, dest += 4) {
    pixels[dest] = state.rgb[source];
    pixels[dest + 1] = state.rgb[source + 1];
    pixels[dest + 2] = state.rgb[source + 2];
    pixels[dest + 3] = 255;
  }
  ctx.putImageData(new ImageData(pixels, state.width, state.height), 0, 0);
  for (const [index, line] of state.lines.entries()) {
    ctx.beginPath();
    for (const quad of line.quads) {
      quad.forEach(([x, y], point) => (point ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
    }
    ctx.lineWidth = Math.max(2, state.width / 400);
    ctx.strokeStyle = index === state.hover ? '#e08a52' : '#2f6fdd';
    ctx.stroke();
  }
}

function useImage(source) {
  const longest = Math.max(source.width, source.height);
  const scale = Math.min(1, 2000 / longest);
  view.width = Math.max(1, Math.round(source.width * scale));
  view.height = Math.max(1, Math.round(source.height * scale));
  ctx.drawImage(source, 0, 0, view.width, view.height);
  const pixels = ctx.getImageData(0, 0, view.width, view.height).data;
  source.close?.();
  const rgb = new Uint8Array(view.width * view.height * 3);
  for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
    rgb[j] = pixels[i];
    rgb[j + 1] = pixels[i + 1];
    rgb[j + 2] = pixels[i + 2];
  }
  state.rgb = rgb;
  state.width = view.width;
  state.height = view.height;
  state.lines = [];
  state.hover = -1;
  showLines();
  video.hidden = true;
  view.hidden = false;
  $('snap').hidden = true;
  draw();
  recognize();
}

function recognize() {
  if (!worker || !state.rgb) return;
  // A download already on screen belongs to this model. Leave it up while the
  // picture changes, or the bar resets and the reader looks idle.
  if (state.phase !== 'model') {
    barBusy();
    jobStarted = performance.now();
    beginPace('wait', 'Finding text…');
  }
  const rgb = state.rgb.slice();
  worker.postMessage(
    {
      tier: modelSelect.value,
      runtime: $('runtime').value,
      rgb,
      width: state.width,
      height: state.height,
    },
    [rgb.buffer],
  );
}

async function loadSample() {
  const token = ++request;
  stopCamera();
  status('Loading the sample…');
  const image = await createImageBitmap(await (await fetch('sample.png')).blob());
  if (token !== request) return;
  useImage(image);
}

$('sample').onclick = () => loadSample().catch((error) => status(error.message, true));
$('upload').onclick = () => $('file').click();
$('file').onchange = async () => {
  const file = $('file').files?.[0];
  $('file').value = '';
  if (!file) return;
  const token = ++request;
  stopCamera();
  try {
    const image = await createImageBitmap(file);
    if (token !== request) {
      image.close?.();
      return;
    }
    useImage(image);
  } catch {
    if (token === request) status('That file is not a picture this browser can read.', true);
  }
};

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    status('This browser exposes no camera here.', true);
    return;
  }
  if (!window.isSecureContext) {
    status('The camera needs HTTPS, or localhost.', true);
    return;
  }
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } },
      audio: false,
    });
  } catch {
    status('The camera is not available.', true);
    return;
  }
  video.srcObject = state.stream;
  video.hidden = false;
  view.hidden = true;
  $('snap').hidden = false;
  try {
    await video.play();
  } catch {
    stopCamera();
    status('The camera preview could not start.', true);
    return;
  }
  status('Take a snapshot to read it.');
}

function stopCamera() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.srcObject = null;
  video.hidden = true;
  view.hidden = false;
  $('snap').hidden = true;
}

$('camera').onclick = () => startCamera();

let snapping = false;
$('snap').onclick = async () => {
  if (snapping) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
    status('The camera has no frame yet.', true);
    return;
  }
  snapping = true;
  $('snap').disabled = true;
  try {
    // The live frame can sit in the camera texture. Copy it before the track
    // stops, or the picture handed to the reader is black.
    const frame = await grabCameraFrame();
    request += 1;
    stopCamera();
    useImage(frame);
  } catch {
    status('The snapshot could not be taken.', true);
  } finally {
    snapping = false;
    $('snap').disabled = false;
  }
};

async function grabCameraFrame() {
  try {
    return await createImageBitmap(video);
  } catch {
    const frame = document.createElement('canvas');
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
    const frameCtx = frame.getContext('2d', { willReadFrequently: true });
    frameCtx.drawImage(video, 0, 0);
    frameCtx.getImageData(0, 0, frame.width, frame.height);
    return frame;
  }
}

modelSelect.onchange = () => {
  if (state.width) recognize();
};
const runtimeSelect = $('runtime');
if (!navigator.gpu) {
  for (const option of runtimeSelect.options) {
    if (option.value === 'wgpu') option.disabled = true;
  }
  runtimeSelect.value = 'cpu';
}
runtimeSelect.onchange = () => {
  if (state.width) recognize();
};

$('copy').onclick = async () => {
  const text = state.lines.map((line) => line.text).join('\n');
  try {
    await navigator.clipboard.writeText(text);
    stopPace();
    status('Copied.');
  } catch {
    stopPace();
    status('Copy was blocked by the browser.', true);
  }
};

window.addEventListener('pagehide', stopCamera);

async function buildInfo() {
  try {
    const { commit, dirty, date } = await (await fetch('build.json')).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/ppocr/web`;
    $('build').textContent = `${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
  } catch {
    // The source tree has no build.json; the link stays on main.
  }
}

buildInfo();
loadSample().catch((error) => status(error.message, true));
