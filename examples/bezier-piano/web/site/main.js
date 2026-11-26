const stamp = new URL(import.meta.url).searchParams.get('stamp') || '';
const stamped = file => import(`./${file}${stamp ? `?stamp=${stamp}` : ''}`);
const {FrameRate} = await stamped('frame-rate.mjs');
const {emptyPump, rememberFrame, startSend, finishSend} = await stamped('frame-pump.mjs');
const {HandClient} = await stamped('hands.mjs');
const {drawFrame, sparkline} = await stamped('render.mjs');
const {Piano} = await stamped('piano.mjs');
const {
  FINGERTIPS, PINCH_ON, PINCH_OFF, SCALES, scaleById, clamp, sampleCurve, firstHit,
  pinchDistance, pinchPoint, nearestControl, velocityGain, allowPluck, midiFor,
  noteName, layoutStrings, pathLength, fitCubic, assignHands,
} = await stamped('strings.mjs');

const $ = id => document.getElementById(id);
const status = text => { $('status').textContent = text; };
const bar = fraction => { $('bar').style.width = `${Math.round(fraction * 100)}%`; };
const go = $('go');
const video = $('camera-feed');
const canvas = $('out');
const ctx = canvas.getContext('2d', {alpha: true, desynchronized: true});

const state = {
  running: false,
  starting: false,
  generation: 0,
  callback: null,
  anim: 0,
  pump: emptyPump(),
  presentedFallback: 0,
  lastTimestamp: 0,
  rate: new FrameRate(),
  cameraRate: new FrameRate(),
  frames: 0,
  infer: [],
  inferSpark: [],
  cameraSpark: [],
  lastStats: 0,
  strings: [],
  colorSeq: 0,
  tracks: [],
  nextId: 1,
  trails: [],
  draft: null,
  draftOwner: null,
  activeControl: null,
  plucked: new Map(),
  lastHandAt: 0,
  lastRender: 0,
  lastNote: '',
  backend: '',
  piano: null,
  loaded: null,
  connections: [],
  captureWidth: 0,
  captureHeight: 0,
  switching: false,
  showLandmarks: false,
  drawMode: false,
  runningVideo: false,
};

function volumeValue() {
  return Number($('volume').value) / 100;
}

function layOut(count) {
  state.strings = layoutStrings(count).map((string, i) => ({
    p: string.p.map(point => ({...point})),
    color: i,
    energy: 0,
    ripples: [],
  }));
  state.colorSeq = count;
  $('strings-value').textContent = String(state.strings.length);
  requestDraw();
}

for (const scale of SCALES) $('scale').append(new Option(scale.name, scale.id));
$('scale').value = 'major-pentatonic';
layOut(Number($('strings').value));

function idle(label = 'Start camera') {
  go.disabled = false;
  go.textContent = label;
}

function median(values) {
  if (!values.length) return null;
  return [...values].sort((a, b) => a - b)[values.length >> 1];
}

function showStats() {
  const now = performance.now();
  if (now - state.lastStats < 250) return;
  state.lastStats = now;
  const fps = state.rate.fps();
  const camera = state.cameraRate.fps();
  const inf = median(state.infer);
  $('out-fps').textContent = fps === null ? '' : `${fps.toFixed(1)} fps`;
  $('backend-chip').textContent = state.running && state.backend ? state.backend : '';
  const hands = state.tracks.length;
  $('stats').textContent = state.running
    ? `${hands} ${hands === 1 ? 'hand' : 'hands'} · camera ${camera === null ? '—' : camera.toFixed(1)} fps`
    : '';
  $('camera-value').textContent = camera === null ? '—' : camera.toFixed(1);
  $('camera-detail').textContent = state.running && state.captureWidth
    ? `${state.captureWidth} × ${state.captureHeight}`
    : 'Waiting for the camera';
  $('inference-value').textContent = inf === null ? '—' : inf.toFixed(0);
  $('inference-detail').textContent = inf === null ? 'Waiting for a frame' : 'median milliseconds';
  $('camera-spark').setAttribute('points', sparkline(state.cameraSpark));
  $('inference-spark').setAttribute('points', sparkline(state.inferSpark));
  if (!state.running) return;
  const fpsText = fps === null ? '— fps' : `${fps.toFixed(1)} fps`;
  const infText = inf === null ? '— ms' : `${inf.toFixed(0)} ms`;
  const note = state.lastNote ? ` · ${state.lastNote}` : '';
  status(`Live · ${state.backend} · ${fpsText} · ${infText} inference${note}`);
}

function requestDraw() {
  if (state.anim) return;
  const loop = now => {
    drawFrame(ctx, state, now);
    if (state.running) {
      state.frames++;
      state.rate.add(now, state.frames);
      showStats();
    }
    const alive = state.running
      || state.trails.length > 0
      || state.draft
      || state.strings.some(string => string.energy > 0.02 || string.ripples.length);
    state.anim = alive ? requestAnimationFrame(loop) : 0;
  };
  state.anim = requestAnimationFrame(loop);
}

function samplesFor(string) {
  const key = string.p.map(point => `${point.x.toFixed(4)},${point.y.toFixed(4)}`).join(';');
  if (string.sampleKey !== key) {
    string.samples = sampleCurve(string.p, 64);
    string.sampleKey = key;
  }
  return string.samples;
}

function pluck(index, t, gain) {
  const string = state.strings[index];
  if (!string || !state.piano) return;
  string.energy = Math.max(string.energy, 0.45 + gain * 0.55);
  string.ripples.push({t, born: performance.now()});
  if (string.ripples.length > 5) string.ripples.shift();
  const scale = scaleById($('scale').value);
  const midi = midiFor(index, state.strings.length, t, scale.degrees);
  state.lastNote = noteName(midi);
  state.piano.play(midi, gain);
  requestDraw();
}

function commitDraft() {
  const points = state.draft;
  state.draft = null;
  state.draftOwner = null;
  if (!points || points.length < 4 || pathLength(points) < 0.08) return;
  const fit = fitCubic(points);
  if (!fit) return;
  const shaped = fit.map(point => ({x: clamp(point.x, 0.02, 0.98), y: clamp(point.y, 0.02, 0.98)}));
  if (state.strings.length >= 12) {
    const last = state.strings[state.strings.length - 1];
    last.p = shaped;
    last.sampleKey = '';
    last.energy = 0.4;
  } else {
    state.strings.push({p: shaped, color: state.colorSeq++, energy: 0.4, ripples: []});
  }
  $('strings').value = String(state.strings.length);
  $('strings-value').textContent = String(state.strings.length);
}

function considerPlucks(prevLm, currLm, id, now, dt, skip) {
  for (const tip of FINGERTIPS) {
    if (skip.has(tip)) continue;
    const a = prevLm[tip];
    const b = currLm[tip];
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    if (dist < 0.003) continue;
    for (let s = 0; s < state.strings.length; s++) {
      const hit = firstHit(a, b, samplesFor(state.strings[s]));
      if (!hit) continue;
      if (!allowPluck(state.plucked, `${id}:${tip}:${s}`, now, 110)) continue;
      pluck(s, hit.u, velocityGain(dist / dt));
    }
  }
}

function onHands(hands, now) {
  const rawDt = state.lastHandAt ? (now - state.lastHandAt) / 1000 : 1 / 30;
  const dt = clamp(rawDt, 1 / 120, 0.1);
  state.lastHandAt = now;
  const copied = hands.map(hand => hand.map(point => ({x: point.x, y: point.y})));
  const links = assignHands(state.tracks.map(track => track.landmarks[0]), copied);
  const next = [];
  state.activeControl = null;
  for (const link of links) {
    const curr = copied[link.index];
    const old = link.prev >= 0 ? state.tracks[link.prev] : null;
    const track = {
      id: old ? old.id : state.nextId++,
      landmarks: curr,
      pinch: Boolean(old?.pinch),
      drag: old?.drag || null,
      drafting: Boolean(old?.drafting),
    };
    const was = track.pinch;
    const dist = pinchDistance(curr);
    if (!track.pinch && dist < PINCH_ON) track.pinch = true;
    else if (track.pinch && dist > PINCH_OFF) track.pinch = false;
    const point = pinchPoint(curr);
    if (track.pinch && !was) {
      if (state.drawMode && (state.draftOwner == null || state.draftOwner === track.id)) {
        track.drafting = true;
        track.drag = null;
        state.draftOwner = track.id;
        state.draft = [point];
      } else if (!state.drawMode) {
        const hit = nearestControl(state.strings, point, 0.09);
        track.drag = hit ? {
          s: hit.s,
          i: hit.i,
          dx: state.strings[hit.s].p[hit.i].x - point.x,
          dy: state.strings[hit.s].p[hit.i].y - point.y,
        } : null;
        track.drafting = false;
      }
    }
    if (track.pinch && track.drag && state.strings[track.drag.s]) {
      const control = state.strings[track.drag.s].p[track.drag.i];
      control.x = clamp(point.x + track.drag.dx, 0.02, 0.98);
      control.y = clamp(point.y + track.drag.dy, 0.02, 0.98);
      state.strings[track.drag.s].sampleKey = '';
      state.activeControl = {s: track.drag.s, i: track.drag.i};
    }
    if (track.pinch && track.drafting && state.draftOwner === track.id) {
      const last = state.draft[state.draft.length - 1];
      if (!last || Math.hypot(point.x - last.x, point.y - last.y) > 0.008) state.draft.push({...point});
    }
    if (was && !track.pinch) {
      if (track.drafting && state.draftOwner === track.id) commitDraft();
      track.drag = null;
      track.drafting = false;
    }
    if (old && !(track.pinch && track.drafting)) {
      const skip = track.pinch ? new Set([4, 8]) : new Set();
      considerPlucks(old.landmarks, curr, track.id, now, dt, skip);
    }
    for (const tip of FINGERTIPS) {
      state.trails.push({
        id: `${track.id}:${tip}`,
        x: curr[tip].x,
        y: curr[tip].y,
        born: now,
        color: tip === 4 || tip === 8 ? '#3ddc97' : '#6c9cf0',
      });
    }
    next.push(track);
  }
  for (const old of state.tracks) {
    if (!next.some(track => track.id === old.id) && old.drafting && state.draftOwner === old.id) commitDraft();
  }
  state.tracks = next;
  if (state.trails.length > 280) state.trails.splice(0, state.trails.length - 280);
  if (state.plucked.size > 400) {
    for (const [key, at] of state.plucked) if (now - at > 500) state.plucked.delete(key);
  }
}

function fitStage() {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  $('stage').style.aspectRatio = `${w} / ${h}`;
  state.captureWidth = video.videoWidth || 0;
  state.captureHeight = video.videoHeight || 0;
}

function noteInference(ms) {
  state.infer.push(ms);
  if (state.infer.length > 60) state.infer.shift();
  state.inferSpark.push(ms);
  if (state.inferSpark.length > 48) state.inferSpark.shift();
}

function resetRates() {
  state.rate = new FrameRate();
  state.cameraRate = new FrameRate();
  state.frames = 0;
  state.infer = [];
  state.cameraSpark = [];
  state.lastCameraAt = 0;
  state.lastStats = 0;
}

const LOAD_CONTROLS = ['strings', 'scale', 'volume', 'mirror', 'landmarks', 'draw'];

function stop(message = 'Camera stopped.') {
  state.generation++;
  state.running = false;
  state.runningVideo = false;
  state.switching = false;
  if (state.callback !== null) video.cancelVideoFrameCallback?.(state.callback);
  state.callback = null;
  state.pump = emptyPump();
  state.stream?.getTracks().forEach(track => track.stop());
  state.stream = null;
  video.pause();
  video.srcObject = null;
  $('stage').classList.remove('live');
  state.tracks = [];
  state.trails = [];
  state.activeControl = null;
  resetRates();
  $('out-fps').textContent = '';
  $('backend-chip').textContent = '';
  showStats();
  for (const id of LOAD_CONTROLS) $(id).disabled = false;
  status(message);
  idle();
  requestDraw();
}

async function openCamera() {
  const tries = [
    {audio: false, video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}, facingMode: 'user'}},
    {audio: false, video: {facingMode: 'user'}},
    {audio: false, video: true},
  ];
  let last;
  for (const constraints of tries) {
    try { return await navigator.mediaDevices.getUserMedia(constraints); }
    catch (error) {
      last = error;
      if (error.name === 'NotAllowedError' || error.name === 'NotFoundError') throw error;
    }
  }
  throw last;
}

async function start() {
  if (state.running || state.starting || !state.loaded) return;
  state.starting = true;
  const generation = ++state.generation;
  state.piano = state.piano || new Piano();
  // Started inside the click, so the gesture counts. Not awaited: without a gesture the
  // promise never settles, and every note asks for the context again anyway.
  state.piano.resume().catch(() => {});
  state.piano.setVolume(volumeValue());
  go.disabled = true;
  go.textContent = 'Starting…';
  for (const id of LOAD_CONTROLS) $(id).disabled = true;
  try {
    status('Asking for the camera…');
    const streamPromise = openCamera();
    const stream = await streamPromise;
    if (generation !== state.generation) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    if (generation !== state.generation) return;
    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => generation === state.generation && stop('The camera disconnected.'), {once: true});
    fitStage();
    $('stage').classList.add('live');
    state.runningVideo = true;
    state.tracks = [];
    state.lastHandAt = 0;
    state.lastNote = '';
    state.plucked.clear();
    resetRates();
    state.pump = emptyPump();
    state.presentedFallback = 0;
    await state.hands.request('reset');
    if (generation !== state.generation) return;
    state.running = true;
    go.disabled = false;
    go.textContent = 'Stop camera';
    // Sliders stay live. The string count and the scale change the instrument
    // while the camera is open; they do not restart it.
    for (const id of LOAD_CONTROLS) $(id).disabled = false;
    status(`Live · ${state.backend} · waiting for frames`);
    // A frame that arrives while detectForVideo is still running is kept.
    // The send that finishes starts that frame at once. Drawing does not wait.
    const launchSend = () => {
      state.inFlight = (async () => {
        let bitmap;
        try {
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
          bitmap = await createImageBitmap(video);
          if (!state.running || generation !== state.generation) return bitmap.close();
          const reply = await state.hands.request('frame', bitmap, 15000);
          bitmap = null;
          if (!state.running || generation !== state.generation) return;
          noteInference(reply.ms);
          onHands(reply.landmarks || [], performance.now());
        } catch (error) {
          try { bitmap?.close(); } catch { /* already transferred */ }
          if (generation === state.generation) stop(`Stopped: ${error.message || error}`);
        } finally {
          if (generation !== state.generation) return;
          state.inFlight = null;
          state.pump = finishSend(state.pump);
          if (document.hidden || !state.running) return;
          const step = startSend(state.pump);
          state.pump = step.pump;
          if (step.start) launchSend();
        }
      })();
    };
    const frame = (_now, metadata) => {
      if (!state.running || generation !== state.generation) return;
      state.callback = video.requestVideoFrameCallback(frame);
      const presented = Number.isFinite(metadata?.presentedFrames) ? metadata.presentedFrames : state.presentedFallback++;
      const noted = rememberFrame(state.pump, presented);
      state.pump = noted.pump;
      if (!noted.fresh) return;
      const stampNow = performance.now();
      state.cameraRate.add(stampNow, presented);
      if (state.lastCameraAt) {
        state.cameraSpark.push(stampNow - state.lastCameraAt);
        if (state.cameraSpark.length > 48) state.cameraSpark.shift();
      }
      state.lastCameraAt = stampNow;
      if (document.hidden) return;
      const step = startSend(state.pump);
      state.pump = step.pump;
      if (step.start) launchSend();
    };
    state.callback = video.requestVideoFrameCallback(frame);
    requestDraw();
  } catch (error) {
    if (generation !== state.generation) return;
    const reasons = {
      NotAllowedError: 'Camera access was denied. Allow it in the site settings, then retry.',
      NotFoundError: 'No camera found.',
      NotReadableError: 'The camera is busy in another application.',
      OverconstrainedError: 'This camera cannot capture a frame. Retry, or try another browser.',
      SecurityError: 'The camera needs a secure context: HTTPS, or localhost.',
    };
    stop(reasons[error.name] || `Error: ${error.message || error}`);
  } finally {
    state.starting = false;
  }
}

go.onclick = () => (state.running ? stop() : start());

$('strings').oninput = () => {
  $('strings-value').textContent = $('strings').value;
  layOut(Number($('strings').value));
};
$('volume').oninput = () => {
  $('volume-value').textContent = $('volume').value;
  state.piano?.setVolume(volumeValue());
};
$('mirror').onchange = () => {
  $('stage').classList.toggle('mirror', $('mirror').checked);
};
$('landmarks').onchange = () => {
  state.showLandmarks = $('landmarks').checked;
  requestDraw();
};
$('draw').onchange = () => {
  state.drawMode = $('draw').checked;
  if (state.draftOwner != null) commitDraft();
  for (const track of state.tracks) {
    track.drag = null;
    track.drafting = false;
  }
  requestDraw();
};
$('mirror').onchange();
state.showLandmarks = $('landmarks').checked;
state.drawMode = $('draw').checked;
$('volume-value').textContent = $('volume').value;

document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  if (state.running) resetRates();
  // Coming back to the tab is a good moment to wake a context the browser
  // suspended while it was hidden.
  if (state.piano?.wanted && state.piano.ctx.state !== 'running') state.piano.kick();
});
window.addEventListener('pagehide', () => stop());

function check() {
  if (!window.isSecureContext) return 'The camera needs a secure context: HTTPS, or localhost.';
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser exposes no camera here.';
  if (!HTMLVideoElement.prototype.requestVideoFrameCallback) return 'This browser lacks requestVideoFrameCallback.';
  if (!window.Worker) return 'This browser has no workers.';
  if (!window.AudioContext && !window.webkitAudioContext) return 'This browser has no WebAudio.';
  return null;
}

window.bezierPiano = {
  status: () => $('status').textContent,
  backend: () => state.backend,
  notes: () => state.piano?.played ?? 0,
  audio: () => state.piano?.stats() ?? null,
  fps: () => state.rate?.fps() ?? null,
  inferenceMs: () => median(state.infer),
  strings: () => state.strings.length,
  // One synthetic cross of the middle string, so a headless run can hear the path.
  pluckCenter: () => {
    if (!state.piano) return false;
    pluck(Math.floor(state.strings.length / 2), 0.5, 0.9);
    return true;
  },
};

// ?backend=cpu or ?backend=wgpu in the URL wins, then the last choice made here.
function wantedBackend() {
  const asked = new URL(location.href).searchParams.get('backend');
  if (asked === 'cpu' || asked === 'wgpu') return asked;
  const saved = localStorage.getItem('bezier-piano-backend');
  if (saved === 'cpu' || saved === 'wgpu') return saved;
  return 'wgpu';
}

function showBackend(loaded) {
  state.backend = loaded.backend;
  $('backend').value = loaded.name;
  $('backend').disabled = false;
  if (loaded.fallback) $('backend').querySelector('option[value="wgpu"]').disabled = true;
  showStats();
}

$('backend').onchange = async () => {
  const wanted = $('backend').value;
  $('backend').disabled = true;
  const before = $('status').textContent;
  status(wanted === 'wgpu' ? 'Switching to WebGPU…' : 'Switching to the CPU…');
  try {
    const loaded = await state.hands.request('backend', null, 120000, {backend: wanted});
    localStorage.setItem('bezier-piano-backend', wanted);
    state.infer = [];
    state.inferSpark = [];
    showBackend(loaded);
    if (!state.running) status(`Ready on ${loaded.backend}.`);
  } catch (error) {
    status(`${error.message || error} Staying on ${state.backend}.`);
    $('backend').value = state.backend.includes('WebGPU') ? 'wgpu' : 'cpu';
    $('backend').disabled = false;
    if (!state.running && !before.startsWith('Switching')) setTimeout(() => status(before), 4000);
  }
};

async function showBuild() {
  try {
    const {version, commit, dirty, date} = await (await fetch('build.json')).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/bezier-piano/web`;
    $('build').textContent = `${version ? `tract ${version} · ` : ''}${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
  } catch { /* a local copy without build.json */ }
}
showBuild();

const problem = check();
if (problem) {
  status(problem);
  go.textContent = 'Unsupported';
} else {
  try {
    state.hands = HandClient((text, fraction) => {
      status(text);
      if (fraction !== undefined) bar(fraction);
    });
    $('backend').disabled = true;
    state.loaded = await state.hands.request('init', null, 300000, {backend: wantedBackend()});
    state.connections = state.loaded.connections || [];
    showBackend(state.loaded);
    const note = state.loaded.fallback ? ` WebGPU did not start (${state.loaded.fallback}), so this is the CPU.` : '';
    status(`Ready. The camera and the sound start together.${note}`);
    bar(1);
    idle();
  } catch (error) {
    console.error(error);
    status(error.message || 'The hand tracker did not start.');
    go.textContent = 'Unavailable';
  }
}

requestDraw();
