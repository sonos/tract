// The microphone goes to an audio worklet, whose blocks go straight to the detector's worker; the
// page draws the scores the worker sends back and flags each crossing of the threshold.
const $ = id => document.getElementById(id);
const go = $('go');
const FRAMES_SHOWN = 1000;
const REFRACTORY_FRAMES = 100;
const RATE = 48000;

const worker = new Worker('worker.js', {type: 'module'});
const state = {running: false, context: null, stream: null, nodes: null, probabilities: [], frame: 0, lastHit: -Infinity, hit: null};
const status = text => ($('status').textContent = text);
const threshold = () => Number($('threshold').value);
const sigmoid = x => 1 / (1 + Math.exp(-x));

function idle() {
  go.disabled = false;
  go.textContent = 'Start listening';
}

function stop(message) {
  state.stream?.getTracks().forEach(track => track.stop());
  state.context?.close();
  Object.assign(state, {running: false, context: null, stream: null, nodes: null});
  status(message ?? 'Stopped.');
  idle();
}

async function start() {
  go.disabled = true;
  go.textContent = 'Starting…';
  try {
    const raw = {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false};
    state.stream = await navigator.mediaDevices.getUserMedia({audio: raw}).catch(error => {
      if (error.name === 'NotAllowedError') throw error;
      return navigator.mediaDevices.getUserMedia({audio: true});
    });
    state.context = new AudioContext({sampleRate: RATE});
    await state.context.audioWorklet.addModule('capture.js');
    const capture = new AudioWorkletNode(state.context, 'capture', {numberOfOutputs: 0});
    const source = state.context.createMediaStreamSource(state.stream);
    source.connect(capture);
    state.nodes = [source, capture];
    const channel = new MessageChannel();
    capture.port.postMessage({port: channel.port1}, [channel.port1]);
    worker.postMessage({port: channel.port2}, [channel.port2]);
    Object.assign(state, {running: true, probabilities: [], frame: 0, lastHit: -Infinity, hit: null});
    $('hits').replaceChildren();
    go.textContent = 'Stop';
    go.disabled = false;
  } catch (error) {
    const reasons = {
      NotAllowedError: 'Microphone access was refused.',
      NotFoundError: 'No microphone found. If there is one, the system may be keeping it from this browser: on macOS, see System Settings › Privacy & Security › Microphone.',
      NotSupportedError: 'This browser cannot capture the microphone at 48 kHz.',
    };
    stop(reasons[error.name] || `Error: ${error.message || error}`);
  }
}

function record(probability) {
  const frame = state.frame++;
  state.probabilities.push(probability);
  if (state.probabilities.length > FRAMES_SHOWN) state.probabilities.shift();
  if (state.hit && probability > state.hit.peak) {
    state.hit.peak = probability;
    state.hit.item.textContent = `${state.hit.at} · ${(100 * probability).toFixed(1)} %`;
  }
  if (probability < threshold()) state.hit = null;
  else if (!state.hit && frame - state.lastHit > REFRACTORY_FRAMES) {
    state.lastHit = frame;
    const item = document.createElement('li');
    state.hit = {peak: 0, at: new Date().toLocaleTimeString(), item};
    $('hits').prepend(item);
    $('wake').textContent = 'Hey Snips!';
    $('wake').classList.add('hit');
    setTimeout(() => {
      $('wake').classList.remove('hit');
      $('wake').textContent = 'Say “Hey Snips”';
    }, 1200);
  }
}

function draw() {
  const canvas = $('trace');
  const width = canvas.clientWidth * devicePixelRatio, height = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== width || canvas.height !== height) Object.assign(canvas, {width, height});
  const g = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, width, height);
  const y = p => height - 6 * devicePixelRatio - p * (height - 12 * devicePixelRatio);
  g.setLineDash([6 * devicePixelRatio, 6 * devicePixelRatio]);
  g.strokeStyle = style.getPropertyValue('--muted');
  g.lineWidth = devicePixelRatio;
  g.beginPath();
  g.moveTo(0, y(threshold()));
  g.lineTo(width, y(threshold()));
  g.stroke();
  g.setLineDash([]);
  g.strokeStyle = style.getPropertyValue('--accent');
  g.lineWidth = 2 * devicePixelRatio;
  g.beginPath();
  const offset = FRAMES_SHOWN - state.probabilities.length;
  state.probabilities.forEach((p, i) => g[i ? 'lineTo' : 'moveTo'](((offset + i) / (FRAMES_SHOWN - 1)) * width, y(p)));
  g.stroke();
  requestAnimationFrame(draw);
}

worker.onmessage = ({data}) => {
  if (data.status) status(data.status);
  if (data.error) stop(`Error: ${data.error}`);
  if (data.ready) status('Listening.');
  if (data.scores && state.running) {
    data.scores.forEach(score => record(sigmoid(score)));
    const audio = (1000 * data.samples) / RATE;
    $('stats').textContent = `tract runs at ${(100 * data.busy / audio).toFixed(1)} % of real time on this CPU`;
  }
};

go.onclick = () => (state.running ? stop() : start());
const showThreshold = () => ($('threshold-value').textContent = `${(100 * threshold()).toFixed(1)} %`);
$('threshold').oninput = showThreshold;
window.addEventListener('pagehide', () => stop());

// Which commit and day this page was built from, when the build says so.
async function buildInfo() {
  try {
    const {commit, dirty, date} = await (await fetch('build.json')).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/hey-snips/web`;
    $('build').textContent = `${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
  } catch {}
}

function check() {
  if (!window.isSecureContext) return 'The microphone needs a secure context: HTTPS, or localhost.';
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser exposes no microphone here.';
  if (!window.AudioWorkletNode) return 'This browser lacks AudioWorklet.';
  return null;
}

showThreshold();
buildInfo();
requestAnimationFrame(draw);
const problem = check();
if (problem) {
  status(problem);
  go.textContent = 'Unsupported';
} else {
  status('Ready. Nothing is recorded or sent anywhere.');
  idle();
}
