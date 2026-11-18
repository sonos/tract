import { RELEASES, SAMPLES, VOICES, graphBytes, languageName, licenseUrl, megabytes, release } from './models.mjs';

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

for (const candidate of RELEASES) {
  const option = new Option(`${candidate.name} · ${candidate.note} · ${megabytes(graphBytes(candidate))} MB`, candidate.id);
  $('model').add(option);
}
for (const voice of VOICES) {
  $('voice').add(new Option(`${voice[0] === 'F' ? 'Female' : 'Male'} ${voice.slice(1)}`, voice));
}
if (!navigator.gpu) $('runtime').value = 'cpu';

let sampleShown = true;
function fillLanguages() {
  const model = release($('model').value);
  const previous = $('lang').value || 'en';
  $('lang').replaceChildren(...model.languages.map((code) => new Option(languageName(code), code)));
  $('lang').value = model.languages.includes(previous) ? previous : 'en';
  $('card').href = `https://huggingface.co/${model.repo}`;
  $('license').href = licenseUrl(model);
}
function fillSample() {
  if (sampleShown) $('text').value = SAMPLES[$('lang').value] || SAMPLES.en;
}
fillLanguages();
fillSample();
$('model').addEventListener('change', () => {
  fillLanguages();
  fillSample();
});
$('lang').addEventListener('change', fillSample);
$('text').addEventListener('input', () => {
  sampleShown = false;
});

let worker;
try {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
} catch {
  status('This browser cannot start the speaker.', true);
}

let audio;
let ticket = 0;
let playhead = 0;
let pieces = [];
let rate = 44100;
let spoken = { audio: 0, took: 0 };
const sources = new Set();

function stopPlayback() {
  for (const source of sources) source.stop();
  sources.clear();
  playhead = 0;
}

function play(samples) {
  audio ??= new AudioContext();
  const buffer = audio.createBuffer(1, samples.length, rate);
  buffer.copyToChannel(samples, 0);
  const source = audio.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.destination);
  const at = Math.max(audio.currentTime + 0.05, playhead);
  source.start(at);
  playhead = at + buffer.duration + 0.3;
  sources.add(source);
  source.onended = () => sources.delete(source);
}

function busy(running) {
  $('speak').disabled = running;
  $('stop').disabled = !running;
  for (const id of ['model', 'runtime', 'lang', 'voice', 'steps', 'speed', 'text']) $(id).disabled = running;
}

$('speak').addEventListener('click', () => {
  if (!worker) return;
  audio ??= new AudioContext();
  audio.resume();
  stopPlayback();
  pieces = [];
  spoken = { audio: 0, took: 0 };
  $('save').disabled = true;
  $('pace').textContent = '';
  busy(true);
  ticket += 1;
  status('Starting…');
  barBusy();
  worker.postMessage({
    ticket,
    model: $('model').value,
    runtime: $('runtime').value,
    lang: $('lang').value,
    voice: $('voice').value,
    text: $('text').value,
    steps: Number($('steps').value),
    speed: Number($('speed').value),
    seed: 42,
  });
});

$('stop').addEventListener('click', () => {
  ticket += 1;
  worker.postMessage({ stop: true });
  stopPlayback();
  busy(false);
  bar(0);
  status('Stopped.');
});

$('save').addEventListener('click', () => {
  const gap = new Float32Array(Math.round(rate * 0.3));
  const all = pieces.flatMap((piece, ix) => (ix ? [gap, piece] : [piece]));
  const total = all.reduce((n, piece) => n + piece.length, 0);
  const view = new DataView(new ArrayBuffer(44 + total * 2));
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + total * 2, true);
  text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, total * 2, true);
  let offset = 44;
  for (const piece of all) {
    for (const sample of piece) {
      view.setInt16(offset, Math.max(-1, Math.min(1, sample)) * 32767, true);
      offset += 2;
    }
  }
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([view], { type: 'audio/wav' }));
  link.download = 'supertonic.wav';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

worker?.addEventListener('message', ({ data }) => {
  if (data.ticket !== undefined && data.ticket !== ticket) return;
  if (data.progress) {
    bar(data.progress.got / data.progress.total);
    return;
  }
  if (data.error) {
    busy(false);
    bar(0);
    status(data.error, true);
    return;
  }
  if (data.audio) {
    rate = data.rate;
    pieces.push(data.audio);
    play(data.audio);
    let energy = 0;
    let finite = true;
    for (const sample of data.audio) {
      energy += sample * sample;
      finite &&= Number.isFinite(sample);
    }
    spoken.energy = (spoken.energy || 0) + energy;
    spoken.samples = (spoken.samples || 0) + data.audio.length;
    spoken.finite = (spoken.finite ?? true) && finite;
    $('pace').dataset.rms = Math.sqrt(spoken.energy / spoken.samples).toFixed(4);
    $('pace').dataset.finite = spoken.finite;
    const all = pieces.length === 1 ? pieces[0] : Float32Array.from(pieces.flatMap((piece) => [...piece]));
    const step = Math.ceil(all.length / 16);
    const envelope = [];
    for (let at = 0; at < all.length; at += step) {
      const part = all.subarray(at, at + step);
      envelope.push(Math.sqrt(part.reduce((sum, x) => sum + x * x, 0) / part.length).toFixed(4));
    }
    $('pace').dataset.envelope = envelope.join(' ');
    spoken.audio += data.audio.length / rate;
    spoken.took += data.took;
    $('pace').textContent = `${spoken.audio.toFixed(1)} s of speech made in ${spoken.took.toFixed(2)} s · ${(spoken.audio / spoken.took).toFixed(1)}× real time`;
    $('save').disabled = false;
  }
  if (data.done) {
    busy(false);
    bar(1);
    status(data.status || (pieces.length ? 'Done.' : 'Nothing to say.'));
    return;
  }
  if (data.status) {
    status(data.status);
    if (data.busy) barBusy();
  }
});

async function buildInfo() {
  try {
    const { version, commit, dirty, date } = await (await fetch('build.json')).json();
    $('source').href = `https://github.com/sonos/tract/tree/${commit}/examples/supertonic/web`;
    $('build').textContent = `${version ? `tract ${version} · ` : ''}${commit.slice(0, 9)}${dirty ? '+dirty' : ''} · ${date}`;
  } catch {
    // The source tree has no build.json; the link stays on main.
  }
}
buildInfo();
