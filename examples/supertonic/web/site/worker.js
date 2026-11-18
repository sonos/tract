import init, { Engine, chunks } from './pkg/supertonic_web.js';
import { fileUrl, release } from './models.mjs';

const post = (message, transfer) => self.postMessage(message, transfer || []);
const files = new Map();
const engines = new Map();
let wasm;
let active = 0;

function runtimeName(runtime) {
  return runtime === 'wgpu' ? 'GPU' : 'CPU';
}

async function fetchBytes(url, expected, onProgress) {
  const cache = await caches.open('supertonic');
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('content-length')) || expected || 0;
  const reader = response.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.byteLength;
    onProgress?.(got, total);
  }
  const bytes = new Uint8Array(got);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  await cache.put(url, new Response(bytes.slice(), { headers: { 'content-type': 'application/octet-stream' } }));
  return bytes;
}

function once(map, key, make) {
  if (!map.has(key)) {
    map.set(key, make().catch((error) => {
      map.delete(key);
      throw error;
    }));
  }
  return map.get(key);
}

const GRAPHS = [
  ['duration_predictor', 'duration predictor'],
  ['text_encoder', 'text encoder'],
  ['vector_estimator', 'vector estimator'],
  ['vocoder', 'vocoder'],
];

function graphsFor(id) {
  return once(files, id, async () => {
    const model = release(id);
    const total = Object.values(model.graphs).reduce((a, b) => a + b, 0);
    let before = 0;
    const out = {};
    for (const [name, label] of GRAPHS) {
      post({ status: `Downloading the ${model.name} ${label}…` });
      const expected = model.graphs[name];
      out[name] = await fetchBytes(fileUrl(model, `onnx/${name}.onnx`), expected, (got) =>
        post({ progress: { got: before + got, total } }),
      );
      before += expected;
    }
    out.tts_json = await fetchBytes(fileUrl(model, 'onnx/tts.json'));
    out.unicode_indexer = await fetchBytes(fileUrl(model, 'onnx/unicode_indexer.json'));
    return out;
  });
}

function engineFor(runtime, id) {
  return once(engines, `${runtime}:${id}`, async () => {
    wasm ??= init();
    await wasm;
    const graphs = await graphsFor(id);
    post({ status: `Preparing ${release(id).name} on the ${runtimeName(runtime)}…`, busy: true });
    return Engine.create(
      runtime,
      graphs.duration_predictor,
      graphs.text_encoder,
      graphs.vector_estimator,
      graphs.vocoder,
      graphs.tts_json,
      graphs.unicode_indexer,
    );
  });
}

function voiceFor(id, voice) {
  return once(files, `${id}:${voice}`, () => fetchBytes(fileUrl(release(id), `voice_styles/${voice}.json`)));
}

self.onmessage = ({ data }) => {
  active += 1;
  if (data.stop) return;
  speak(data, active);
};

async function speak(data, job) {
  const reply = (message, transfer) => job === active && post({ ticket: data.ticket, ...message }, transfer);
  const where = runtimeName(data.runtime);
  try {
    const engine = await engineFor(data.runtime, data.model);
    const voice = await voiceFor(data.model, data.voice);
    await wasm;
    const parts = chunks(data.text, data.lang);
    if (!parts.length) {
      reply({ done: true, status: 'Nothing to say.' });
      return;
    }
    for (let ix = 0; ix < parts.length; ix++) {
      if (job !== active) return;
      reply({ status: `Speaking part ${ix + 1} of ${parts.length} on the ${where}…`, busy: true });
      const started = performance.now();
      const samples = await engine.speak(voice, parts[ix], data.lang, data.steps, data.speed, data.seed + ix);
      const took = (performance.now() - started) / 1000;
      reply({ audio: samples, index: ix, count: parts.length, rate: engine.sampleRate, took }, [samples.buffer]);
    }
    reply({ done: true });
  } catch (error) {
    reply({ error: error?.message || String(error) });
  }
}
