import init, { Engine } from './pkg/ppocr_web.js';
import { alphabetFromYaml, recognize } from './ocr.mjs';
import { dictUrl, tier, weightUrl } from './models.mjs';

const post = (message) => self.postMessage(message);
const weights = new Map();
const engines = new Map();
let wasm;
let active = 0;
let gate = Promise.resolve();

function acquire() {
  let release;
  const next = new Promise((resolve) => {
    release = resolve;
  });
  const turn = gate.then(() => release);
  gate = next;
  return turn;
}

function runtimeName(runtime) {
  return runtime === 'wgpu' ? 'GPU' : 'CPU';
}

async function fetchBytes(url, expected, onProgress) {
  const cache = await caches.open('ppocr-v6');
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('content-length')) || expected || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onProgress(got, total);
  }
  const bytes = new Uint8Array(got);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  await cache.put(url, new Response(bytes.slice(), { headers: { 'content-type': 'application/octet-stream' } }));
  return bytes;
}

function weightsFor(id) {
  if (!weights.has(id)) {
    const pending = download(id).catch((error) => {
      weights.delete(id);
      throw error;
    });
    weights.set(id, pending);
  }
  return weights.get(id);
}

async function download(id) {
  const model = tier(id);
  const report = (label) => (got, total) => post({ model: id, progress: { label, got, total } });
  post({ model: id, status: `Downloading ${model.name} detection…` });
  const det = await fetchBytes(weightUrl(model.det), model.det.bytes, report(`${model.name} detection`));
  post({ model: id, status: `Downloading ${model.name} recognition…` });
  const rec = await fetchBytes(weightUrl(model.rec), model.rec.bytes, report(`${model.name} recognition`));
  post({ model: id, status: `Downloading the ${model.name} dictionary…` });
  const yml = new TextDecoder().decode(await fetchBytes(dictUrl(model), 0, report(`${model.name} dictionary`)));
  return { model, alphabet: alphabetFromYaml(yml), det, rec };
}

function sessionsFor(engine) {
  class Tensor {
    constructor(_type, data, dims) {
      this.data = data;
      this.dims = dims;
    }
  }
  const feed = (feeds) => Object.values(feeds)[0];
  return {
    Tensor,
    det: {
      inputNames: ['x'],
      outputNames: ['map'],
      async run(feeds) {
        const tensor = feed(feeds);
        const height = tensor.dims[2];
        const width = tensor.dims[3];
        const map = await engine.detect(tensor.data, height, width);
        return { map: { data: map.data, dims: [1, 1, map.height, map.width] } };
      },
    },
    rec: {
      inputNames: ['x'],
      outputNames: ['seq'],
      async run(feeds) {
        const tensor = feed(feeds);
        const seq = await engine.recognize(tensor.data, tensor.dims[3]);
        return { seq: { data: seq.data, dims: [1, seq.time, seq.classes] } };
      },
    },
  };
}

function engineFor(runtime, id) {
  const key = `${runtime}:${id}`;
  if (!engines.has(key)) {
    const pending = (async () => {
      wasm ??= init();
      await wasm;
      const loaded = await weightsFor(id);
      post({ model: id, status: `Preparing ${loaded.model.name} on the ${runtimeName(runtime)}…` });
      const engine = await Engine.create(runtime, loaded.det, loaded.rec);
      return { ...loaded, sessions: sessionsFor(engine) };
    })().catch((error) => {
      engines.delete(key);
      throw error;
    });
    engines.set(key, pending);
  }
  return engines.get(key);
}

self.onmessage = ({ data }) => {
  if (data.stop) {
    active += 1;
    post({ ticket: data.ticket, stopped: true });
    return;
  }
  const job = ++active;
  post({ ticket: data.ticket, job, accepted: true });
  const pending = engineFor(data.runtime || 'cpu', data.tier);
  task(data, job, pending);
};

async function task(data, job, pending) {
  const where = runtimeName(data.runtime || 'cpu');
  const reply = (message) => post({ ticket: data.ticket, job, ...message });
  try {
    const prepared = await pending;
    if (job !== active) return;
    const release = await acquire();
    try {
      if (job !== active) return;
      reply({ status: `Finding text on the ${where}…` });
      const lines = await recognize(
        { rgb: data.rgb, width: data.width, height: data.height },
        prepared.sessions,
        prepared.alphabet,
        prepared.model,
        {
          cancelled: () => job !== active,
          onLine: (done, total) => {
            if (job === active) {
              reply({ status: `Reading line ${done} of ${total} on the ${where}…`, read: { done, total } });
            }
          },
        },
      );
      if (!lines || job !== active) return;
      const status = lines.length ? `Read ${lines.length} lines on the ${where}.` : `No text found on the ${where}.`;
      reply({ status, lines });
    } finally {
      release();
    }
  } catch (error) {
    if (job === active) reply({ error: error?.message || String(error) });
  }
}
