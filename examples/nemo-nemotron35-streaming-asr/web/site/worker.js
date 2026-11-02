// Runs the model: weights cached in OPFS from Hugging Face, graphs from this site, the encoder on
// WebGPU. Audio comes from the page's audio worklet, or from a file, and goes through one stream at
// a time; the transcript so far goes back after each step.
import initWasm, * as asr from "./pkg/nemotron_web.js";

const post = (message) => self.postMessage(message);

let weights = null;
globalThis.nemotronReadInto = (offset, dest) => weights.read(dest, { at: offset }) === dest.length;

// The file at `url` in OPFS under `name`, downloaded (or resumed) unless already complete.
async function cache(dir, name, url, size, onBytes) {
  const handle = await dir.getFileHandle(name, { create: true });
  if ((await handle.getFile()).size === size) {
    onBytes(size);
    return handle;
  }
  const access = await handle.createSyncAccessHandle();
  try {
    let position = access.getSize();
    if (position > size) {
      access.truncate(0);
      position = 0;
    }
    const response = await fetch(url, position ? { headers: { Range: `bytes=${position}-` } } : {});
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    if (position && response.status !== 206) {
      access.truncate(0);
      position = 0;
    }
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      access.write(value, { at: position });
      position += value.length;
      onBytes(position);
    }
    access.flush();
    if (position !== size) throw new Error(`${name}: got ${position} bytes, expected ${size}`);
  } finally {
    access.close();
  }
  return handle;
}

class StorageUnavailable extends Error {}

async function storage() {
  try {
    return await navigator.storage.getDirectory();
  } catch (e) {
    throw new StorageUnavailable(
      "This browser does not let the page store the model (2.6 GB) on this device, which happens " +
        "in private windows and when site data is blocked. Try a regular window.",
      { cause: e },
    );
  }
}

// What went wrong, whatever threw it: Firefox leaves the name and message out of a stack.
function describe(e) {
  if (!(e instanceof Error) && !(e instanceof DOMException)) return String(e);
  const head = `${e.name}: ${e.message}`;
  return e.stack && !e.stack.startsWith(head) ? `${head}\n${e.stack}` : e.stack || head;
}

async function start() {
  await initWasm();
  await asr.init();
  const manifest = await (await fetch("manifest.json")).json();
  const tokenizer = await (await fetch(manifest.hf + manifest.tokenizer)).text();

  const root = await storage();
  const dir = await root.getDirectoryHandle("nemotron35", { create: true });
  const handle = await cache(dir, manifest.weights, manifest.hf + manifest.weights, manifest.size, (loaded) =>
    post({ type: "download", loaded, total: manifest.size }),
  );
  weights = await handle.createSyncAccessHandle();

  post({ type: "loading" });
  const graphs = await Promise.all(
    manifest.graphs.map(async (name) => {
      const response = await fetch(`models/${name}.nnef.tar`);
      if (!response.ok) throw new Error(`${name} graph: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    }),
  );
  const t = performance.now();
  asr.load(...graphs, tokenizer);
  post({ type: "ready", ms: performance.now() - t });
}

// The stream being fed: blocks queue up while a step runs, and the next step takes them all.
let stream = null;

function open(lang, rate) {
  asr.start(lang, rate);
  stream = { queue: [], busy: false, last: false, samples: 0, compute: 0, rate };
  return stream;
}

async function drain(s) {
  if (s.busy) return;
  s.busy = true;
  try {
    while (s === stream && (s.queue.length || s.last)) {
      const blocks = s.queue.splice(0);
      const samples = new Float32Array(blocks.reduce((n, b) => n + b.length, 0));
      let at = 0;
      for (const b of blocks) {
        samples.set(b, at);
        at += b.length;
      }
      const last = s.last;
      s.last = false;
      const t = performance.now();
      const text = await asr.push(samples, last);
      s.compute += performance.now() - t;
      s.samples += samples.length;
      if (s !== stream) break;
      const audio = (1000 * s.samples) / s.rate;
      post({ type: "text", text, audio, compute: s.compute, backlog: s.queue.reduce((n, b) => n + b.length, 0) / s.rate, done: last });
    }
  } catch (e) {
    if (s === stream) post({ type: "error", message: describe(e) });
    stream = null;
  } finally {
    s.busy = false;
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "start") {
      await start();
    } else if (data.type === "listen") {
      const s = open(data.lang, data.rate);
      data.port.onmessage = ({ data: block }) => {
        if (s !== stream) return;
        s.queue.push(block);
        drain(s);
      };
    } else if (data.type === "file") {
      const s = open(data.lang, data.rate);
      // 320 ms at a time, as if it came from the microphone, but as fast as it goes.
      const step = Math.round(data.rate * 0.32);
      for (let at = 0; at < data.samples.length; at += step) s.queue.push(data.samples.subarray(at, at + step));
      s.last = true;
      drain(s);
    } else if (data.type === "stop") {
      if (stream) {
        stream.last = !data.drop;
        if (data.drop) stream = null;
        else drain(stream);
      }
    } else if (data.type === "forget") {
      stream = null;
      weights?.close();
      weights = null;
      const root = await navigator.storage.getDirectory();
      await root.removeEntry("nemotron35", { recursive: true });
      post({ type: "forgotten" });
    }
  } catch (e) {
    const busy = e && e.name === "NoModificationAllowedError";
    post({
      type: "error",
      message: busy
        ? "The model is already open in another tab of this site: close it and reload."
        : e instanceof StorageUnavailable
          ? e.message
          : describe(e),
    });
  }
};
