// Runs the model: stage tars from this site cached in OPFS, read one stage at a time, on WebGPU.
import initWasm, * as bonsai from "./pkg/bonsai_web.js";

const post = (message, transfer) => self.postMessage(message, transfer || []);

// One sync access handle per cached file, read synchronously from the wasm side.
const handles = {};
const reading = { file: null, sizes: {}, last: 0 };
globalThis.bonsaiRead = (file, offset, dest) => {
  const n = handles[file].read(dest, { at: offset });
  const now = performance.now();
  if (now - reading.last > 60) {
    reading.last = now;
    post({ type: "bytes", file, offset: offset + n, size: reading.sizes[file], sample: dest[n >> 1] ?? 0 });
  }
  return n;
};

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

// The origin private file system, where the weights are cached. Private windows and blocked
// site data refuse it.
async function storage() {
  try {
    return await navigator.storage.getDirectory();
  } catch (e) {
    throw new StorageUnavailable(
      "This browser does not let the page store the model on this device, which happens " +
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
  await bonsai.init();
  const manifest = await (await fetch("manifest.json")).json();

  const tokenizer = new Uint8Array(await (await fetch(manifest.tokenizer)).arrayBuffer());
  bonsai.set_tokenizer(tokenizer);
  post({ type: "tokenizer" });

  const root = await storage();
  const total = manifest.files.reduce((n, f) => n + f.size, 0);
  const { quota, usage } = await navigator.storage.estimate();
  if (quota !== undefined && quota - usage < total) {
    const cached = await root.getDirectoryHandle("bonsai").catch(() => null);
    if (!cached) {
      throw new StorageUnavailable(
        `The model needs ${(total / 1e9).toFixed(1)} GB of browser storage and this browser allows ` +
          `${((quota - usage) / 1e9).toFixed(1)} GB here.`,
      );
    }
  }
  const dir = await root.getDirectoryHandle("bonsai", { create: true });
  for (const f of manifest.files) reading.sizes[f.name] = f.size;
  const wanted = new Set(manifest.files.map((f) => f.name));
  for await (const name of dir.keys()) {
    if (!wanted.has(name)) await dir.removeEntry(name);
  }
  const loaded = manifest.files.map(() => 0);
  for (const [i, f] of manifest.files.entries()) {
    const handle = await cache(dir, f.name, f.url, f.size, (bytes) => {
      loaded[i] = bytes;
      post({ type: "download", loaded: loaded.reduce((a, b) => a + b, 0), total });
    });
    handles[f.name] = await handle.createSyncAccessHandle();
  }
  post({ type: "ready", sizes: manifest.sizes });
}

self.onmessage = async ({ data }) => {
  if (data.type === "count") {
    // Before the wasm and the tokenizer are in, there is nothing to count with yet: the page
    // asks again once the model is ready.
    try {
      post({ type: "count", field: data.field, seq: data.seq, n: bonsai.count_tokens(data.text), limit: bonsai.prompt_token_limit() });
    } catch {}
    return;
  }
  try {
    if (data.type === "start") {
      await start();
    } else if (data.type === "drop") {
      for (const [name, handle] of Object.entries(handles)) {
        handle.close();
        delete handles[name];
      }
      const root = await navigator.storage.getDirectory();
      await root.removeEntry("bonsai", { recursive: true });
      post({ type: "dropped" });
    } else if (data.type === "generate") {
      const t = performance.now();
      const rgba = await bonsai.generate(data.prompt, data.steps, data.seed, data.width, data.height, (stage, step, steps) =>
        post({ type: "progress", stage, step, steps }),
        (rgba, step, width) => post({ type: "preview", rgba, step, width }, [rgba.buffer]),
      );
      post({ type: "image", rgba, ms: performance.now() - t }, [rgba.buffer]);
    }
  } catch (e) {
    const busy = e && e.name === "NoModificationAllowedError";
    const plain = e instanceof StorageUnavailable;
    post({
      type: "error",
      message: busy
        ? "The model is already open in another tab of this site: close it and reload."
        : plain
          ? e.message
          : describe(e),
    });
  }
};
