// Runs the model: weights cached in OPFS from Hugging Face, graphs from this site, on WebGPU.
import initWasm, * as sd from "./pkg/sd15_web.js";

const post = (message, transfer) => self.postMessage(message, transfer || []);

// One sync access handle per weight file, read synchronously from the wasm side.
const handles = [];
globalThis.sd15ReadInto = (file, offset, dest) =>
  handles[file].read(dest, { at: offset }) === dest.length;

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

async function start() {
  await initWasm();
  await sd.init();
  const manifest = await (await fetch("manifest.json")).json();

  const [vocab, merges] = await Promise.all(
    manifest.tokenizer.map(async (path) => (await fetch(manifest.hf + path)).text()),
  );
  sd.set_tokenizer(vocab, merges);
  post({ type: "tokenizer" });

  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle("sd15", { create: true });
  const total = manifest.components.reduce((n, c) => n + c.size, 0);
  const loaded = manifest.components.map(() => 0);
  for (const [i, c] of manifest.components.entries()) {
    const name = c.weights.replaceAll("/", "_");
    const handle = await cache(dir, name, manifest.hf + c.weights, c.size, (bytes) => {
      loaded[i] = bytes;
      post({ type: "download", loaded: loaded.reduce((a, b) => a + b, 0), total });
    });
    handles[i] = await handle.createSyncAccessHandle();
  }

  for (const [i, c] of manifest.components.entries()) {
    post({ type: "loading", component: c.name });
    const graph = new Uint8Array(await (await fetch(c.graph)).arrayBuffer());
    const t = performance.now();
    sd.load_component(c.name, graph, i);
    post({ type: "loaded", component: c.name, ms: performance.now() - t });
  }
  post({ type: "ready" });
}

self.onmessage = async ({ data }) => {
  if (data.type === "count") {
    // Before the wasm and the tokenizer are in, there is nothing to count with yet: the page
    // asks again once the model is ready.
    try {
      post({ type: "count", field: data.field, seq: data.seq, n: sd.count_tokens(data.text), limit: sd.prompt_token_limit() });
    } catch {}
    return;
  }
  try {
    if (data.type === "start") {
      await start();
    } else if (data.type === "drop") {
      for (const h of handles) h.close();
      handles.length = 0;
      const root = await navigator.storage.getDirectory();
      await root.removeEntry("sd15", { recursive: true });
      post({ type: "dropped" });
    } else if (data.type === "generate") {
      const t = performance.now();
      const rgba = await sd.generate(data.prompt, data.negative, data.steps, data.seed, data.guidance, data.sampler, (stage, step, steps) =>
        post({ type: "progress", stage, step, steps }),
      );
      post({ type: "image", rgba, ms: performance.now() - t }, [rgba.buffer]);
    }
  } catch (e) {
    const busy = e && e.name === "NoModificationAllowedError";
    post({
      type: "error",
      message: busy ? "The model is already open in another tab of this site: close it and reload." : String(e && e.stack ? e.stack : e),
    });
  }
};
