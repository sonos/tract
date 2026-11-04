import { Stage } from "./stage.js";

const $ = (id) => document.getElementById(id);
const status = (text) => ($("status").textContent = text);
const bar = (fraction) => ($("bar").style.width = `${Math.round(fraction * 100)}%`);
const go = $("go");
const gb = (bytes) => (bytes / 1e9).toFixed(2);

async function check() {
  if (!navigator.gpu) return "This browser has no WebGPU. Try a recent Chrome or Edge.";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return "No WebGPU adapter is available.";
  if (!adapter.features.has("shader-f16")) return "This GPU or browser lacks WebGPU's shader-f16 feature, which the model needs.";
  if (!navigator.storage || !navigator.storage.getDirectory) return "This browser has no origin private file system to cache the weights in.";
  return null;
}

let showCount = () => {};
let request = null;
let stage = null;

// Back to the prompt, unless the user is busy in another field.
function focusPrompt(force) {
  const active = document.activeElement;
  const elsewhere = active && active !== document.body && active !== $("go") && active !== $("prompt") && active.matches("input, textarea, select");
  if (force || !elsewhere) {
    const p = $("prompt");
    p.focus();
    p.setSelectionRange(p.value.length, p.value.length);
  }
}

const SHORTCUT = navigator.platform.startsWith("Mac") ? "⇧↵" : "Shift+Enter";
const ready = () => {
  go.disabled = false;
  go.innerHTML = `Generate<span class="key">${SHORTCUT}</span>`;
};
let dropped = false;

// The button says how much this site keeps, when the browser tells.
async function dropLabel() {
  try {
    const { usage } = await navigator.storage.estimate();
    if (usage) $("drop").textContent = `Delete the downloaded model (${gb(usage)} GB)`;
  } catch {}
}
let recount = () => {};

// Prompts to start from, each with a seed.
const EXAMPLES = [
  ["Bonsai", "a photo of a bonsai tree on a wooden table, soft window light", 42],
  ["Noodle market", "a cyberpunk noodle market in a dense alley, neon kanji signs, steam, reflective wet pavement", 1],
  ["Fox", "a red fox sitting in a snowy forest, wildlife photography, sharp focus", 7],
  ["Lighthouse", "a lighthouse on a rocky coast during a storm, dramatic waves, oil painting", 7],
  ["Ramen", "a bowl of ramen with a soft boiled egg and scallions, food photography, soft studio lighting", 7],
  ["Astronaut", "an astronaut riding a horse on mars, digital art", 7],
];
const STYLES = ["photorealistic", "cinematic", "anime", "oil painting", "watercolor", "pixel art", "golden hour", "vibrant", "muted"];

function styles() {
  for (const style of STYLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = style;
    b.onclick = () => {
      const p = $("prompt");
      p.value = `${p.value.trimEnd().replace(/,$/, "")}, ${style}`;
      recount();
      focusPrompt(true);
    };
    $("styles").append(b);
  }
}

// Each phase of a run, in order, as the worker names them.
const PHASES = [
  ["loading text encoder", "Load text encoder", true],
  ["encoding", "Encode prompt", false],
  ["loading transformer", "Load transformer", true],
  ["denoising", "Denoise", false],
  ["loading image decoder", "Load image decoder", true],
  ["decoding", "Decode image", false],
];
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const TIMINGS = "bonsai-timings";

// Past durations, for an ETA before the run has measured anything itself.
function pastTimings() {
  try {
    return JSON.parse(localStorage.getItem(TIMINGS)) || {};
  } catch {
    return {};
  }
}

function keepTimings(timings) {
  try {
    localStorage.setItem(TIMINGS, JSON.stringify(timings));
  } catch {}
}

let run = null;
const loaded = new Set();

function startRun(steps, width, height) {
  const past = pastTimings();
  stage.begin(width, height, $("prompt").value, past.encoding);
  run = { start: performance.now(), steps, tokens: (width * height) / 256, at: null, phase: null, spent: {}, stepStart: null, stepMs: [], past: pastTimings() };
  $("run").hidden = false;
  renderRun();
}

function enterPhase(phase, step) {
  const now = performance.now();
  if (run.phase !== phase) {
    if (run.phase) run.spent[run.phase] = now - run.at;
    run.phase = phase;
    run.at = now;
    stage.enter(phase);
  }
  if (phase === "denoising") {
    if (run.stepStart !== null && step > run.stepMs.length) run.stepMs.push(now - run.stepStart);
    run.stepStart = now;
    run.step = step;
  }
}

function perStep() {
  if (run.stepMs.length) return run.stepMs.reduce((a, b) => a + b) / run.stepMs.length;
  return run.past[`step:${run.tokens}`];
}

// What a phase should still take, or undefined with nothing to go on.
function remaining(stage, isLoad, now) {
  if (stage in run.spent) return 0;
  const running = run.phase === stage ? now - run.at : 0;
  if (isLoad && loaded.has(`${stage}:${run.tokens}`)) return 0;
  if (stage === "denoising") {
    const each = perStep();
    if (each === undefined) return undefined;
    const started = run.phase === stage ? now - run.stepStart : 0;
    return Math.max(0, (run.steps - run.stepMs.length) * each - started);
  }
  const before = run.past[`${stage}:${run.tokens}`] ?? run.past[stage];
  return before === undefined ? undefined : Math.max(0, before - running);
}

function renderRun() {
  if (!run) return;
  const now = performance.now();
  $("elapsed").textContent = `Elapsed ${secs((run.end ?? now) - run.start)}`;
  let left = 0;
  let known = true;
  const rows = PHASES.map(([stage, label, isLoad]) => {
    const rest = remaining(stage, isLoad, now);
    if (rest === undefined) known = false;
    else left += rest;
    const active = !run.end && run.phase === stage;
    let time;
    if (stage in run.spent) {
      time = isLoad && run.spent[stage] < 100 ? "ready" : secs(run.spent[stage]);
    } else if (active) {
      time = `${secs(now - run.at)}…`;
    } else {
      time = rest === undefined ? "–" : rest === 0 ? "ready" : `~${secs(rest)}`;
    }
    if (stage === "denoising" && active) time = `step ${run.step + 1} / ${run.steps}, ${time}`;
    return `<tr class="${active ? "active" : ""}"><td>${label}</td><td>${time}</td></tr>`;
  });
  $("phases").innerHTML = rows.join("");
  $("eta").textContent = run.end ? "" : known ? `About ${secs(left)} left` : "Measuring…";
  const each = perStep();
  $("rate").textContent = each === undefined ? "" : `${secs(each)} per step · ${Math.round((run.tokens * 1000) / each)} image tokens/s`;
}

function finishRun() {
  const now = performance.now();
  if (run.phase) run.spent[run.phase] = now - run.at;
  run.end = now;
  if (run.stepStart !== null && run.stepMs.length < run.steps) run.stepMs.push(run.spent.denoising - run.stepMs.reduce((a, b) => a + b, 0));
  const past = pastTimings();
  for (const [stage, , isLoad] of PHASES) {
    if (stage === "denoising" || !(stage in run.spent)) continue;
    if (!isLoad || run.spent[stage] >= 100) {
      past[`${stage}:${run.tokens}`] = run.spent[stage];
      past[stage] = run.spent[stage];
      if (isLoad) loaded.add(`${stage}:${run.tokens}`);
    }
  }
  past[`step:${run.tokens}`] = perStep();
  keepTimings(past);
  renderRun();
}

setInterval(() => run && !run.end && renderRun(), 200);

// Generated images, newest first, each with what made it. Kept in IndexedDB; a browser that
// refuses it still has them until the page closes.
const gallery = [];
let shown = null;

function openGallery() {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("bonsai", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("images", { keyPath: "id" });
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
  });
}

async function withImages(mode, act) {
  const db = await openGallery();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("images", mode);
    const request = act(tx.objectStore("images"));
    tx.oncomplete = () => {
      db.close();
      resolve(request.result);
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(tx.error);
    };
  });
}

const stored = ({ url, ...item }) => item;
const quietly = (work) => work().catch((e) => console.warn("gallery storage", e));

async function show(item) {
  if (stage.active) return;
  shown = item;
  const c = $("canvas");
  if (item) {
    const bitmap = await createImageBitmap(item.blob);
    if (shown !== item) return;
    c.width = item.width;
    c.height = item.height;
    c.getContext("2d").drawImage(bitmap, 0, 0);
  } else {
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
  }
  $("save").disabled = !item;
  $("remix").disabled = !item || go.disabled;
  for (const t of $("gallery").children) t.classList.toggle("on", item !== null && t.dataset.id === String(item.id));
}

function forget(item) {
  const at = gallery.indexOf(item);
  gallery.splice(at, 1);
  URL.revokeObjectURL(item.url);
  quietly(() => withImages("readwrite", (images) => images.delete(item.id)));
  renderGallery();
  if (shown === item) show(gallery[Math.min(at, gallery.length - 1)] ?? null);
}

const TRASH = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>`;

function renderGallery() {
  $("gallery").replaceChildren(
    ...gallery.map((item) => {
      const thumb = document.createElement("div");
      thumb.className = "thumb";
      thumb.dataset.id = item.id;
      thumb.classList.toggle("on", item === shown);
      const open = document.createElement("button");
      open.type = "button";
      open.className = "open";
      open.title = `${item.prompt} (seed ${item.seed})`;
      const img = document.createElement("img");
      img.alt = item.prompt;
      img.src = item.url;
      open.append(img);
      open.onclick = () => show(item);
      const del = document.createElement("button");
      del.type = "button";
      del.className = "del";
      del.title = "Delete this image";
      del.setAttribute("aria-label", "Delete this image");
      del.innerHTML = TRASH;
      del.onclick = () => forget(item);
      thumb.append(open, del);
      return thumb;
    }),
  );
}

async function remember(image, width, height, made) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  c.getContext("2d").putImageData(image, 0, 0);
  const blob = await new Promise((resolve) => c.toBlob(resolve, "image/png"));
  const item = { ...made, id: Date.now(), width, height, blob, url: URL.createObjectURL(blob) };
  gallery.unshift(item);
  quietly(() => withImages("readwrite", (images) => images.put(stored(item))));
  renderGallery();
  await show(item);
}

async function restoreGallery() {
  await quietly(async () => {
    const items = await withImages("readonly", (images) => images.getAll());
    for (const item of items.sort((a, b) => b.id - a.id)) {
      gallery.push({ ...item, url: URL.createObjectURL(item.blob) });
    }
  });
  if (gallery.length) {
    renderGallery();
    await show(gallery[0]);
  }
}

// The sizes the site carries, smallest first; heights and widths as `HxW`.
function sizes(list) {
  const shape = (s) => s.split("x").map(Number);
  const area = (s) => shape(s)[0] * shape(s)[1];
  const saved = (() => {
    try {
      return localStorage.getItem("bonsai-size");
    } catch {
      return null;
    }
  })();
  $("size").replaceChildren(
    ...[...list].sort((a, b) => area(a) - area(b)).map((s) => {
      const [h, w] = shape(s);
      const o = new Option(`${w} × ${h}${h === w ? "" : w > h ? " landscape" : " portrait"}`, s);
      o.selected = s === saved || (!list.includes(saved) && s === "256x256");
      return o;
    }),
  );
  $("size").onchange = () => {
    try {
      localStorage.setItem("bonsai-size", $("size").value);
    } catch {}
  };
}

function saveImage() {
  if (!shown) return;
  const slug = shown.prompt.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "bonsai";
  const a = document.createElement("a");
  a.href = shown.url;
  a.download = `${slug}-${shown.seed}.png`;
  a.click();
}

function examples() {
  for (const [name, prompt, seed] of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name;
    b.onclick = () => {
      $("prompt").value = prompt;
      $("seed").value = seed;
      recount();
      focusPrompt(true);
    };
    $("examples").append(b);
  }
}

// Which commit and day this page was built from, when the build says so.
async function buildInfo() {
  try {
    const { commit, dirty, date } = await (await fetch("build.json")).json();
    $("source").href = `https://github.com/sonos/tract/tree/${commit}/examples/bonsai-image/web`;
    $("build").textContent = `${commit.slice(0, 9)}${dirty ? "+dirty" : ""} · ${date}`;
  } catch {}
}

async function main() {
  stage = new Stage($("canvas"));
  buildInfo();
  examples();
  styles();
  restoreGallery();
  const problem = await check();
  if (problem) {
    status(problem);
    go.textContent = "Unsupported";
    return;
  }
  if (navigator.storage.persist) await navigator.storage.persist();

  const worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    switch (data.type) {
      case "download":
        bar(data.loaded / data.total);
        status(`Downloading the model: ${gb(data.loaded)} / ${gb(data.total)} GB`);
        break;
      case "count":
        showCount(data);
        break;
      case "tokenizer":
        recount();
        break;
      case "ready":
        sizes(data.sizes);
        recount();
        dropLabel();
        $("drop").disabled = false;
        bar(1);
        status("Ready.");
        ready();
        $("remix").disabled = !shown;
        focusPrompt(false);
        break;
      case "bytes":
        stage.bytes(data.offset / data.size, data.sample);
        break;
      case "preview":
        stage.preview(data.rgba, data.width);
        break;
      case "progress":
        enterPhase(data.stage, data.step);
        bar(data.stage === "denoising" ? data.step / data.steps : 0);
        status(data.stage === "denoising" ? `Denoising: step ${data.step + 1} / ${data.steps}` : `${data.stage[0].toUpperCase()}${data.stage.slice(1)}…`);
        break;
      case "dropped":
        dropped = true;
        go.disabled = true;
        go.textContent = "Model deleted";
        $("drop").disabled = true;
        $("drop").textContent = "Model deleted";
        status("The model was removed from this browser's storage. Reload the page to download it again.");
        break;
      case "image": {
        const { width, height } = request;
        const image = new ImageData(new Uint8ClampedArray(data.rgba.buffer), width, height);
        finishRun();
        bar(1);
        status(`Done in ${(data.ms / 1000).toFixed(1)} s.`);
        go.disabled = false;
        stage.finish(image).then(() => remember(image, width, height, request));
        $("drop").disabled = false;
        focusPrompt(false);
        break;
      }
      case "error":
        stage.stop();
        status(`Error: ${data.message}`);
        console.error(data.message);
        go.disabled = dropped;
        $("remix").disabled = dropped || !shown;
        $("drop").disabled = dropped;
        focusPrompt(false);
        break;
    }
  };
  worker.postMessage({ type: "start" });

  // Token counts come from the worker, which holds the tokenizer; the latest request wins.
  let countSeq = 0;
  const latest = {};
  const requestCount = (field) => {
    latest[field] = ++countSeq;
    worker.postMessage({ type: "count", field, seq: countSeq, text: $(field).value });
  };
  $("prompt").addEventListener("input", () => requestCount("prompt"));
  showCount = (data) => {
    if (data.seq !== latest[data.field] || data.n < 0) return;
    const el = $(`${data.field}-count`);
    const over = data.n > data.limit;
    el.textContent = over
      ? `${data.n} / ${data.limit} tokens: the end is cut off`
      : `${data.n} / ${data.limit} tokens`;
    el.classList.toggle("over", over);
  };
  recount = () => requestCount("prompt");

  $("drop").onclick = () => {
    if (!confirm("Delete the downloaded model from this browser? It will be downloaded again (about 4.7 GB) next time.")) return;
    $("drop").disabled = true;
    go.disabled = true;
    worker.postMessage({ type: "drop" });
  };

  // Shift+Enter, or Ctrl/Cmd+Enter, generates from anywhere on the page.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || !(e.shiftKey || e.ctrlKey || e.metaKey) || e.isComposing) return;
    e.preventDefault();
    if (!go.disabled) go.click();
  });

  $("save").onclick = saveImage;
  $("remix").onclick = () => {
    if (!shown) return;
    $("prompt").value = shown.prompt;
    $("steps").value = shown.steps;
    $("size").value = `${shown.height}x${shown.width}`;
    $("seed").value = Math.floor(Math.random() * 1e9);
    recount();
    go.click();
  };

  go.onclick = () => {
    go.disabled = true;
    $("drop").disabled = true;
    $("remix").disabled = true;
    const [height, width] = $("size").value.split("x").map(Number);
    request = { prompt: $("prompt").value, steps: Number($("steps").value), seed: Number($("seed").value), width, height };
    startRun(request.steps, width, height);
    worker.postMessage({ type: "generate", ...request });
  };
}

main();
