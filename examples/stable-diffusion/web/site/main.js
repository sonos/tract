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

// Prompts that SD 1.5 renders well, each with a seed that came out nicely.
const EXAMPLES = [
  ["Cottage", "a cozy cottage in an autumn forest, warm light in the windows, oil painting, highly detailed", 7],
  ["Mountain lake", "a beautiful mountain lake at sunrise, pine forest, morning mist, golden hour, highly detailed landscape photograph", 1],
  ["Ramen", "a bowl of ramen with a soft boiled egg and scallions, food photography, soft studio lighting", 7],
  ["Astronaut", "an astronaut riding a horse on mars, digital art, highly detailed", 7],
  ["Fox", "a red fox sitting in a snowy forest, wildlife photography, sharp focus, bokeh", 7],
  ["Sleeping cat", "a fluffy ginger cat sleeping on a windowsill, sunlight, cozy, highly detailed photograph", 1],
  ["Lighthouse", "a lighthouse on a rocky coast during a storm, dramatic waves, oil painting by ivan aivazovsky", 7],
  ["Sunflowers", "a field of sunflowers under a blue sky, impressionist painting in the style of van gogh", 7],
  ["Neon city", "a futuristic city skyline at night, neon lights, flying cars, cyberpunk, digital art, highly detailed", 1],
  ["Temple", "a japanese temple in a misty bamboo forest, cherry blossoms, ukiyo-e style", 7],
  ["Treehouse", "a treehouse village in a giant redwood forest, fantasy concept art, volumetric light", 7],
];

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
    $("source").href = `https://github.com/sonos/tract/tree/${commit}/examples/stable-diffusion/web`;
    $("build").textContent = `${commit.slice(0, 9)}${dirty ? "+dirty" : ""} · ${date}`;
  } catch {}
}

async function main() {
  buildInfo();
  examples();
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
        status(`Downloading weights: ${gb(data.loaded)} / ${gb(data.total)} GB`);
        break;
      case "loading":
        status(`Loading ${data.component.replace("_", " ")} onto the GPU…`);
        break;
      case "loaded":
        console.log(`${data.component} loaded in ${(data.ms / 1000).toFixed(1)} s`);
        break;
      case "count":
        showCount(data);
        break;
      case "tokenizer":
        recount();
        break;
      case "ready":
        recount();
        dropLabel();
        $("drop").disabled = false;
        bar(1);
        status("Ready.");
        ready();
        focusPrompt(false);
        break;
      case "progress":
        bar(data.stage === "denoising" ? data.step / data.steps : data.stage === "decoding" ? 1 : 0);
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
        const image = new ImageData(new Uint8ClampedArray(data.rgba.buffer), 512, 512);
        $("out").getContext("2d").putImageData(image, 0, 0);
        status(`Done in ${(data.ms / 1000).toFixed(1)} s.`);
        go.disabled = false;
        $("drop").disabled = false;
        focusPrompt(false);
        break;
      }
      case "error":
        status(`Error: ${data.message}`);
        console.error(data.message);
        go.disabled = dropped;
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
  for (const field of ["prompt", "negative"]) {
    $(field).addEventListener("input", () => requestCount(field));
  }
  showCount = (data) => {
    if (data.seq !== latest[data.field] || data.n < 0) return;
    const el = $(`${data.field}-count`);
    const over = data.n > data.limit;
    el.textContent = over
      ? `${data.n} / ${data.limit} tokens: the end is cut off`
      : `${data.n} / ${data.limit} tokens`;
    el.classList.toggle("over", over);
  };
  recount = () => ["prompt", "negative"].forEach(requestCount);

  $("drop").onclick = () => {
    if (!confirm("Delete the downloaded model from this browser? It will be downloaded again (about 2.1 GB) next time.")) return;
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

  go.onclick = () => {
    go.disabled = true;
    $("drop").disabled = true;
    worker.postMessage({
      type: "generate",
      prompt: $("prompt").value,
      negative: $("negative").value,
      sampler: $("sampler").value,
      steps: Number($("steps").value),
      guidance: Number($("guidance").value),
      seed: Number($("seed").value),
    });
  };
}

main();
