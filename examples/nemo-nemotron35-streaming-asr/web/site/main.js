// The microphone goes to an audio worklet, whose blocks go straight to the model's worker; the page
// shows the transcript the worker sends back after each step.
const $ = (id) => document.getElementById(id);
const go = $("go");
const RATE = 48000;

const worker = new Worker("worker.js", { type: "module" });
const state = { ready: false, listening: false, busy: false, context: null, stream: null };
const status = (text, warn = false) => {
  $("status").textContent = text;
  $("status").className = warn ? "warn" : "";
};
const gb = (bytes) => (bytes / 1e9).toFixed(2);

function controls() {
  go.disabled = !state.ready || (state.busy && !state.listening);
  go.textContent = state.listening ? "Stop" : "Start listening";
  $("file-button").disabled = !state.ready || state.busy;
  $("lang").disabled = state.busy;
}

// The transcript, with the language tags the model emits (`<en-US>`) as labels.
function show(text) {
  const out = $("transcript");
  out.replaceChildren();
  for (const part of text.trimStart().split(/(<[a-z]{2,3}-[A-Z]{2}>)/)) {
    if (/^<.*>$/.test(part)) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = part.slice(1, -1);
      out.append(tag);
    } else {
      out.append(part);
    }
  }
}

function release() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.context?.close();
  Object.assign(state, { listening: false, context: null, stream: null });
}

async function listen() {
  state.busy = true;
  controls();
  go.textContent = "Starting…";
  try {
    const raw = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: raw }).catch((error) => {
      if (error.name === "NotAllowedError") throw error;
      return navigator.mediaDevices.getUserMedia({ audio: true });
    });
    state.context = new AudioContext({ sampleRate: RATE });
    await state.context.audioWorklet.addModule("capture.js");
    const capture = new AudioWorkletNode(state.context, "capture", { numberOfOutputs: 0 });
    state.context.createMediaStreamSource(state.stream).connect(capture);
    const channel = new MessageChannel();
    capture.port.postMessage({ port: channel.port1 }, [channel.port1]);
    worker.postMessage({ type: "listen", lang: Number($("lang").value), rate: RATE, port: channel.port2 }, [channel.port2]);
    state.listening = true;
    show("");
    status("Listening.");
  } catch (error) {
    release();
    state.busy = false;
    const reasons = {
      NotAllowedError: "Microphone access was refused.",
      NotFoundError: "No microphone found. If there is one, the system may be keeping it from this browser: on macOS, see System Settings › Privacy & Security › Microphone.",
      NotSupportedError: "This browser cannot capture the microphone at 48 kHz.",
    };
    status(reasons[error.name] || `Error: ${error.message || error}`, true);
  }
  controls();
}

function stopListening() {
  release();
  worker.postMessage({ type: "stop" });
  status("Finishing…");
  controls();
}

async function transcribeFile(file) {
  state.busy = true;
  controls();
  status(`Decoding ${file.name}…`);
  try {
    const context = new OfflineAudioContext(1, 1, RATE);
    const audio = await context.decodeAudioData(await file.arrayBuffer());
    const samples = new Float32Array(audio.length);
    for (let c = 0; c < audio.numberOfChannels; c++) {
      const channel = audio.getChannelData(c);
      for (let i = 0; i < samples.length; i++) samples[i] += channel[i] / audio.numberOfChannels;
    }
    show("");
    status(`Transcribing ${file.name} (${audio.duration.toFixed(1)} s)…`);
    worker.postMessage({ type: "file", lang: Number($("lang").value), rate: RATE, samples }, [samples.buffer]);
  } catch (error) {
    state.busy = false;
    status(`Could not decode ${file.name}: ${error.message || error}`, true);
    controls();
  }
}

worker.onmessage = ({ data }) => {
  if (data.type === "download") {
    $("progress").hidden = false;
    $("progress").max = data.total;
    $("progress").value = data.loaded;
    status(data.loaded < data.total ? `Downloading the weights from Hugging Face: ${gb(data.loaded)} / ${gb(data.total)} GB` : "Weights cached.");
  } else if (data.type === "loading") {
    $("progress").hidden = true;
    status("Preparing the model for streaming and loading it on the GPU…");
  } else if (data.type === "ready") {
    state.ready = true;
    $("forget").hidden = false;
    status(`Ready (prepared in ${(data.ms / 1000).toFixed(1)} s). Nothing is recorded or sent anywhere.`);
    controls();
  } else if (data.type === "text") {
    show(data.text);
    const speed = data.compute / data.audio;
    $("stats").textContent =
      `${(data.audio / 1000).toFixed(1)} s of audio, computed in ${(data.compute / 1000).toFixed(1)} s: ` +
      `${(100 * speed).toFixed(0)} % of real time` + (data.backlog > 0.5 ? `, ${data.backlog.toFixed(1)} s behind` : "");
    if (data.done) {
      state.busy = false;
      status("Done.");
      controls();
    }
  } else if (data.type === "forgotten") {
    state.ready = false;
    $("forget").hidden = true;
    status("The cached model is deleted. Reload the page to download it again.");
    controls();
  } else if (data.type === "error") {
    release();
    state.busy = false;
    status(`Error: ${data.message}`, true);
    controls();
  }
};

go.onclick = () => (state.listening ? stopListening() : listen());
$("file-button").onclick = () => $("file").click();
$("file").onchange = () => {
  const file = $("file").files[0];
  $("file").value = "";
  if (file) transcribeFile(file);
};
$("forget").onclick = () => {
  release();
  state.busy = true;
  controls();
  worker.postMessage({ type: "forget" });
};
window.addEventListener("pagehide", () => release());

async function languages() {
  const manifest = await (await fetch("manifest.json")).json();
  const names = new Intl.DisplayNames([navigator.language, "en"], { type: "language" });
  const name = (code) => {
    try {
      return names.of(code);
    } catch {
      return code;
    }
  };
  const options = Object.entries(manifest.languages)
    .map(([code, id]) => [`${name(code)} (${code})`, id])
    .sort(([a], [b]) => a.localeCompare(b));
  const select = $("lang");
  select.append(new Option("Detect automatically", manifest.auto));
  for (const [label, id] of options) select.append(new Option(label, id));
}

async function buildInfo() {
  try {
    const { commit, dirty, date } = await (await fetch("build.json")).json();
    $("source").href = `https://github.com/sonos/tract/tree/${commit}/examples/nemo-nemotron35-streaming-asr/web`;
    $("build").textContent = `${commit.slice(0, 9)}${dirty ? "+dirty" : ""} · ${date}`;
  } catch {}
}

function check() {
  if (!navigator.gpu) return "This browser gives no WebGPU, which the model needs. Recent Chrome, Edge, Firefox and Safari support it on most GPUs.";
  if (!window.isSecureContext) return "The microphone needs a secure context: HTTPS, or localhost.";
  if (!window.AudioWorkletNode) return "This browser lacks AudioWorklet.";
  return null;
}

languages();
buildInfo();
controls();
const problem = check();
if (problem) {
  status(problem, true);
  go.textContent = "Unsupported";
} else {
  status("Starting…");
  worker.postMessage({ type: "start" });
}
