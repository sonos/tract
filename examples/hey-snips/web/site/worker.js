// Runs the detector off the page's thread: samples come straight from the audio worklet, scores go
// back to the page, one per 10 ms frame. The audio context runs at the model's 48 kHz.
let loaded, detector, samples = 0, busy = 0;
const post = message => postMessage(message);

async function load() {
  post({status: 'Loading Rust / Wasm…'});
  const wasm = await import('./pkg/hey_snips_web.js');
  await wasm.default({module_or_path: new URL('./pkg/hey_snips_web_bg.wasm', import.meta.url)});
  post({status: 'Loading the model…'});
  const response = await fetch(new URL('./models/hey_snips.nnef.tar', import.meta.url));
  if (!response.ok) throw new Error(`Model: HTTP ${response.status}`);
  return {wasm, model: new Uint8Array(await response.arrayBuffer())};
}

function feed({data, target}) {
  try {
    const start = performance.now();
    const scores = detector.push(data);
    busy += performance.now() - start;
    samples += data.length;
    if (scores.length) post({scores, samples, busy});
  } catch (error) {
    target.onmessage = null;
    post({error: `${error.message || error}`});
  }
}

onmessage = async ({data}) => {
  try {
    loaded ??= load();
    const {wasm, model} = await loaded;
    detector?.free();
    detector = new wasm.Detector(model);
    samples = 0;
    busy = 0;
    data.port.onmessage = feed;
    post({ready: true});
  } catch (error) {
    loaded = null;
    post({error: `${error.message || error}`});
  }
};
