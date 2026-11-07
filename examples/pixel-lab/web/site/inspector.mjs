// Live pipeline meters. The numbers are the samples the page already has:
// camera arrivals, mask time, composite finish time, and pictures that changed.

const SPARK_WIDTH = 120;
const SPARK_HEIGHT = 36;

export function sparkPoints(values, width = SPARK_WIDTH, height = SPARK_HEIGHT) {
  if (!values.length) return '';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  return values.map((value, index) => {
    const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * (width - 1);
    const y = span === 0 ? height / 2 : height - 2 - ((value - min) / span) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

// Spread of the gaps between timestamps, in milliseconds.
export function spread(times) {
  if (!times || times.length < 2) return null;
  const gaps = [];
  for (let i = 1; i < times.length; i++) gaps.push(times[i] - times[i - 1]);
  const mean = gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length;
  const variance = gaps.reduce((sum, gap) => sum + (gap - mean) ** 2, 0) / gaps.length;
  return Math.sqrt(variance);
}

// Inference time as a percent of the camera frame. A missing camera rate uses 30 fps.
export function duty(latencyMs, fps) {
  if (!(latencyMs >= 0)) return null;
  const budget = fps > 0 ? 1000 / fps : 1000 / 30;
  return latencyMs / budget * 100;
}

const rgba8 = (width, height) => width * height * 4;
const rgba16 = (width, height) => width * height * 8;

// Textures MotionWarp.create allocates at its default 320×180 pyramid.
export function motionBytes() {
  const sizes = [];
  for (let level = 0, width = 320, height = 180; level < 5; level++, width = Math.ceil(width / 2), height = Math.ceil(height / 2)) {
    sizes.push([width, height]);
  }
  const pyramid = sizes.reduce((sum, [width, height]) => sum + rgba16(width, height), 0);
  const flows = sizes.reduce((sum, [width, height]) => sum + rgba16(width, height) * 2, 0);
  const [gridWidth, gridHeight] = sizes[2];
  const [coarseWidth, coarseHeight] = sizes[4];
  return pyramid * 4 + flows + rgba16(coarseWidth, coarseHeight) + rgba16(gridWidth, gridHeight) * 3 + rgba16(320, 180);
}

// Resident compositor textures: 8 camera frames, 2 blur passes, one full frame,
// and the mask. Motion textures are added once that path has been created.
export function textureBytes({width, height, maskWidth, maskHeight, motion = false}) {
  if (!(width > 0) || !(height > 0) || !(maskWidth > 0) || !(maskHeight > 0)) return 0;
  const frames = rgba8(width, height) * 11 + rgba8(maskWidth, maskHeight);
  return frames + (motion ? motionBytes() : 0);
}

const median = values => values.length ? [...values].sort((a, b) => a - b)[values.length >> 1] : null;
const last = (values, count = 48) => values.slice(-count);
const RATE_WINDOW_MS = 2000;

// Keep the recent arrivals. A graph of the last minute still shows the old speed.
function recentTimes(times, windowMs = RATE_WINDOW_MS) {
  if (times.length < 2) return times;
  const end = times.at(-1);
  const kept = times.filter(time => end - time <= windowMs);
  return kept.length >= 2 ? kept : times.slice(-2);
}
const fpsText = value => value === null || value === undefined ? '—' : `${value.toFixed(1)} FPS`;
const msText = value => value === null || value === undefined ? '—' : `${value.toFixed(1)} ms`;
const mbText = bytes => bytes > 0 ? `${(bytes / 1e6).toFixed(1)} MB` : '—';

export function inspectorView({
  cameraFps = null,
  cameraTimes = [],
  captureWidth = 0,
  captureHeight = 0,
  latencies = [],
  tensor = '',
  compositeMs = [],
  bytes = 0,
  outputFps = null,
  outputTimes = [],
  sent = 0,
  drops = 0,
} = {}) {
  const cameraWindow = recentTimes(cameraTimes);
  const outputWindow = recentTimes(outputTimes);
  const cameraSpread = spread(cameraWindow);
  const size = captureWidth > 0 && captureHeight > 0 ? `${captureWidth}×${captureHeight}` : '';
  const inference = median(last(latencies, 8));
  const composite = median(last(compositeMs, 8));
  const outputSpread = spread(outputWindow);
  const drop = sent > 0 ? drops / sent * 100 : null;
  return {
    ingest: {
      value: fpsText(cameraFps),
      detail: size ? `${size} | ${cameraSpread === null ? '—' : `σ ${cameraSpread.toFixed(1)} ms`}` : 'Waiting for the camera',
      points: sparkPoints(last(cameraWindow.length > 1 ? cameraSteps(cameraWindow) : [])),
    },
    inference: {
      value: msText(inference),
      detail: inference === null ? 'Waiting for a mask' : `Duty ${duty(inference, cameraFps).toFixed(0)}% | Tensor ${tensor || '—'}`,
      points: sparkPoints(last(latencies)),
    },
    composite: {
      value: msText(composite),
      detail: composite === null ? 'Waiting for a frame' : `GPU textures ${mbText(bytes)}`,
      points: sparkPoints(last(compositeMs)),
    },
    output: {
      value: fpsText(outputFps),
      detail: outputFps === null ? 'Waiting for a picture' : `σ ${outputSpread === null ? '—' : `${outputSpread.toFixed(1)} ms`} | Drop ${drop === null ? '—' : `${drop.toFixed(1)}%`}`,
      points: sparkPoints(last(outputWindow.length > 1 ? cameraSteps(outputWindow) : [])),
    },
    budget: frameBudget(inference, composite, cameraFps),
  };
}

// One captured frame. The bar is inference plus compositing against that interval.
export function frameBudget(inferenceMs, compositeMs, cameraFps = null) {
  const limit = cameraFps > 0 ? 1000 / cameraFps : null;
  const limitText = limit === null ? '— ms' : `${limit.toFixed(1)} ms`;
  if (inferenceMs === null && compositeMs === null) {
    return {text: `— / ${limitText}`, fill: 0, over: false};
  }
  const used = (inferenceMs ?? 0) + (compositeMs ?? 0);
  if (limit === null) return {text: `${used.toFixed(1)} / — ms`, fill: 0, over: false};
  return {
    text: `${used.toFixed(1)} / ${limitText}`,
    fill: Math.min(1, used / limit),
    over: used > limit,
  };
}

// Frame rate at each arrival, so the line follows the same quantity as the big number.
function cameraSteps(times) {
  const rates = [];
  for (let i = 1; i < times.length; i++) {
    const gap = times[i] - times[i - 1];
    if (gap > 0) rates.push(1000 / gap);
  }
  return rates;
}
