// Geometry around MediaPipe's two hand models: the palm detector's anchors and
// boxes, the rotated square crop the landmark model reads, the landmarks
// projected back to the camera picture, and the crop the next frame tracks.
// Pixel coordinates throughout, y down. tract runs the models; see worker.js.

export const DETECTOR_SIZE = 192;
export const LANDMARKS_SIZE = 224;
export const ANCHORS = 2016;
export const COORDS = 18;

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
].map(([start, end]) => ({start, end}));

// SSD anchors of the full-range palm detector: stride 8 with two anchors per
// cell, then strides 16, 16, 16 grouped into six anchors per cell. Every
// anchor is a unit box at its cell center.
export function palmAnchors() {
  const anchors = [];
  for (const [stride, perCell] of [[8, 2], [16, 6]]) {
    const cells = Math.ceil(DETECTOR_SIZE / stride);
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        for (let k = 0; k < perCell; k++) anchors.push({x: (x + 0.5) / cells, y: (y + 0.5) / cells});
      }
    }
  }
  return anchors;
}

const ANCHOR_LIST = palmAnchors();

// Fit a width x height picture into a side x side square, centered, the rest
// left black.
export function letterbox(width, height, side = DETECTOR_SIZE) {
  const scale = side / Math.max(width, height);
  return {scale, padX: (side - width * scale) / 2, padY: (side - height * scale) / 2, side};
}

const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-100, Math.min(100, x))));

// Detector output (2016 x 18 offsets, then 2016 logits) to palms in camera
// pixels: a box and seven keypoints, keypoint 0 the wrist and 2 the middle
// finger's knuckle.
export function decodePalms(raw, box, minScore = 0.5) {
  const palms = [];
  const toImage = (u, v) => [(u * box.side - box.padX) / box.scale, (v * box.side - box.padY) / box.scale];
  for (let i = 0; i < ANCHORS; i++) {
    const score = sigmoid(raw[ANCHORS * COORDS + i]);
    if (score < minScore) continue;
    const a = ANCHOR_LIST[i];
    const r = raw.subarray ? raw.subarray(i * COORDS, (i + 1) * COORDS) : raw.slice(i * COORDS, (i + 1) * COORDS);
    const cx = r[0] / DETECTOR_SIZE + a.x;
    const cy = r[1] / DETECTOR_SIZE + a.y;
    const w = r[2] / DETECTOR_SIZE;
    const h = r[3] / DETECTOR_SIZE;
    const [x0, y0] = toImage(cx - w / 2, cy - h / 2);
    const [x1, y1] = toImage(cx + w / 2, cy + h / 2);
    const keypoints = [];
    for (let k = 0; k < 7; k++) {
      keypoints.push(toImage(r[4 + 2 * k] / DETECTOR_SIZE + a.x, r[5 + 2 * k] / DETECTOR_SIZE + a.y));
    }
    palms.push({score, box: [x0, y0, x1, y1], keypoints});
  }
  return palms;
}

export function boxIou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

// Weighted non-maximum suppression: each kept palm is the score-weighted mean
// of the palms that overlap it by more than the threshold.
export function weightedNms(palms, threshold = 0.3, limit = 2) {
  const left = [...palms].sort((a, b) => b.score - a.score);
  const kept = [];
  while (left.length && kept.length < limit) {
    const top = left[0];
    const group = [];
    for (let i = left.length - 1; i >= 0; i--) {
      if (boxIou(top.box, left[i].box) > threshold) group.push(...left.splice(i, 1));
    }
    const total = group.reduce((sum, palm) => sum + palm.score, 0);
    const mean = pick => group.reduce((sum, palm) => sum + pick(palm) * palm.score, 0) / total;
    kept.push({
      score: top.score,
      box: [0, 1, 2, 3].map(j => mean(palm => palm.box[j])),
      keypoints: top.keypoints.map((_, k) => [mean(palm => palm.keypoints[k][0]), mean(palm => palm.keypoints[k][1])]),
    });
  }
  return kept;
}

function normalizeAngle(angle) {
  return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
}

// The rotation that brings the from -> to direction pointing straight up.
export function uprightAngle(from, to) {
  return normalizeAngle(Math.PI / 2 - Math.atan2(-(to[1] - from[1]), to[0] - from[0]));
}

// Shift along the rotated axes, make square on the long side, then scale.
function transformRect({cx, cy, w, h, angle}, scale, shiftY) {
  const dx = -h * shiftY * Math.sin(angle);
  const dy = h * shiftY * Math.cos(angle);
  const size = Math.max(w, h) * scale;
  return {cx: cx + dx, cy: cy + dy, size, angle};
}

// The square, rotated crop the landmark model reads for a palm.
export function palmRoi(palm) {
  const [x0, y0, x1, y1] = palm.box;
  const angle = uprightAngle(palm.keypoints[0], palm.keypoints[2]);
  return transformRect({cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, angle}, 2.6, -0.5);
}

const ROI_POINTS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

// The crop the next frame tracks, from this frame's landmarks: upright from the
// wrist toward the knuckles, around the palm and the finger bases.
export function landmarksRoi(points) {
  const wrist = [points[0].x, points[0].y];
  const knuckles = [
    ((points[5].x + points[13].x) / 2 + points[9].x) / 2,
    ((points[5].y + points[13].y) / 2 + points[9].y) / 2,
  ];
  const angle = uprightAngle(wrist, knuckles);
  const subset = ROI_POINTS.map(i => points[i]);
  const xs = subset.map(p => p.x);
  const ys = subset.map(p => p.y);
  const ax = (Math.min(...xs) + Math.max(...xs)) / 2;
  const ay = (Math.min(...ys) + Math.max(...ys)) / 2;
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of subset) {
    const x = (p.x - ax) * cos - (p.y - ay) * sin;
    const y = (p.x - ax) * sin + (p.y - ay) * cos;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const px = (minX + maxX) / 2;
  const py = (minY + maxY) / 2;
  const cx = ax + Math.cos(angle) * px - Math.sin(angle) * py;
  const cy = ay + Math.sin(angle) * px + Math.cos(angle) * py;
  return transformRect({cx, cy, w: maxX - minX, h: maxY - minY, angle}, 2.0, -0.1);
}

// Canvas transform from camera pixels to crop pixels, for setTransform.
export function cropTransform(roi, side = LANDMARKS_SIZE) {
  const k = side / roi.size;
  const cos = Math.cos(-roi.angle) * k;
  const sin = Math.sin(-roi.angle) * k;
  const e = side / 2 - (cos * roi.cx - sin * roi.cy);
  const f = side / 2 - (sin * roi.cx + cos * roi.cy);
  return [cos, sin, -sin, cos, e, f];
}

// Landmark model output (21 x (x, y, z) in crop pixels) to camera pixels.
// z is scaled like x and y, relative to the wrist.
export function projectLandmarks(raw, roi, side = LANDMARKS_SIZE) {
  const cos = Math.cos(roi.angle);
  const sin = Math.sin(roi.angle);
  const k = roi.size / side;
  const points = [];
  for (let i = 0; i < 21; i++) {
    const u = (raw[3 * i] - side / 2) * k;
    const v = (raw[3 * i + 1] - side / 2) * k;
    points.push({x: roi.cx + cos * u - sin * v, y: roi.cy + sin * u + cos * v, z: raw[3 * i + 2] * k});
  }
  return points;
}

export function roiBox(roi) {
  const r = roi.size / 2;
  return [roi.cx - r, roi.cy - r, roi.cx + r, roi.cy + r];
}

// One Euro filter on every landmark coordinate, its speed measured relative to
// the hand's size, so a still hand stops shaking and a fast one does not lag.
export class LandmarkSmoother {
  constructor({minCutoff = 0.05, beta = 80, derivateCutoff = 1} = {}) {
    Object.assign(this, {minCutoff, beta, derivateCutoff});
    this.last = null;
  }

  apply(points, timeMs) {
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const scale = ((Math.max(...xs) - Math.min(...xs)) + (Math.max(...ys) - Math.min(...ys))) / 2;
    if (!this.last || !(scale > 0) || timeMs <= this.last.time) {
      this.last = {time: timeMs, raw: points.map(p => [p.x, p.y, p.z]), out: points.map(p => [p.x, p.y, p.z]), dx: points.map(() => [0, 0, 0])};
      return points;
    }
    const rate = 1000 / (timeMs - this.last.time);
    const alpha = cutoff => 1 / (1 + rate / (2 * Math.PI * cutoff));
    const ad = alpha(this.derivateCutoff);
    const out = points.map((p, i) => [p.x, p.y, p.z].map((value, j) => {
      const dvalue = (value - this.last.raw[i][j]) / scale * rate;
      const dx = this.last.dx[i][j] + ad * (dvalue - this.last.dx[i][j]);
      this.last.dx[i][j] = dx;
      const a = alpha(this.minCutoff + this.beta * Math.abs(dx));
      return this.last.out[i][j] + a * (value - this.last.out[i][j]);
    }));
    this.last.time = timeMs;
    this.last.raw = points.map(p => [p.x, p.y, p.z]);
    this.last.out = out;
    return out.map(([x, y, z]) => ({x, y, z}));
  }
}
