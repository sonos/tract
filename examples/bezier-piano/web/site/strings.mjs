// Geometry, pitch, and pinch tests for the strings. No DOM, so node can run it.

export const FINGERTIPS = [4, 8, 12, 16, 20];
export const THUMB = 4;
export const INDEX = 8;

// A pinch is thumb tip to index tip. The open/close gap is hysteresis so a
// shaky pinch does not flicker between drag and pluck.
export const PINCH_ON = 0.05;
export const PINCH_OFF = 0.075;

export const SCALES = [
  {id: 'major-pentatonic', name: 'Major pentatonic', degrees: [0, 2, 4, 7, 9, 12]},
  {id: 'minor-pentatonic', name: 'Minor pentatonic', degrees: [0, 3, 5, 7, 10, 12]},
  {id: 'major', name: 'Major', degrees: [0, 2, 4, 5, 7, 9, 11, 12]},
  {id: 'natural-minor', name: 'Natural minor', degrees: [0, 2, 3, 5, 7, 8, 10, 12]},
];

export function scaleById(id) {
  return SCALES.find(scale => scale.id === id) || SCALES[0];
}

export function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

export function cubicAt(p, t) {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return {
    x: a * p[0].x + b * p[1].x + c * p[2].x + d * p[3].x,
    y: a * p[0].y + b * p[1].y + c * p[2].y + d * p[3].y,
  };
}

export function sampleCurve(p, steps = 64) {
  const points = [];
  for (let i = 0; i <= steps; i++) points.push(cubicAt(p, i / steps));
  return points;
}

// Segment AB against segment CD. t is along AB, u along CD.
export function segmentIntersect(a, b, c, d) {
  const rx = b.x - a.x;
  const ry = b.y - a.y;
  const sx = d.x - c.x;
  const sy = d.y - c.y;
  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < 1e-10) return null;
  const qx = c.x - a.x;
  const qy = c.y - a.y;
  const t = (qx * sy - qy * sx) / denom;
  const u = (qx * ry - qy * rx) / denom;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return {t, u};
}

// First place the fingertip segment crosses the sampled curve.
// `u` is the parameter along the curve, 0 at the first control point.
export function firstHit(a, b, samples) {
  let best = null;
  const last = samples.length - 1;
  if (last < 1) return null;
  for (let i = 0; i < last; i++) {
    const hit = segmentIntersect(a, b, samples[i], samples[i + 1]);
    if (!hit) continue;
    if (!best || hit.t < best.t) best = {t: hit.t, u: (i + hit.u) / last};
  }
  return best;
}

export function pinchDistance(landmarks) {
  const a = landmarks[THUMB];
  const b = landmarks[INDEX];
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function pinchPoint(landmarks) {
  return {
    x: (landmarks[THUMB].x + landmarks[INDEX].x) / 2,
    y: (landmarks[THUMB].y + landmarks[INDEX].y) / 2,
  };
}

export function nearestControl(strings, point, maxDist) {
  let best = null;
  for (let s = 0; s < strings.length; s++) {
    const controls = strings[s].p;
    for (let i = 0; i < controls.length; i++) {
      const d = Math.hypot(controls[i].x - point.x, controls[i].y - point.y);
      if (d <= maxDist && (!best || d < best.d)) best = {s, i, d};
    }
  }
  return best;
}

// speed is normalized screen-lengths per second. A slow cross is still audible.
export function velocityGain(speed) {
  if (!Number.isFinite(speed)) return 0.15;
  return clamp((speed - 0.35) / 3.2, 0.15, 1);
}

// One pluck per finger per string until the gap has passed, so a fingertip
// that chatters on the polyline does not retrigger every frame.
export function allowPluck(memory, key, now, refractoryMs = 110) {
  const prev = memory.get(key);
  if (prev !== undefined && now - prev < refractoryMs) return false;
  memory.set(key, now);
  return true;
}

const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function noteName(midi) {
  const n = ((midi % 12) + 12) % 12;
  const oct = Math.floor(midi / 12) - 1;
  return NAMES[n] + oct;
}

// Bottom string is the low octave. Six strings are C3 through C8. More than
// six squeeze into that same range so the top stays playable.
export function stringBaseMidi(indexFromBottom, count) {
  const low = 48;
  const high = 108;
  if (count <= 1) return 60;
  const natural = low + (count - 1) * 12;
  if (natural <= high) return low + indexFromBottom * 12;
  return Math.round(low + (indexFromBottom * (high - low)) / (count - 1));
}

export function midiFor(indexFromTop, count, t, degrees) {
  const fromBottom = count - 1 - indexFromTop;
  const base = stringBaseMidi(fromBottom, count);
  const i = clamp(Math.round(clamp(t, 0, 1) * (degrees.length - 1)), 0, degrees.length - 1);
  return base + degrees[i];
}

export function layoutStrings(count) {
  const n = clamp(count, 1, 12);
  const strings = [];
  for (let i = 0; i < n; i++) {
    const y = n === 1 ? 0.5 : 0.16 + (0.68 * i) / (n - 1);
    const bow = 0.045 * (i % 2 === 0 ? 1 : -1);
    strings.push({
      p: [
        {x: 0.05, y},
        {x: 0.34, y: clamp(y + bow, 0.04, 0.96)},
        {x: 0.66, y: clamp(y - bow * 0.65, 0.04, 0.96)},
        {x: 0.95, y},
      ],
    });
  }
  return strings;
}

export function pathLength(points) {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  return total;
}

function pointAlong(points, cum, total, u) {
  const target = u * total;
  let i = 1;
  while (i < cum.length - 1 && cum[i] < target) i++;
  const span = cum[i] - cum[i - 1] || 1;
  const t = clamp((target - cum[i - 1]) / span, 0, 1);
  const a = points[i - 1];
  const b = points[i];
  return {x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t};
}

// Fit a cubic that passes through the stroke's ends and its one-third and
// two-third points. Four samples determine the two free control points.
export function fitCubic(points) {
  if (!points || points.length < 2) return null;
  const cum = [0];
  for (let i = 1; i < points.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y));
  }
  const total = cum[cum.length - 1];
  if (total < 1e-4) return null;
  const p0 = points[0];
  const p3 = points[points.length - 1];
  const q1 = pointAlong(points, cum, total, 1 / 3);
  const q2 = pointAlong(points, cum, total, 2 / 3);
  // B(1/3) = q1 and B(2/3) = q2, with p0 and p3 fixed.
  // (4/9) p1 + (2/9) p2 = r1, (2/9) p1 + (4/9) p2 = r2.
  const a = 4 / 9;
  const b = 2 / 9;
  const det = a * a - b * b;
  const solve = (axis) => {
    const r1 = q1[axis] - (8 / 27) * p0[axis] - (1 / 27) * p3[axis];
    const r2 = q2[axis] - (1 / 27) * p0[axis] - (8 / 27) * p3[axis];
    return {
      c1: (a * r1 - b * r2) / det,
      c2: (a * r2 - b * r1) / det,
    };
  };
  const x = solve('x');
  const y = solve('y');
  return [
    {x: p0.x, y: p0.y},
    {x: x.c1, y: y.c1},
    {x: x.c2, y: y.c2},
    {x: p3.x, y: p3.y},
  ];
}

// previous entries are {x, y} wrists. prev is the matched previous index, or -1.
export function assignHands(previous, hands, maxDist = 0.22) {
  const used = new Set();
  const assigned = [];
  for (let i = 0; i < hands.length; i++) {
    const wrist = hands[i][0];
    let best = -1;
    let bestD = maxDist;
    for (let j = 0; j < previous.length; j++) {
      if (used.has(j)) continue;
      const d = Math.hypot(wrist.x - previous[j].x, wrist.y - previous[j].y);
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    if (best >= 0) used.add(best);
    assigned.push({index: i, prev: best});
  }
  return assigned;
}
