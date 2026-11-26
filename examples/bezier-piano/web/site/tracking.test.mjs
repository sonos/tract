import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ANCHORS, COORDS, DETECTOR_SIZE, LANDMARKS_SIZE, LandmarkSmoother, boxIou, cropTransform, decodePalms,
  landmarksRoi, letterbox, palmAnchors, palmRoi, projectLandmarks, uprightAngle, weightedNms,
} from './tracking.mjs';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('2016 anchors: 24x24x2 at stride 8, then 12x12x6', () => {
  const anchors = palmAnchors();
  assert.equal(anchors.length, ANCHORS);
  assert.deepEqual(anchors[0], {x: 0.5 / 24, y: 0.5 / 24});
  assert.deepEqual(anchors[1], anchors[0]);
  assert.deepEqual(anchors[1152], {x: 0.5 / 12, y: 0.5 / 12});
  assert.deepEqual(anchors[ANCHORS - 1], {x: 11.5 / 12, y: 11.5 / 12});
});

test('a palm decodes at its anchor, through the letterbox, into camera pixels', () => {
  const raw = new Float32Array(ANCHORS * (COORDS + 1)).fill(-50);
  const i = 1152 + 6 * (12 * 5 + 7); // stride 16, cell x=7 y=5
  raw.set([0, 0, 48, 48, 0, 24, 0, 0, 0, -24], i * COORDS);
  raw[ANCHORS * COORDS + i] = 4;
  const box = letterbox(1280, 720);
  const [palm] = decodePalms(raw, box);
  const cx = (7.5 / 12 * DETECTOR_SIZE - box.padX) / box.scale;
  const cy = (5.5 / 12 * DETECTOR_SIZE - box.padY) / box.scale;
  close(palm.box[0], cx - 24 / box.scale, 1e-3);
  close(palm.box[3], cy + 24 / box.scale, 1e-3);
  close(palm.keypoints[0][1], cy + 24 / box.scale, 1e-3);
  close(palm.score, 1 / (1 + Math.exp(-4)));
  assert.equal(decodePalms(raw, box).length, 1);
});

test('weighted NMS merges overlapping palms and keeps the separate one', () => {
  const kp = (x, y) => Array.from({length: 7}, () => [x, y]);
  const palms = [
    {score: 0.9, box: [0, 0, 10, 10], keypoints: kp(5, 5)},
    {score: 0.6, box: [1, 1, 11, 11], keypoints: kp(6, 6)},
    {score: 0.7, box: [50, 50, 60, 60], keypoints: kp(55, 55)},
  ];
  const kept = weightedNms(palms);
  assert.equal(kept.length, 2);
  close(kept[0].box[0], 0.6 / 1.5, 1e-9);
  close(kept[0].keypoints[0][0], (5 * 0.9 + 6 * 0.6) / 1.5, 1e-9);
  assert.equal(kept[1].box[0], 50);
  assert.equal(boxIou([0, 0, 1, 1], [2, 2, 3, 3]), 0);
});

test('upright angle: a hand pointing up is 0, pointing right is a quarter turn', () => {
  close(uprightAngle([0, 10], [0, 0]), 0);
  close(uprightAngle([0, 0], [10, 0]), Math.PI / 2);
  close(Math.abs(uprightAngle([0, 0], [0, 10])), Math.PI);
});

test('palm crop: upright palm, square, scaled 2.6, shifted half a box up', () => {
  const keypoints = Array.from({length: 7}, () => [100, 100]);
  keypoints[0] = [100, 120];
  keypoints[2] = [100, 80];
  const roi = palmRoi({box: [80, 80, 120, 110], keypoints});
  close(roi.angle, 0);
  close(roi.size, 40 * 2.6);
  close(roi.cx, 100);
  close(roi.cy, 95 - 15);
});

test('crop transform and landmark projection are inverses', () => {
  const roi = {cx: 400, cy: 300, size: 180, angle: 0.7};
  const [a, b, c, d, e, f] = cropTransform(roi);
  const raw = [];
  const points = [[350, 260], [400, 300], [470, 330]];
  for (const [x, y] of points) raw.push(a * x + c * y + e, b * x + d * y + f, 0);
  while (raw.length < 63) raw.push(LANDMARKS_SIZE / 2, LANDMARKS_SIZE / 2, 0);
  const back = projectLandmarks(raw, roi);
  points.forEach(([x, y], i) => { close(back[i].x, x, 1e-6); close(back[i].y, y, 1e-6); });
  close(back[1].x, 400, 1e-9);
  close(raw[3], LANDMARKS_SIZE / 2, 1e-9);
});

test('a crop rotated a quarter turn puts the fingers at the top of the crop', () => {
  // Hand pointing right: the crop's up must map to the camera's right.
  const roi = {cx: 0, cy: 0, size: LANDMARKS_SIZE, angle: Math.PI / 2};
  const raw = Array(63).fill(LANDMARKS_SIZE / 2);
  raw[1] = 0; // landmark 0 at the top center of the crop
  const [top] = projectLandmarks(raw, roi);
  close(top.x, LANDMARKS_SIZE / 2, 1e-9);
  close(top.y, 0, 1e-9);
});

test('landmarks crop follows the hand and is twice the palm', () => {
  const points = Array.from({length: 21}, () => ({x: 0, y: 0, z: 0}));
  points[0] = {x: 100, y: 200, z: 0};
  for (const [i, x] of [[5, 80], [9, 100], [13, 120], [17, 135]]) points[i] = {x, y: 150, z: 0};
  for (const [i, x] of [[1, 70], [2, 65], [3, 62], [6, 80], [10, 100], [14, 120], [18, 135]]) points[i] = {x, y: i < 4 ? 180 : 130, z: 0};
  const roi = landmarksRoi(points);
  close(roi.angle, 0);
  close(roi.size, 2 * 73, 1e-9);
  close(roi.cx, (62 + 135) / 2, 1e-9);
  close(roi.cy, 165 - 0.1 * 70, 1e-9);
});

test('smoother passes the first frame and damps a jitter on a still hand', () => {
  const hand = (dx = 0) => Array.from({length: 21}, (_, i) => ({x: 100 + (i % 5) * 20 + dx, y: 100 + Math.floor(i / 5) * 20, z: 0}));
  const smoother = new LandmarkSmoother();
  assert.deepEqual(smoother.apply(hand(), 0), hand());
  const out = smoother.apply(hand(2), 33);
  assert.ok(out[0].x > 100 && out[0].x < 102);
  let moved;
  for (let t = 1; t <= 10; t++) moved = smoother.apply(hand(2 + t * 30), 33 + t * 33);
  assert.ok(moved[0].x > 100 + 2 + 270, `fast motion lags too much: ${moved[0].x}`);
});
