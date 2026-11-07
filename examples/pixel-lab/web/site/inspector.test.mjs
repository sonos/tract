import {test} from 'node:test';
import assert from 'node:assert/strict';
import {duty, frameBudget, inspectorView, motionBytes, sparkPoints, spread, textureBytes} from './inspector.mjs';

test('a flat series is a level line and an empty series draws nothing', () => {
  assert.equal(sparkPoints([]), '');
  const line = sparkPoints([4, 4, 4]).split(' ');
  assert.equal(line.length, 3);
  assert.equal(line[0].split(',')[1], line[2].split(',')[1]);
});

test('duty is the mask time against the camera frame', () => {
  assert.equal(Math.round(duty(14.6, 30)), 44);
  assert.equal(Math.round(duty(300, 30)), 900);
});

test('spread is the gap variation', () => {
  assert.equal(spread([0]), null);
  assert.equal(spread([0, 10, 20, 30]), 0);
});

test('texture bytes count the camera frames, the mask, and motion once it exists', () => {
  const still = textureBytes({width: 1280, height: 720, maskWidth: 1280, maskHeight: 720});
  assert.equal(still, 1280 * 720 * 4 * 12);
  assert.equal(motionBytes(), 4233600);
  assert.equal(textureBytes({width: 320, height: 180, maskWidth: 256, maskHeight: 144, motion: true}), 320 * 180 * 4 * 11 + 256 * 144 * 4 + 4233600);
});

test('the four meters use the samples they are given', () => {
  const view = inspectorView({
    cameraFps: 30,
    cameraTimes: [0, 33, 66, 100],
    captureWidth: 1280,
    captureHeight: 720,
    latencies: [14, 15, 16],
    tensor: '512×288',
    compositeMs: [8, 9, 10],
    bytes: 5e6,
    outputFps: 10,
    outputTimes: [0, 100, 200],
    sent: 100,
    drops: 0,
  });
  assert.equal(view.ingest.value, '30.0 FPS');
  assert.match(view.ingest.detail, /^1280×720 \| σ /);
  assert.equal(view.inference.value, '15.0 ms');
  assert.match(view.inference.detail, /^Duty \d+% \| Tensor 512×288$/);
  assert.equal(view.composite.value, '9.0 ms');
  assert.equal(view.composite.detail, 'GPU textures 5.0 MB');
  assert.equal(view.output.value, '10.0 FPS');
  assert.match(view.output.detail, /Drop 0\.0%$/);
  assert.ok(view.ingest.points.includes(','));
  assert.ok(view.output.points.includes(','));
  assert.equal(view.budget.text, '24.0 / 33.3 ms');
  assert.ok(view.budget.fill < 1);
  assert.equal(view.budget.over, false);
});

test('the frame budget is inference plus compositing against one captured frame', () => {
  const inside = frameBudget(14.6, 8.7, 30);
  assert.equal(inside.text, '23.3 / 33.3 ms');
  assert.ok(Math.abs(inside.fill - 23.3 / (1000 / 30)) < 1e-9);
  const over = frameBudget(300, 8, 30);
  assert.equal(over.fill, 1);
  assert.equal(over.over, true);
  assert.equal(frameBudget(null, null).text, '— / — ms');
  assert.equal(frameBudget(null, null, 29.9).text, '— / 33.4 ms');
  const slowCamera = frameBudget(14.6, 8.7, 24);
  assert.equal(slowCamera.text, '23.3 / 41.7 ms');
  assert.equal(slowCamera.over, false);
  const fastCamera = frameBudget(14.6, 8.7, 60);
  assert.equal(fastCamera.text, '23.3 / 16.7 ms');
  assert.equal(fastCamera.fill, 1);
  assert.equal(fastCamera.over, true);
  const view = inspectorView({cameraFps: 24, latencies: [10], compositeMs: [5]});
  assert.equal(view.budget.text, '15.0 / 41.7 ms');
  assert.equal(view.budget.over, false);
});
