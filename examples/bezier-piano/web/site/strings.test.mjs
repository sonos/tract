import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  SCALES, cubicAt, sampleCurve, segmentIntersect, firstHit, pinchDistance,
  velocityGain, allowPluck, midiFor, noteName, stringBaseMidi, layoutStrings,
  fitCubic, assignHands, nearestControl,
} from './strings.mjs';

test('segment intersection meets in the middle of a cross', () => {
  const hit = segmentIntersect({x: 0, y: 0}, {x: 1, y: 1}, {x: 0, y: 1}, {x: 1, y: 0});
  assert.ok(hit);
  assert.ok(Math.abs(hit.t - 0.5) < 1e-9);
  assert.ok(Math.abs(hit.u - 0.5) < 1e-9);
});

test('a parallel miss is not a pluck', () => {
  const hit = segmentIntersect({x: 0, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}, {x: 1, y: 1});
  assert.equal(hit, null);
});

test('a fingertip crossing a flat string reports the position along it', () => {
  const curve = [
    {x: 0, y: 0.5}, {x: 0.33, y: 0.5}, {x: 0.66, y: 0.5}, {x: 1, y: 0.5},
  ];
  const samples = sampleCurve(curve, 64);
  const hit = firstHit({x: 0.25, y: 0.1}, {x: 0.25, y: 0.9}, samples);
  assert.ok(hit);
  assert.ok(Math.abs(hit.u - 0.25) < 0.02, `u=${hit.u}`);
  assert.equal(firstHit({x: 0.25, y: 0.1}, {x: 0.25, y: 0.2}, samples), null);
});

test('cubic ends are the outer control points', () => {
  const p = [{x: 0, y: 1}, {x: 1, y: 2}, {x: 2, y: -1}, {x: 3, y: 0}];
  assert.deepEqual(cubicAt(p, 0), {x: 0, y: 1});
  assert.deepEqual(cubicAt(p, 1), {x: 3, y: 0});
});

test('six strings are an octave apart from C3 up to C8', () => {
  for (let i = 0; i < 6; i++) assert.equal(stringBaseMidi(i, 6), 48 + i * 12);
  const major = SCALES.find(scale => scale.id === 'major');
  assert.equal(noteName(midiFor(5, 6, 0, major.degrees)), 'C3');
  assert.equal(noteName(midiFor(5, 6, 1, major.degrees)), 'C4');
  assert.equal(noteName(midiFor(0, 6, 0, major.degrees)), 'C8');
  const pentatonic = SCALES[0].degrees;
  const low = midiFor(5, 6, 0, pentatonic);
  const mid = midiFor(5, 6, 0.5, pentatonic);
  const high = midiFor(5, 6, 1, pentatonic);
  assert.ok(low < mid && mid < high);
  assert.equal(low % 12, 0);
});

test('more than six strings stay inside C3 to C8', () => {
  const bases = Array.from({length: 12}, (_, i) => stringBaseMidi(i, 12));
  assert.equal(bases[0], 48);
  assert.equal(bases[11], 108);
  for (let i = 1; i < bases.length; i++) assert.ok(bases[i] > bases[i - 1]);
});

test('velocity is quiet when slow and full when fast', () => {
  assert.equal(velocityGain(0), 0.15);
  assert.equal(velocityGain(100), 1);
  const mid = velocityGain(2);
  assert.ok(mid > 0.15 && mid < 1);
});

test('a string stays quiet inside the refractory gap', () => {
  const memory = new Map();
  assert.equal(allowPluck(memory, 'a', 1000), true);
  assert.equal(allowPluck(memory, 'a', 1050), false);
  assert.equal(allowPluck(memory, 'a', 1120), true);
  assert.equal(allowPluck(memory, 'b', 1050), true);
});

test('pinch distance is the thumb to index gap', () => {
  const landmarks = Array.from({length: 21}, () => ({x: 0, y: 0}));
  landmarks[4] = {x: 0, y: 0};
  landmarks[8] = {x: 0.03, y: 0.04};
  assert.ok(Math.abs(pinchDistance(landmarks) - 0.05) < 1e-9);
});

test('layout puts the first string above the last', () => {
  const strings = layoutStrings(6);
  assert.equal(strings.length, 6);
  assert.ok(strings[0].p[0].y < strings[5].p[0].y);
  assert.equal(strings[0].p.length, 4);
});

test('a fitted cubic passes through the stroke thirds', () => {
  const source = [
    {x: 0.1, y: 0.2}, {x: 0.3, y: 0.8}, {x: 0.7, y: 0.1}, {x: 0.9, y: 0.6},
  ];
  const samples = sampleCurve(source, 30);
  const fit = fitCubic(samples);
  assert.ok(fit);
  const cum = [0];
  for (let i = 1; i < samples.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(samples[i].x - samples[i - 1].x, samples[i].y - samples[i - 1].y));
  }
  const at = u => {
    const target = u * cum[cum.length - 1];
    let i = 1;
    while (i < cum.length - 1 && cum[i] < target) i++;
    const span = cum[i] - cum[i - 1] || 1;
    const t = (target - cum[i - 1]) / span;
    const a = samples[i - 1];
    const b = samples[i];
    return {x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t};
  };
  for (const u of [0, 1 / 3, 2 / 3, 1]) {
    const got = cubicAt(fit, u);
    const want = at(u);
    assert.ok(Math.hypot(got.x - want.x, got.y - want.y) < 1e-6, `${u} drifted`);
  }
});

test('hands keep their identity when the wrists stay put', () => {
  const previous = [{x: 0.2, y: 0.2}, {x: 0.8, y: 0.4}];
  const hands = [
    [{x: 0.81, y: 0.41}],
    [{x: 0.19, y: 0.22}],
  ];
  const assigned = assignHands(previous, hands);
  assert.equal(assigned[0].prev, 1);
  assert.equal(assigned[1].prev, 0);
});

test('the nearest control point wins', () => {
  const strings = [{p: [{x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 2}, {x: 3, y: 3}]}];
  assert.equal(nearestControl(strings, {x: 1.1, y: 0.9}, 0.3).i, 1);
  assert.equal(nearestControl(strings, {x: 5, y: 5}, 0.3), null);
});
