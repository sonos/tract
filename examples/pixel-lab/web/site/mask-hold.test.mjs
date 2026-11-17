import {test} from 'node:test';
import assert from 'node:assert/strict';
import {holdMask, showMask} from './mask-hold.mjs';

test('a sure score replaces the previous mask', () => {
  assert.equal(holdMask(1, 0).blended, 0);
  assert.equal(holdMask(0, 1).blended, 1);
  assert.equal(holdMask(1, 0).shown, 0);
  assert.equal(holdMask(0, 1).shown, 1);
});

test('a split score keeps most of the previous person', () => {
  const step = holdMask(1, 0.5);
  assert.ok(step.blended > 0.9);
  assert.equal(step.shown, 1);
});

test('a moderate background score does not knock out a person', () => {
  const step = holdMask(1, 0.3);
  assert.ok(step.blended > 0.7);
  assert.equal(step.shown, 1);
});

test('a flicker around one half stays solid person', () => {
  let previous = 1;
  for (const current of [0.45, 0.6, 0.4, 0.55, 0.35]) {
    const step = holdMask(previous, current);
    previous = step.blended;
    assert.equal(step.shown, 1);
  }
});

test('a sure background that repeats is dropped, and the picture uses the hard gate', () => {
  assert.equal(holdMask(1, 0).shown, 0);
  assert.equal(showMask(0.1), 0);
  assert.equal(showMask(0.69), 0.69);
  assert.equal(showMask(0.7), 1);
});
