import {test} from 'node:test';
import assert from 'node:assert/strict';
import {presentationKey, ringSlot, shouldWarp, shownPicture} from './pairing.mjs';

test('a slow mask keeps the frame it was measured on', () => {
  const ring = Array(8).fill(-1);
  let mask = -1;
  let inference = -1;
  const upload = id => {
    const keep = new Set([mask, inference].filter(frame => frame >= 0 && frame !== id));
    const index = ringSlot(ring, id, keep);
    assert.ok(index >= 0);
    ring[index] = id;
  };
  for (let id = 1; id <= 40; id++) {
    upload(id);
    if (id === 1) inference = 1;
    if (id === 20) {
      mask = 1;
      inference = 20;
    }
  }
  assert.ok(ring.includes(1));
  assert.ok(ring.includes(20));
  assert.ok(ring.includes(40));
});

test('recycle chooses the oldest frame that is not reserved', () => {
  const ring = [1, 2, 3, 4, 5, 6, 7, 8];
  assert.equal(ringSlot(ring, 9, new Set([1, 8])), 1);
  assert.equal(ringSlot(ring, 8, new Set([1, 8])), 7);
});

test('the output count stays on the mask until motion shows each new frame', () => {
  let last = null;
  let frames = 0;
  const present = picture => {
    const key = presentationKey(picture);
    if (key === last) return;
    last = key;
    frames++;
  };
  for (let id = 5; id <= 20; id++) {
    present(shownPicture({warped: false, frameId: id, maskFrameId: 5, matched: true, heldFrameId: -1}));
  }
  assert.equal(frames, 1);
  for (let id = 21; id <= 30; id++) {
    present(shownPicture({warped: true, frameId: id, maskFrameId: 5, matched: true, heldFrameId: 5}));
  }
  assert.equal(frames, 11);
});

test('the warp fills camera frames that will not get their own mask', () => {
  assert.equal(shouldWarp(false, 5, 4, 3), false);
  assert.equal(shouldWarp(true, 4, 4, 3), false);
  assert.equal(shouldWarp(true, 4, 4, 4), false);
  assert.equal(shouldWarp(true, 5, 4, -1), false);
  assert.equal(shouldWarp(true, 5, 4, 3), true);
});

test('the true mask landing on the frame already shown is not a new picture', () => {
  const warped = shownPicture({warped: true, frameId: 30, maskFrameId: 20, matched: true, heldFrameId: 20});
  const landed = shownPicture({warped: false, frameId: 30, maskFrameId: 30, matched: true, heldFrameId: 20});
  assert.equal(presentationKey(warped), '30');
  assert.equal(presentationKey(landed), presentationKey(warped));
});

test('a missing mask frame repeats the last matched picture', () => {
  const picture = shownPicture({warped: false, frameId: 30, maskFrameId: 12, matched: false, heldFrameId: 12});
  assert.deepEqual(picture, {live: false, id: 12});
  assert.equal(presentationKey(picture), presentationKey(
    shownPicture({warped: false, frameId: 31, maskFrameId: 12, matched: false, heldFrameId: 12})
  ));
});
