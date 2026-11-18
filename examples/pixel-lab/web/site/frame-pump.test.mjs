import {test} from 'node:test';
import assert from 'node:assert/strict';
import {emptyPump, rememberFrame, startSend, finishSend} from './frame-pump.mjs';

test('a new camera frame starts a send, and the same frame does not', () => {
  let pump = emptyPump();
  const first = rememberFrame(pump, 1);
  assert.equal(first.fresh, true);
  const started = startSend(first.pump);
  assert.equal(started.start, true);
  assert.equal(started.pump.busy, true);
  pump = started.pump;
  const again = rememberFrame(pump, 1);
  assert.equal(again.fresh, false);
  assert.equal(startSend(again.pump).start, false);
});

test('a frame that arrives during a send goes out when that send finishes', () => {
  let pump = startSend(rememberFrame(emptyPump(), 1).pump).pump;
  const during = rememberFrame(pump, 2);
  assert.equal(during.fresh, true);
  assert.equal(startSend(during.pump).start, false);
  pump = finishSend(during.pump);
  const next = startSend(pump);
  assert.equal(next.start, true);
  assert.equal(next.pump.sent, 2);
  pump = finishSend(next.pump);
  assert.equal(startSend(pump).start, false);
});

test('several frames during one send leave only the newest', () => {
  let pump = startSend(rememberFrame(emptyPump(), 4).pump).pump;
  pump = rememberFrame(pump, 5).pump;
  pump = rememberFrame(pump, 6).pump;
  const next = startSend(finishSend(pump));
  assert.equal(next.start, true);
  assert.equal(next.pump.sent, 6);
});
