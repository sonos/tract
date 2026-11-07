import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createCompletionPoll, isFirefox, nextPollDelay} from './firefox-poll.mjs';

test('only Firefox gets the completion poll', () => {
  assert.equal(isFirefox('Mozilla/5.0 (Macintosh) Gecko/20100101 Firefox/155.0'), true);
  assert.equal(isFirefox('Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/156.0.0.0 Safari/537.36'), false);
  assert.equal(isFirefox('Mozilla/5.0 (Macintosh) Version/18.0 Safari/605.1.15'), false);
  assert.equal(isFirefox(''), false);
});

test('the kick starts at half a millisecond and caps at 8', () => {
  assert.equal(nextPollDelay(0), 0.5);
  assert.equal(nextPollDelay(0.5), 1);
  assert.equal(nextPollDelay(1), 2);
  assert.equal(nextPollDelay(2), 4);
  assert.equal(nextPollDelay(4), 8);
  assert.equal(nextPollDelay(8), 8);
});

function harness() {
  const submits = [];
  const pending = [];
  let nextId = 1;
  const device = {queue: {submit(buffers) { submits.push(buffers); }}};
  return {
    submits,
    pending,
    device,
    schedule(fn, ms) {
      const id = nextId++;
      pending.push({id, fn, ms});
      return id;
    },
    clear(id) {
      const index = pending.findIndex(timer => timer.id === id);
      if (index >= 0) pending.splice(index, 1);
    },
    fire() {
      const timer = pending.shift();
      timer.fn();
    },
  };
}

test('Chrome does not submit while waiting for completion', async () => {
  const clock = harness();
  const poll = createCompletionPoll({firefox: false, device: clock.device, schedule: clock.schedule, clear: clock.clear});
  assert.equal(await poll.track(Promise.resolve('done')), 'done');
  assert.equal(clock.pending.length, 0);
  assert.equal(clock.submits.length, 0);
});

test('Firefox submits an empty queue until the completion settles', async () => {
  const clock = harness();
  const poll = createCompletionPoll({firefox: true, device: clock.device, schedule: clock.schedule, clear: clock.clear});
  let resolve;
  const tracked = poll.track(new Promise(done => { resolve = done; }));
  assert.deepEqual(clock.pending.map(timer => timer.ms), [0.5]);
  clock.fire();
  assert.deepEqual(clock.submits, [[]]);
  assert.deepEqual(clock.pending.map(timer => timer.ms), [1]);
  clock.fire();
  assert.deepEqual(clock.pending.map(timer => timer.ms), [2]);
  resolve('mask');
  assert.equal(await tracked, 'mask');
  assert.equal(clock.pending.length, 0);
});

test('one device keeps a single kicker until every completion has settled', async () => {
  const clock = harness();
  const poll = createCompletionPoll({firefox: true, device: clock.device, schedule: clock.schedule, clear: clock.clear});
  let resolveA, resolveB;
  const first = poll.track(new Promise(done => { resolveA = done; }));
  const second = poll.track(new Promise(done => { resolveB = done; }));
  assert.equal(clock.pending.length, 1);
  resolveA();
  await first;
  assert.equal(clock.pending.length, 1);
  clock.fire();
  assert.equal(clock.submits.length, 1);
  resolveB();
  await second;
  assert.equal(clock.pending.length, 0);
});

test('a rejected completion stops the kicker', async () => {
  const clock = harness();
  const poll = createCompletionPoll({firefox: true, device: clock.device, schedule: clock.schedule, clear: clock.clear});
  const tracked = poll.track(Promise.reject(new Error('lost')));
  await assert.rejects(tracked, /lost/);
  assert.equal(clock.pending.length, 0);
  assert.equal(clock.submits.length, 0);
});

test('a failed kick stops rather than spinning', () => {
  const clock = harness();
  clock.device.queue.submit = () => { throw new Error('device lost'); };
  const poll = createCompletionPoll({firefox: true, device: clock.device, schedule: clock.schedule, clear: clock.clear});
  let resolve;
  poll.track(new Promise(done => { resolve = done; }));
  clock.fire();
  assert.equal(clock.pending.length, 0);
  resolve();
});
