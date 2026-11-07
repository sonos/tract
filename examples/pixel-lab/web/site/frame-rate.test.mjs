import {test} from 'node:test';
import assert from 'node:assert/strict';
import {FrameRate, isNewSourceFrame} from './frame-rate.mjs';

test('frame rate follows completions, not reciprocal inference time',()=>{
  const output=new FrameRate(3), camera=new FrameRate(3);
  for(const [time,count] of [[0,1],[100,4],[200,7]]) camera.add(time,count);
  for(const [time,count] of [[20,1],[120,2],[220,3]]) output.add(time,count);
  assert.equal(camera.fps(),30);
  assert.equal(output.fps(),10);
  output.add(320,4);
  assert.equal(output.samples.length,3);
  assert.equal(output.fps(),10);
});
test('a repeated camera presentation is not a new source frame', () => {
  assert.equal(isNewSourceFrame(undefined, 0), true);
  assert.equal(isNewSourceFrame(0, 0), false);
  assert.equal(isNewSourceFrame(0, 1), true);
  assert.equal(isNewSourceFrame(4, 4), false);
});
test('a rate change shows up once the old samples leave the window', () => {
  const rate = new FrameRate();
  for (let i = 0; i < 10; i++) rate.add(i * 33, i);
  assert.ok(Math.abs(rate.fps() - 30.3) < 0.2);
  for (let i = 0; i < 4; i++) rate.add(10000 + i * 400, 100 + i);
  assert.ok(Math.abs(rate.fps() - 2.5) < 0.05);
});
test('frame rate waits for observations and resets if a counter restarts',()=>{
  const rate=new FrameRate();
  assert.equal(rate.fps(),null);
  rate.add(100,10);rate.add(200,13);
  assert.equal(rate.fps(),30);
  rate.add(300,1);
  assert.equal(rate.fps(),null);
});
