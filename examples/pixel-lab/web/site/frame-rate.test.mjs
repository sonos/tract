import {test} from 'node:test';
import assert from 'node:assert/strict';
import {FrameRate} from './frame-rate.mjs';

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
test('frame rate waits for observations and resets if a counter restarts',()=>{
  const rate=new FrameRate();
  assert.equal(rate.fps(),null);
  rate.add(100,10);rate.add(200,13);
  assert.equal(rate.fps(),30);
  rate.add(300,1);
  assert.equal(rate.fps(),null);
});
