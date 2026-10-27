import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PipelineStats,FrameRate} from './pipeline-stats.mjs';

const sample = value => ({capture:value,inference:value*2,compositing:value*3});

test('exclude shader warmup and use a bounded rolling median and p95',()=>{
  const stats=new PipelineStats({warmup:2,capacity:4});
  assert.equal(stats.add(sample(1000),10000),false);
  assert.equal(stats.add(sample(1000),10000),false);
  for(const value of [1,2,3,4,20]) stats.add(sample(value),value*10);
  assert.equal(stats.samples.length,4);
  assert.deepEqual(stats.summary().capture,{median:3,p95:20});
  assert.deepEqual(stats.summary().inference,{median:6,p95:40});
  assert.deepEqual(stats.summary().total,{median:30,p95:200});
});

test('no invented values before warmup, reject non-finite or negative timings',()=>{
  const stats=new PipelineStats();
  assert.deepEqual(stats.summary().capture,{median:null,p95:null});
  for(const value of [-1,NaN,Infinity]) assert.throws(()=>stats.add(sample(value),1),/Invalid/);
  assert.throws(()=>stats.add(sample(1),NaN),/Invalid/);
  assert.equal(stats.frames,0);
});

test('pipeline FPS uses real completion cadence, not reciprocal inference time',()=>{
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
