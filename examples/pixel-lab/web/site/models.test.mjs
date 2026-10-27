import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MODELS, CAPTURES, model, capture, modelUrl} from './models.mjs';

test('every model is pinned and fits a 32-pixel grid', () => {
  assert.equal(new Set(MODELS.map(m => m.id)).size, MODELS.length);
  for (const m of MODELS) {
    assert.match(m.revision, /^[0-9a-f]{40}$/);
    assert.equal(m.width % 32, 0);
    assert.equal(m.height % 32, 0);
    assert.equal(m.mean.length, 3);
    assert.ok(m.std.every(s => s > 0));
  }
  assert.equal(modelUrl(model('modnet')), 'https://huggingface.co/Xenova/modnet/resolve/fa2fa546052fba4c08921230a26cc69a333fca12/onnx/model.onnx');
  assert.throws(() => model('nope'), /Unknown/);
});

test('capture sizes are independent of the model size', () => {
  assert.deepEqual(CAPTURES.map(c => [c.width, c.height]), [[640, 360], [1280, 720], [1920, 1080]]);
  assert.equal(capture('640x360').width, 640);
  assert.throws(() => capture('4000x3000'), /Unsupported/);
});
