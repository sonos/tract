import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MODELS, CAPTURES, model, capture, modelUrl, rvmRatio, rvmNetworkSize, rvmScale, formatR, frameShape, modelsFor, holdsUncertainMask, offersF16, runsF16} from './models.mjs';

test('every model is pinned and has a 16-pixel-aligned size', () => {
  assert.equal(new Set(MODELS.map(m => m.id)).size, MODELS.length);
  for (const m of MODELS) {
    assert.equal(m.mean.length, 3);
    assert.ok(m.std.every(s => s > 0));
    if (m.url) {
      assert.match(m.sha256, /^[0-9a-f]{64}$/);
      assert.ok(m.source);
    } else {
      assert.match(m.revision, /^[0-9a-f]{40}$/);
    }
    if (m.kind === 'rvm') continue;
    assert.equal(m.width % 16, 0);
    assert.equal(m.height % 16, 0);
  }
  assert.deepEqual(MODELS.filter(m => m.id.startsWith('mediapipe')).map(m => [m.id, m.width, m.height]), [
    ['mediapipe', 256, 256],
    ['mediapipe-landscape', 256, 144],
  ]);
  assert.deepEqual(MODELS.filter(m => m.id.startsWith('modnet')).map(m => [m.id, m.width, m.height]), [
    ['modnet', 512, 288],
  ]);
  assert.deepEqual(MODELS.filter(m => m.id.startsWith('lp-')).map(m => [m.id, m.width, m.height, m.f16Body, m.bytes, m.sha256]), [
    ['lp-xs', 384, 240, true, 3613647, '75749759daa4d7e839e8aa189ce6a4e95edc72253b1f5028e833455ebae2f792'],
    ['lp-small', 384, 224, true, 3613647, 'e6aca89d9df1cfe26670693eaf138d615b3b74979fe613f8f0195e9f5da14a90'],
    ['lp-medium', 512, 320, true, 12959999, 'a0b385d6690b2d08c1e393ff2258b6f3b3065d8bee0f1ff0f0ce2d68a0ce0cf7'],
    ['lp-large', 640, 400, true, 14209298, '2dab0b195cd712477a1549d38465e28dcca4ad05efe407c9406f08caa6c6e4da'],
    ['lp-xl', 1280, 768, true, 29198204, '9ee02f5a73df90d5b62af667eaf57ca85a4065e7fe676e845d4a28d410f56daa'],
  ]);
  assert.equal(MODELS.filter(m => m.f16Body === true).length, 5);
  assert.equal(modelUrl(model('modnet')), 'models/modnet_webcam.onnx');
  assert.equal(model('modnet').bytes, 25969398);
  assert.equal(model('modnet').sha256, 'de03cc16f3c91f25b7c2f0b42ea1a8d34f40a752234f3887572655e744e55306');
  assert.equal(model('pp-humanseg-v2').bytes, 3831310);
  assert.equal(model('pp-humanseg-v2').sha256, '6734ac175ca3c3d94bf1dfb7e253573c84a602b8b49d27499d256b73faa476f7');
  assert.equal(modelUrl(model('pp-humanseg-v2')), 'models/pp_humanseg_v2.onnx');
  assert.equal(model('rvm').name, 'RVM MobileNetV3 fp16');
  assert.equal(model('rvm').bytes, 7503609);
  assert.equal(model('rvm').sha256, '84c01698359a946d2167152f378d27e59eaebc86666190ffd9b5642aba7e4e98');
  assert.equal(model('rvm').source, 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx');
  assert.equal(modelUrl(model('rvm')), 'models/rvm_mobilenetv3_fp16.onnx');
  assert.equal(model('rvm-fp32').bytes, 14975696);
  assert.equal(model('rvm-fp32').sha256, '88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828');
  assert.equal(modelUrl(model('rvm-fp32')), 'https://huggingface.co/eafish/web-onnx/resolve/576473dca1b5792d8c30753b21325edc4063440b/rvm_mobilenetv3_fp32.onnx');
  assert.throws(() => model('nope'), /Unknown/);
});

test('the uncertain-mask hold is for the single-frame models', () => {
  for (const id of ['rvm', 'rvm-fp32']) assert.equal(holdsUncertainMask(model(id)), false);
  for (const id of ['mediapipe', 'mediapipe-landscape', 'pp-humanseg-v2', 'modnet', 'lp-xs', 'lp-small', 'lp-medium', 'lp-large', 'lp-xl']) {
    assert.equal(holdsUncertainMask(model(id)), true);
  }
});

test('RVM scale keeps a 512 long side and does not enlarge a smaller camera', () => {
  assert.equal(rvmRatio(320, 180), 1);
  assert.equal(rvmNetworkSize(320, 180).width, 320);
  assert.equal(rvmNetworkSize(640, 360).width, 512);
  assert.equal(rvmNetworkSize(640, 360).height, 288);
  assert.equal(rvmNetworkSize(960, 540).width, 512);
  assert.equal(rvmNetworkSize(1280, 720).width, 512);
  assert.equal(rvmNetworkSize(1920, 1080).width, 512);
  assert.equal(rvmNetworkSize(1920, 1080).height, 288);
  assert.ok(Math.abs(rvmRatio(1280, 720) - 0.4) < 1e-12);
  assert.throws(() => rvmRatio(0, 180), /camera/);
});

test('a fixed R factor replaces Auto, and the label shows that factor', () => {
  assert.equal(formatR(rvmRatio(320, 180)), '1.0');
  assert.equal(formatR(rvmRatio(640, 360)), '0.8');
  assert.equal(formatR(rvmRatio(1280, 720)), '0.4');
  assert.equal(formatR(rvmRatio(960, 540)), '0.533');
  assert.equal(formatR(rvmRatio(1920, 1080)), '0.267');
  assert.equal(formatR(0.25), '0.25');
  assert.equal(formatR(0.75), '0.75');
  assert.equal(rvmScale(1280, 720, 'auto'), rvmRatio(1280, 720));
  assert.equal(rvmScale(1280, 720), rvmRatio(1280, 720));
  assert.equal(rvmScale(1280, 720, '0.25'), 0.25);
  assert.equal(rvmScale(1920, 1080, '1'), 1);
  assert.deepEqual(rvmNetworkSize(1280, 720, 0.25), {ratio: 0.25, width: 320, height: 180});
  assert.deepEqual(rvmNetworkSize(1280, 720, 1), {ratio: 1, width: 1280, height: 720});
  assert.throws(() => rvmScale(1280, 720, '2'), /between 0 and 1/);
  assert.throws(() => rvmScale(1280, 720, '0'), /between 0 and 1/);
  assert.throws(() => formatR(0), /between 0 and 1/);
});

test('the model menu follows the capture aspect', () => {
  assert.equal(frameShape(1280, 720), 'landscape');
  assert.equal(frameShape(256, 256), 'square');
  assert.equal(frameShape(180, 320), 'portrait');
  assert.deepEqual(modelsFor(1280, 720).map(entry => entry.id), [
    'mediapipe-landscape', 'pp-humanseg-v2', 'modnet',
    'lp-xs', 'lp-small', 'lp-medium', 'lp-large', 'lp-xl',
    'rvm', 'rvm-fp32',
  ]);
  assert.deepEqual(modelsFor(640, 360).map(entry => entry.id), modelsFor(1920, 1080).map(entry => entry.id));
  assert.deepEqual(modelsFor(256, 256).map(entry => entry.id), ['mediapipe', 'rvm', 'rvm-fp32']);
  assert.deepEqual(modelsFor(180, 320).map(entry => entry.id), ['rvm', 'rvm-fp32']);
  assert.throws(() => frameShape(0, 180), /size/);
});

test('capture sizes are independent of the model size', () => {
  assert.deepEqual(CAPTURES.map(c => [c.width, c.height]), [[320, 180], [640, 360], [960, 540], [1280, 720], [1920, 1080]]);
  assert.equal(capture('640x360').width, 640);
  assert.throws(() => capture('4000x3000'), /Unsupported/);
});

test('f16 is an option for MediaPipe and MODNet and always on for Longpipe', () => {
  assert.deepEqual(MODELS.filter(offersF16).map(m => m.id), ['mediapipe', 'mediapipe-landscape', 'modnet']);
  assert.equal(runsF16(model('modnet'), false), false);
  assert.equal(runsF16(model('modnet'), true), true);
  assert.equal(runsF16(model('mediapipe'), undefined), false);
  assert.equal(runsF16(model('lp-xs'), false), true);
  assert.equal(runsF16(model('pp-humanseg-v2'), true), false);
  assert.equal(runsF16(model('rvm'), true), false);
});
