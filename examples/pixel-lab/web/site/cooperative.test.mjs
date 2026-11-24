import {test} from 'node:test';
import assert from 'node:assert/strict';
import {cooperativeSupport, usesCooperative} from './cooperative.mjs';

const MATRIX = 'chromium-experimental-subgroup-matrix';
const square = (componentType, resultComponentType = componentType) => ({M: 8, N: 8, K: 8, componentType, resultComponentType});
const adapter = (features, configs, min = 32, max = 32) => ({
  features: new Set(features),
  info: {subgroupMinSize: min, subgroupMaxSize: max, subgroupMatrixConfigs: configs},
});

test('a flagged Chromium is supported', () => {
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [square('f32'), square('f16')])), true);
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [square('f16')])), true);
});

test('a browser without the matrices is not', () => {
  assert.equal(cooperativeSupport(null), false);
  assert.equal(cooperativeSupport({features: new Set()}), false);
  assert.equal(cooperativeSupport(adapter(['subgroups'], [square('f32')])), false);
  assert.equal(cooperativeSupport(adapter([MATRIX], [square('f32')])), false);
});

test('a matrix or subgroup the kernels cannot use is not', () => {
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [])), false);
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [square('f16', 'f32')])), false);
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [{...square('f32'), M: 16}])), false);
  assert.equal(cooperativeSupport(adapter(['subgroups', MATRIX], [square('f32')], 4, 64)), false);
});

test('the device tells whether it got them', () => {
  assert.equal(usesCooperative({features: new Set([MATRIX])}), true);
  assert.equal(usesCooperative({features: new Set(['subgroups'])}), false);
});
