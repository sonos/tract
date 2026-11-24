// Whether this browser gives tract the matrix units of the GPU: 32-lane subgroups and an 8x8
// product that keeps its scalar type.
const MATRIX = 'chromium-experimental-subgroup-matrix';

export function cooperativeSupport(adapter) {
  const {features, info = {}} = adapter ?? {};
  if (!features?.has('subgroups') || !features.has(MATRIX)) return false;
  if (info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32) return false;
  return [...(info.subgroupMatrixConfigs ?? [])].some(config =>
    config.M === 8 && config.N === 8 && config.K === 8 &&
    config.componentType === config.resultComponentType &&
    (config.componentType === 'f32' || config.componentType === 'f16'));
}

// Whether tract's device got the matrices it asked for.
export function usesCooperative(device) {
  return device.features.has(MATRIX);
}
