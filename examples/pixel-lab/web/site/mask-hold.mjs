// Steady a single-frame mask. A new score is mixed toward the previous blended
// score by how close it sits to one half. A sure 0 or 1 replaces the history.
// A score near one half keeps most of it. The picture is harder than that mix:
// a low score is background and a high score is solid person. The next mask
// blends against the mix, before that step.
export const BLEND_KEEP = 0.9;
export const SHOW_BACKGROUND = 0.1;
export const SHOW_PERSON = 0.7;

// 1 at one half, 0 at 0 and at 1. Squared so the mix falls off before either end.
export function maskUncertainty(value) {
  const centered = Math.min(Math.abs(value - 0.5) * 2, 1);
  return 1 - centered * centered;
}

export function blendConfidence(previous, current) {
  const keep = maskUncertainty(current) * BLEND_KEEP;
  return current + (previous - current) * keep;
}

export function showMask(blended) {
  if (blended >= SHOW_PERSON) return 1;
  if (blended <= SHOW_BACKGROUND) return 0;
  return blended;
}

// `previous` is the stored mix. Returns the mix to store and the value to draw.
export function holdMask(previous, current) {
  const blended = blendConfidence(previous, current);
  return {blended, shown: showMask(blended)};
}
