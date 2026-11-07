// Which stored camera frame a new upload may replace. `keep` holds frame ids that
// must stay: the frame on screen with the current mask, and the frame inference is
// running on. An id already stored is written again in place.
export function ringSlot(frameIds, frameId, keep) {
  const existing = frameIds.indexOf(frameId);
  if (existing >= 0) return existing;
  let best = -1;
  for (let i = 0; i < frameIds.length; i++) {
    if (keep.has(frameIds[i])) continue;
    if (best < 0 || frameIds[i] < frameIds[best]) best = i;
  }
  return best;
}

// Warp a camera frame only to fill the gap between the mask rate and the camera
// clock. The frame inference is running on waits for its own mask. A newer frame
// has no mask coming, so the last mask is moved onto it.
export function shouldWarp(motion, frameId, inferredFrame, maskFrameId) {
  return Boolean(motion) && maskFrameId >= 0 && frameId !== maskFrameId && frameId !== inferredFrame;
}

// The picture actually shown. A fill frame shows the newest camera frame. Otherwise the
// picture is the mask's frame, or the last one of those still stored.
export function shownPicture({warped, frameId, maskFrameId, matched, heldFrameId}) {
  if (warped) return {live: true, id: frameId};
  if (matched) return {live: false, id: maskFrameId};
  if (heldFrameId >= 0) return {live: false, id: heldFrameId};
  return {live: true, id: frameId};
}

// Same key means the output picture did not change. The id is the camera frame on
// screen. A warp of that frame and the true mask landing on it are one picture,
// so a fast model does not count the frame twice.
export function presentationKey(picture) {
  return String(picture.id);
}
