import {isNewSourceFrame} from './frame-rate.mjs';

// A camera frame that arrives while the previous one is still being sent is
// kept. The send that finishes starts that frame at once, so a fast mask is
// not held until a later camera callback.

export function emptyPump() {
  return {presented: undefined, sent: undefined, busy: false};
}

export function rememberFrame(pump, presented) {
  if (!isNewSourceFrame(pump.presented, presented)) return {pump, fresh: false};
  return {pump: {...pump, presented}, fresh: true};
}

// Start a send of the newest remembered frame. A send already running, or a
// frame that was already sent, waits.
export function startSend(pump) {
  if (pump.busy || !isNewSourceFrame(pump.sent, pump.presented)) return {pump, start: false};
  return {pump: {...pump, sent: pump.presented, busy: true}, start: true};
}

export function finishSend(pump) {
  return {...pump, busy: false};
}
