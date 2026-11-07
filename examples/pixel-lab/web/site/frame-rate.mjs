// A display can ask for a frame twice while the camera is still showing the same one.
// `presentedFrames` is the camera clock. The same value is not a new source frame.
export function isNewSourceFrame(previous, presented) {
  return !(Number.isFinite(presented) && presented === previous);
}

// How far back a displayed rate looks. A longer window keeps the old speed on
// screen for a long time after the pipeline slows down.
export const RATE_WINDOW_MS = 2000;

// Frames per second from a running frame counter sampled over time.
export class FrameRate {
  constructor(capacity = 120) {
    this.capacity = capacity;
    this.samples = [];
  }

  add(time, count) {
    if(!Number.isFinite(time) || !Number.isFinite(count) || count < 0) return;
    const previous = this.samples.at(-1);
    if(previous && (time < previous.time || count < previous.count)) this.samples=[];
    if(previous && time === previous.time) return;
    this.samples.push({time,count});
    if(this.samples.length > this.capacity) this.samples.shift();
  }

  fps() {
    if(this.samples.length < 2) return null;
    const end = this.samples.at(-1);
    const recent = this.samples.filter(sample => end.time - sample.time <= RATE_WINDOW_MS);
    const first = recent.length >= 2 ? recent[0] : this.samples.at(-2);
    const span = end.time - first.time;
    if(span <= 0) return null;
    return (end.count - first.count) * 1000 / span;
  }
}
