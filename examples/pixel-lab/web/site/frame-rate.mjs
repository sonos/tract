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
    const first=this.samples[0], last=this.samples.at(-1);
    return (last.count-first.count)*1000/(last.time-first.time);
  }
}
