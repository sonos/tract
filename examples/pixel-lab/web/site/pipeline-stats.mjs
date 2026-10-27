export const STAGES = ['capture','inference','compositing'];

export class PipelineStats {
  constructor({warmup = 8, capacity = 120} = {}) {
    this.warmup = warmup;
    this.capacity = capacity;
    this.frames = 0;
    this.samples = [];
  }

  add(stages, elapsed) {
    if (!STAGES.every(stage=>Number.isFinite(stages[stage]) && stages[stage]>=0) || !Number.isFinite(elapsed) || elapsed<0) {
      throw new Error('Invalid pipeline timing sample');
    }
    this.frames++;
    if (this.frames <= this.warmup) return false;
    this.samples.push({...stages,total:elapsed});
    if (this.samples.length > this.capacity) this.samples.shift();
    return true;
  }

  summary() {
    const percentile = (key, fraction) => {
      const sorted = this.samples.map(sample=>sample[key]).sort((a,b)=>a-b);
      return sorted.length ? sorted[Math.ceil(fraction*sorted.length)-1] : null;
    };
    return Object.fromEntries([...STAGES,'total'].map(stage=>[stage,{median:percentile(stage,.5),p95:percentile(stage,.95)}]));
  }
}

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
