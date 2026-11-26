// A small additive piano. A few inharmonic partials, a fast attack, an
// exponential decay, and a short hammer noise. Nothing is fetched.

const PARTIALS = 6;
const INHARMONICITY = 0.0004;

export class Piano {
  constructor() {
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.played = 0;
    this.voices = [];
    this.master = ctx.createGain();
    this.master.gain.value = 0.8;
    this.comp = ctx.createDynamicsCompressor();
    this.comp.threshold.value = -16;
    this.comp.knee.value = 16;
    this.comp.ratio.value = 3.5;
    this.comp.attack.value = 0.003;
    this.comp.release.value = 0.18;
    this.master.connect(this.comp);
    this.comp.connect(ctx.destination);
    this.noise = this.makeHammer();
    this.active = 0;
    this.resumes = 0;
    // The context can leave 'running' behind the page's back: an output
    // device change, a Bluetooth headset sleeping, Safari's 'interrupted'.
    // The page already had its user gesture, so resuming is allowed.
    ctx.onstatechange = () => {
      if (ctx.state !== 'running' && ctx.state !== 'closed' && this.wanted) this.kick();
    };
  }

  kick() {
    this.resumes++;
    this.ctx.resume().catch(() => { /* retried on the next note */ });
  }

  makeHammer() {
    const seconds = 0.05;
    const length = Math.floor(this.ctx.sampleRate * seconds);
    const buffer = this.ctx.createBuffer(1, length, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      const env = Math.exp(-i / (this.ctx.sampleRate * 0.008));
      data[i] = (Math.random() * 2 - 1) * env;
    }
    return buffer;
  }

  resume() {
    this.wanted = true;
    if (this.ctx.state === 'running') return Promise.resolve();
    return this.ctx.resume();
  }

  setVolume(value) {
    const now = this.ctx.currentTime;
    const gain = this.master.gain;
    const next = Math.max(0, Math.min(1, value));
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(gain.value, now);
    gain.linearRampToValueAtTime(next, now + 0.03);
  }

  play(midi, velocity) {
    const ctx = this.ctx;
    if (ctx.state !== 'running' && ctx.state !== 'closed') this.kick();
    const now = ctx.currentTime;
    const vel = Math.max(0.05, Math.min(1, velocity));
    const f0 = 440 * 2 ** ((midi - 69) / 12);
    if (!Number.isFinite(f0) || f0 < 20) return;
    // High notes die faster, the way a real string does.
    const dur = Math.min(2.6, Math.max(0.4, 1.85 * (440 / f0) ** 0.32));
    const amp = vel ** 1.45 * 0.2;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    const open = Math.min(ctx.sampleRate * 0.45, 500 + vel * 5200 + f0);
    filter.frequency.setValueAtTime(Math.max(200, open), now);
    filter.frequency.exponentialRampToValueAtTime(Math.max(160, f0 * 1.6), now + dur * 0.7);
    filter.Q.value = 0.5;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, amp), now + 0.007);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, amp * 0.42), now + 0.07);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + dur);

    const nyquist = ctx.sampleRate * 0.48;
    // Every node of this voice, so the whole voice can be disconnected when its
    // last source ends. Left connected, finished voices pile up on the master
    // bus until garbage collection gets to them, and the audio thread keeps
    // walking them; after a few minutes of playing the sound stops.
    const nodes = [filter, gain];
    const sources = [];
    for (let n = 1; n <= PARTIALS; n++) {
      const freq = f0 * n * Math.sqrt(1 + INHARMONICITY * n * n);
      if (freq >= nyquist) break;
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = freq;
      osc.detune.value = n % 2 === 0 ? 4 : -4;
      // One gain per partial: its level (a harder hit keeps more of the upper
      // partials) and its own decay.
      const level = vel ** 0.35 / n ** 1.2;
      const partial = ctx.createGain();
      const partialDur = Math.max(0.09, dur * (1.05 - n * 0.12));
      partial.gain.setValueAtTime(level, now);
      partial.gain.exponentialRampToValueAtTime(level * 0.0001, now + partialDur);
      osc.connect(partial);
      partial.connect(filter);
      osc.start(now);
      osc.stop(now + partialDur + 0.02);
      nodes.push(osc, partial);
      sources.push(osc);
    }

    const hammer = ctx.createBufferSource();
    hammer.buffer = this.noise;
    const hammerGain = ctx.createGain();
    hammerGain.gain.value = amp * 1.4;
    const highpass = ctx.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = 1400;
    hammer.connect(highpass);
    highpass.connect(hammerGain);
    hammerGain.connect(filter);
    hammer.start(now);
    hammer.stop(now + 0.05);
    nodes.push(hammer, highpass, hammerGain);
    sources.push(hammer);

    filter.connect(gain);
    gain.connect(this.master);
    const voice = {stopAt: now + dur, gain, nodes, live: sources.length};
    const ended = () => {
      if (--voice.live > 0) return;
      this.release(voice);
    };
    for (const source of sources) source.onended = ended;
    this.voices.push(voice);
    this.active++;
    // Voice stealing: past 24 sounding voices, the oldest is cut short. Its
    // sources still end on their own schedule and then disconnect it.
    while (this.voices.length > 24) {
      const old = this.voices.shift();
      try { old.gain.gain.cancelScheduledValues(now); old.gain.gain.setTargetAtTime(0, now, 0.004); } catch { /* already released */ }
    }
    this.played++;
  }

  release(voice) {
    if (voice.released) return;
    voice.released = true;
    for (const node of voice.nodes) {
      try { node.disconnect(); } catch { /* already disconnected */ }
    }
    voice.nodes = null;
    const at = this.voices.indexOf(voice);
    if (at >= 0) this.voices.splice(at, 1);
    this.active--;
  }

  // Context state and the number of voices still wired to the master bus.
  stats() {
    return {state: this.ctx.state, active: this.active, played: this.played, resumes: this.resumes};
  }
}
