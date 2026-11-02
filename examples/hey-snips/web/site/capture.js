// Hands the microphone's samples, at the audio context's rate, to the detector's worker through
// the port the page passes in, in blocks of about 20 ms.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = Math.round(sampleRate / 50);
    this.block = new Float32Array(this.size);
    this.filled = 0;
    this.worker = null;
    this.port.onmessage = ({data}) => (this.worker = data.port);
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || !this.worker) return true;
    let offset = 0;
    while (offset < channel.length) {
      const taken = Math.min(channel.length - offset, this.size - this.filled);
      this.block.set(channel.subarray(offset, offset + taken), this.filled);
      this.filled += taken;
      offset += taken;
      if (this.filled === this.size) {
        this.worker.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(this.size);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture', Capture);
