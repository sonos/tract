// What the image box shows during a run: the weights arriving, the prompt being read, the
// model's own estimate of the image after each step, then the image sharpening into place.

const TILES = 26;
const COLORS = ["#2a2d2a", "#1D9E75", "#5DCAA5"];
const stillness = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const INK = "#101110";

function tilesPainter(ctx, size) {
  const seen = [];
  return {
    add(sample) {
      seen.push(sample % 3);
    },
    draw(done) {
      const cell = size / TILES;
      ctx.fillStyle = INK;
      ctx.fillRect(0, 0, size, size);
      const lit = Math.floor(done * TILES * TILES);
      for (let i = 0; i < TILES * TILES; i++) {
        const on = i <= lit;
        ctx.fillStyle = on ? COLORS[seen.length ? seen[i % seen.length] : i % 3] : "#1b1d1b";
        ctx.fillRect((i % TILES) * cell + 1, Math.floor(i / TILES) * cell + 1, cell - 2, cell - 2);
      }
    },
  };
}

function chips(ctx, size, words, lit) {
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, size, size);
  const scale = size / 340;
  ctx.font = `500 ${15 * scale}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  let x = 20 * scale;
  let y = 46 * scale;
  words.slice(0, 40).forEach((word, i) => {
    const w = ctx.measureText(word).width + 20 * scale;
    if (x + w > size - 16 * scale) {
      x = 20 * scale;
      y += 42 * scale;
    }
    const on = lit > i;
    ctx.fillStyle = on ? "#0F6E56" : "#232523";
    ctx.beginPath();
    ctx.roundRect(x, y - 15 * scale, w, 30 * scale, 8 * scale);
    ctx.fill();
    ctx.fillStyle = on ? "#E1F5EE" : "#7a7d7a";
    ctx.fillText(word, x + 10 * scale, y + scale);
    x += w + 8 * scale;
  });
}

function blocks(ctx, source, width, height, gap) {
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  if (gap) {
    const cell = width / source.width;
    ctx.fillStyle = INK;
    for (let x = cell; x < width; x += cell) ctx.fillRect(x - 0.5, 0, 1, height);
    for (let y = cell; y < height; y += cell) ctx.fillRect(0, y - 0.5, width, 1);
  }
}

function noise(width, height) {
  const c = new OffscreenCanvas(width, height);
  const image = new ImageData(width, height);
  crypto.getRandomValues(image.data);
  for (let i = 3; i < image.data.length; i += 4) image.data[i] = 255;
  c.getContext("2d").putImageData(image, 0, 0);
  return c;
}

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.active = false;
  }

  begin(width, height, prompt, encodeMs) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.width = width;
    this.height = height;
    this.grid = [width / 16, height / 16];
    this.words = prompt.trim().split(/\s+/);
    this.encodeMs = encodeMs ?? 3000;
    this.phase = null;
    this.previous = noise(...this.grid);
    this.current = this.previous;
    this.turned = 0;
    this.tiles = tilesPainter(this.ctx, Math.max(width, height));
    this.done = 0;
    this.active = true;
    this.still = stillness();
    this.frame();
  }

  enter(stage) {
    this.phase = stage;
    this.at = performance.now();
    this.done = 0;
  }

  bytes(fraction, sample) {
    this.done = fraction;
    this.tiles.add(sample);
  }

  // Step `step` estimated the image: tokens one pixel each.
  preview(rgba, width) {
    const c = new OffscreenCanvas(width, rgba.length / 4 / width);
    c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), width), 0, 0);
    this.previous = this.current;
    this.current = c;
    this.turned = performance.now();
  }

  frame() {
    if (!this.active) return;
    const now = performance.now();
    const { ctx, width, height } = this;
    if (this.phase === "encoding") {
      const lit = this.still ? this.words.length : ((now - this.at) / this.encodeMs) * this.words.length;
      chips(ctx, Math.max(width, height), this.words, lit);
    } else if (this.phase === "denoising" || this.phase === "loading image decoder" || this.phase === "decoding") {
      blocks(ctx, this.previous, width, height, true);
      const fade = this.still || !this.turned ? 1 : Math.min(1, (now - this.turned) / 500);
      ctx.globalAlpha = fade;
      blocks(ctx, this.current, width, height, true);
      ctx.globalAlpha = 1;
    } else if (this.phase) {
      this.tiles.draw(this.done);
    }
    requestAnimationFrame(() => this.frame());
  }

  // Sharpens from the token grid to `image`, then leaves it in place.
  async finish(image) {
    this.active = false;
    const { ctx, width, height } = this;
    if (!this.still) {
      const final = await createImageBitmap(image);
      for (let scale = 16; scale >= 2; scale /= 2) {
        const c = new OffscreenCanvas(Math.ceil(width / scale), Math.ceil(height / scale));
        const x = c.getContext("2d");
        x.imageSmoothingQuality = "high";
        x.drawImage(final, 0, 0, c.width, c.height);
        blocks(ctx, c, width, height, scale === 16);
        await new Promise((r) => setTimeout(r, 130));
      }
    }
    ctx.imageSmoothingEnabled = true;
    ctx.putImageData(image, 0, 0);
  }

  stop() {
    this.active = false;
  }
}
