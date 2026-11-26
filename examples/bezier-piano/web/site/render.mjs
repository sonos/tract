// Canvas 2D strings. Layered strokes stand in for a glow. The video sits
// under the canvas, so the canvas itself stays transparent while the camera runs.

import {FINGERTIPS, cubicAt} from './strings.mjs';

export const STRING_COLORS = ['#6c9cf0', '#3ddc97', '#f0c36c', '#e08a52', '#c58af0', '#7fd0e8', '#f08ab0', '#d4e07a'];

export function sparkline(values, width = 120, height = 36) {
  if (!values.length) return '';
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const span = max - min || 1;
  return values.map((value, i) => {
    const x = values.length === 1 ? 0 : (i / (values.length - 1)) * width;
    const y = height - ((value - min) / span) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function vibrate(p, energy, time) {
  if (energy < 0.02) return p;
  const a = p[0];
  const b = p[3];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const amp = Math.sin(time / 42) * energy * 0.028;
  return [
    a,
    {x: p[1].x + nx * amp, y: p[1].y + ny * amp},
    {x: p[2].x + nx * amp * 0.55, y: p[2].y + ny * amp * 0.55},
    b,
  ];
}

function toPx(points, w, h) {
  return points.map(p => ({x: p.x * w, y: p.y * h}));
}

function strokeBezier(ctx, p, color, width, alpha) {
  ctx.beginPath();
  ctx.moveTo(p[0].x, p[0].y);
  ctx.bezierCurveTo(p[1].x, p[1].y, p[2].x, p[2].y, p[3].x, p[3].y);
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.stroke();
}

export function drawFrame(ctx, state, now) {
  const canvas = ctx.canvas;
  const w = canvas.width;
  const h = canvas.height;
  const dt = Math.min(50, now - (state.lastRender || now));
  state.lastRender = now;
  const decay = Math.exp(-dt / 220);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.running) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#101010';
    ctx.fillRect(0, 0, w, h);
  }
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const unit = w / 1280;
  ctx.globalCompositeOperation = 'lighter';
  for (const string of state.strings) {
    string.energy *= decay;
    string.ripples = string.ripples.filter(ripple => now - ripple.born < 460);
    const color = STRING_COLORS[string.color % STRING_COLORS.length];
    const shaped = toPx(vibrate(string.p, string.energy, now), w, h);
    const energy = string.energy;
    strokeBezier(ctx, shaped, color, (20 + energy * 18) * unit, 0.07 + energy * 0.08);
    strokeBezier(ctx, shaped, color, (7 + energy * 7) * unit, 0.16 + energy * 0.12);
    strokeBezier(ctx, shaped, color, 2.2 * unit, 0.9);
    for (const ripple of string.ripples) {
      const age = (now - ripple.born) / 460;
      const radius = 0.012 + age * 0.14;
      const from = Math.max(0, ripple.t - radius);
      const to = Math.min(1, ripple.t + radius);
      const count = 10;
      ctx.beginPath();
      for (let i = 0; i <= count; i++) {
        const point = cubicAt(vibrate(string.p, energy, now), from + (to - from) * (i / count));
        const x = point.x * w;
        const y = point.y * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#ffffff';
      ctx.globalAlpha = (1 - age) * 0.85;
      ctx.lineWidth = (10 - age * 6) * unit;
      ctx.stroke();
    }
  }
  for (const trail of state.trails) {
    const age = (now - trail.born) / 380;
    if (age >= 1) continue;
    ctx.beginPath();
    ctx.fillStyle = trail.color;
    ctx.globalAlpha = (1 - age) * 0.55;
    ctx.arc(trail.x * w, trail.y * h, (6 * (1 - age) + 1.5) * unit, 0, Math.PI * 2);
    ctx.fill();
  }
  state.trails = state.trails.filter(trail => now - trail.born < 380);

  ctx.globalCompositeOperation = 'source-over';
  if (state.draft && state.draft.length > 1) {
    ctx.beginPath();
    state.draft.forEach((point, i) => {
      const x = point.x * w;
      const y = point.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.setLineDash([7 * unit, 6 * unit]);
    ctx.strokeStyle = '#3ddc97';
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 2 * unit;
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (!state.drawMode) {
    for (let s = 0; s < state.strings.length; s++) {
      state.strings[s].p.forEach((point, i) => {
        const active = state.activeControl && state.activeControl.s === s && state.activeControl.i === i;
        ctx.beginPath();
        ctx.globalAlpha = active ? 1 : 0.75;
        ctx.fillStyle = active ? '#3ddc97' : '#6c9cf0';
        ctx.arc(point.x * w, point.y * h, (active ? 7 : 4) * unit, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
  if (state.showLandmarks) {
    for (const track of state.tracks) {
      const lm = track.landmarks;
      ctx.strokeStyle = 'rgba(108, 156, 240, 0.9)';
      ctx.lineWidth = 2 * unit;
      ctx.globalAlpha = 0.9;
      for (const conn of state.connections || []) {
        const a = lm[conn.start];
        const b = lm[conn.end];
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x * w, a.y * h);
        ctx.lineTo(b.x * w, b.y * h);
        ctx.stroke();
      }
      lm.forEach((point, index) => {
        const tip = FINGERTIPS.includes(index);
        ctx.beginPath();
        ctx.globalAlpha = 1;
        ctx.fillStyle = tip ? '#3ddc97' : '#6c9cf0';
        ctx.arc(point.x * w, point.y * h, (tip ? 5 : 3) * unit, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
  ctx.globalAlpha = 1;
}
