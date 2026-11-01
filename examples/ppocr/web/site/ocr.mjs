// Detection, cropping and CTC decode for PP-OCRv6. No runtime imports: the page
// and the node check both call this with a reader that has `run`.

const DET_MEAN = [0.485, 0.456, 0.406];
const DET_STD = [0.229, 0.224, 0.225];

export function alphabetFromYaml(yaml) {
  const lines = yaml.split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim() === 'character_dict:');
  if (start < 0) throw new Error('Recognition config has no character dictionary');
  const chars = [];
  for (const line of lines.slice(start + 1)) {
    if (!line.startsWith('  - ')) break;
    let raw = line.slice(4).trim();
    if (raw.startsWith("'") && raw.endsWith("'")) raw = raw.slice(1, -1).replaceAll("''", "'");
    else if (raw.startsWith('"') && raw.endsWith('"')) raw = raw.slice(1, -1);
    chars.push(raw);
  }
  // Index 0 is the CTC blank. The export appends a space (use_space_char).
  return ['', ...chars, ' '];
}

export function detGeometry(width, height, limit = 960) {
  const ratio = Math.max(width, height) > limit ? limit / Math.max(width, height) : 1;
  const resizeW = Math.max(32, Math.round((width * ratio) / 32) * 32);
  const resizeH = Math.max(32, Math.round((height * ratio) / 32) * 32);
  return { resizeW, resizeH, ratioW: resizeW / width, ratioH: resizeH / height };
}

function sampleRgb(rgb, width, height, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const dx = x - x0;
  const dy = y - y0;
  const at = (px, py, channel) => {
    const cx = Math.min(width - 1, Math.max(0, px));
    const cy = Math.min(height - 1, Math.max(0, py));
    return rgb[(cy * width + cx) * 3 + channel];
  };
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const top = at(x0, y0, c) * (1 - dx) + at(x1, y0, c) * dx;
    const bottom = at(x0, y1, c) * (1 - dx) + at(x1, y1, c) * dx;
    out[c] = top * (1 - dy) + bottom * dy;
  }
  return out;
}

export function resizeRgb(rgb, width, height, resizeW, resizeH) {
  const out = new Uint8Array(resizeW * resizeH * 3);
  for (let y = 0; y < resizeH; y++) {
    const fy = (y + 0.5) * (height / resizeH) - 0.5;
    for (let x = 0; x < resizeW; x++) {
      const fx = (x + 0.5) * (width / resizeW) - 0.5;
      const pixel = sampleRgb(rgb, width, height, fx, fy);
      const offset = (y * resizeW + x) * 3;
      out[offset] = pixel[0];
      out[offset + 1] = pixel[1];
      out[offset + 2] = pixel[2];
    }
  }
  return out;
}

function detTensor(rgb, width, height, geometry) {
  const resized = resizeRgb(rgb, width, height, geometry.resizeW, geometry.resizeH);
  const data = new Float32Array(3 * geometry.resizeH * geometry.resizeW);
  const plane = geometry.resizeH * geometry.resizeW;
  for (let i = 0; i < plane; i++) {
    // Model inputs are BGR, ImageNet-normalized.
    for (let c = 0; c < 3; c++) {
      const value = resized[i * 3 + (2 - c)] / 255;
      data[c * plane + i] = (value - DET_MEAN[c]) / DET_STD[c];
    }
  }
  return data;
}

function find(parent, index) {
  while (parent[index] !== index) {
    parent[index] = parent[parent[index]];
    index = parent[index];
  }
  return index;
}

function components(bitmap, width, height) {
  const labels = new Int32Array(width * height);
  const parent = [0];
  let next = 0;
  const labelAt = (x, y) => (x < 0 || y < 0 || x >= width ? 0 : labels[y * width + x]);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      if (!bitmap[index]) continue;
      const neighbors = [labelAt(x - 1, y), labelAt(x - 1, y - 1), labelAt(x, y - 1), labelAt(x + 1, y - 1)].filter(Boolean);
      if (!neighbors.length) {
        parent.push(++next);
        labels[index] = next;
        continue;
      }
      labels[index] = neighbors[0];
      const root = find(parent, neighbors[0]);
      for (const neighbor of neighbors.slice(1)) {
        const other = find(parent, neighbor);
        if (other !== root) parent[other] = root;
      }
    }
  }
  const groups = new Map();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      if (!labels[index]) continue;
      const edge =
        !bitmap[index - 1] ||
        !bitmap[index + 1] ||
        !bitmap[index - width] ||
        !bitmap[index + width] ||
        x === 0 ||
        y === 0 ||
        x === width - 1 ||
        y === height - 1;
      if (!edge) continue;
      const id = find(parent, labels[index]);
      let group = groups.get(id);
      if (!group) groups.set(id, (group = []));
      group.push([x, y]);
    }
  }
  return [...groups.values()];
}

function cross(origin, a, b) {
  return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);
}

function convexHull(points) {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique = [];
  for (const point of sorted) {
    const last = unique[unique.length - 1];
    if (!last || last[0] !== point[0] || last[1] !== point[1]) unique.push(point);
  }
  if (unique.length <= 2) return unique;
  const build = (source) => {
    const hull = [];
    for (const point of source) {
      while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], point) <= 0) hull.pop();
      hull.push(point);
    }
    return hull;
  };
  const lower = build(unique);
  const upper = build(unique.slice().reverse());
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function polygonArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const next = points[(i + 1) % points.length];
    sum += points[i][0] * next[1] - next[0] * points[i][1];
  }
  return sum / 2;
}

function perimeter(points) {
  let length = 0;
  for (let i = 0; i < points.length; i++) length += Math.hypot(points[(i + 1) % points.length][0] - points[i][0], points[(i + 1) % points.length][1] - points[i][1]);
  return length;
}

// Minimum-area rectangle expanded by the DB unclip distance. Corners are
// top-left, top-right, bottom-right, bottom-left in image space.
function unclippedQuad(points, unclipRatio) {
  const hull = convexHull(points);
  if (hull.length < 3) return null;
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const start = hull[i];
    const end = hull[(i + 1) % hull.length];
    const length = Math.hypot(end[0] - start[0], end[1] - start[1]) || 1;
    const ux = (end[0] - start[0]) / length;
    const uy = (end[1] - start[1]) / length;
    const vx = -uy;
    const vy = ux;
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const point of hull) {
      const u = point[0] * ux + point[1] * uy;
      const v = point[0] * vx + point[1] * vy;
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
    const area = (maxU - minU) * (maxV - minV);
    if (!best || area < best.area) best = { area, ux, uy, vx, vy, minU, maxU, minV, maxV };
  }
  const distance = (Math.abs(polygonArea(hull)) * unclipRatio) / Math.max(perimeter(hull), 1);
  best.minU -= distance;
  best.maxU += distance;
  best.minV -= distance;
  best.maxV += distance;
  const corner = (u, v) => [u * best.ux + v * best.vx, u * best.uy + v * best.vy];
  const quad = [
    corner(best.minU, best.minV),
    corner(best.maxU, best.minV),
    corner(best.maxU, best.maxV),
    corner(best.minU, best.maxV),
  ];
  return orderQuad(quad);
}

function orderQuad(points) {
  const sum = points.map(([x, y]) => x + y);
  const diff = points.map(([x, y]) => y - x);
  return [
    points[sum.indexOf(Math.min(...sum))],
    points[diff.indexOf(Math.min(...diff))],
    points[sum.indexOf(Math.max(...sum))],
    points[diff.indexOf(Math.max(...diff))],
  ];
}

function meanInside(pred, width, quad) {
  const xs = quad.map((point) => point[0]);
  const ys = quad.map((point) => point[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const x1 = Math.min(width - 1, Math.ceil(Math.max(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const y1 = Math.min(pred.length / width - 1, Math.ceil(Math.max(...ys)));
  let total = 0;
  let count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (!pointInQuad(x + 0.5, y + 0.5, quad)) continue;
      total += pred[y * width + x];
      count++;
    }
  }
  return count ? total / count : 0;
}

function pointInQuad(x, y, quad) {
  let inside = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const [xi, yi] = quad[i];
    const [xj, yj] = quad[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi) inside = !inside;
  }
  return inside;
}

export function boxesFromMap(pred, mapWidth, mapHeight, ratioW, ratioH, options) {
  const bitmap = new Uint8Array(mapWidth * mapHeight);
  for (let i = 0; i < pred.length; i++) bitmap[i] = pred[i] > options.thresh ? 1 : 0;
  const found = [];
  for (const points of components(bitmap, mapWidth, mapHeight)) {
    if (points.length < 4) continue;
    // Score the tight box. The unclip expansion is only for the crop, and
    // scoring that larger box drops thin print on a photograph.
    const tight = unclippedQuad(points, 0);
    if (!tight) continue;
    const score = meanInside(pred, mapWidth, tight);
    if (score < options.boxThresh) continue;
    const quad = unclippedQuad(points, options.unclip);
    if (!quad) continue;
    const width = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]);
    const height = Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]);
    if (Math.min(width, height) < 3) continue;
    found.push({
      score,
      quad: quad.map(([x, y]) => [x / ratioW, y / ratioH]),
    });
  }
  found.sort((a, b) => b.score - a.score);
  return found.slice(0, options.maxBoxes ?? 300);
}

function rollVertical(quad) {
  const width = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]);
  const height = Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]);
  if (height > width * 1.5) return [quad[1], quad[2], quad[3], quad[0]];
  return quad;
}

function solve(rows) {
  const matrix = rows.map((row) => row.slice());
  const size = matrix.length;
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const scale = matrix[column][column];
    if (Math.abs(scale) < 1e-12) continue;
    for (let columnIndex = column; columnIndex <= size; columnIndex++) matrix[column][columnIndex] /= scale;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = matrix[row][column];
      for (let columnIndex = column; columnIndex <= size; columnIndex++) {
        matrix[row][columnIndex] -= factor * matrix[column][columnIndex];
      }
    }
  }
  return matrix.map((row) => row[size]);
}

function warpRgb(rgb, width, height, quad) {
  const upright = rollVertical(quad);
  const cropW = Math.max(2, Math.round(Math.hypot(upright[1][0] - upright[0][0], upright[1][1] - upright[0][1])));
  const cropH = Math.max(2, Math.round(Math.hypot(upright[3][0] - upright[0][0], upright[3][1] - upright[0][1])));
  const destination = [
    [0, 0],
    [cropW, 0],
    [cropW, cropH],
    [0, cropH],
  ];
  const equations = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = destination[i];
    const [u, v] = upright[i];
    equations.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    equations.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  const homography = solve(equations);
  const out = new Uint8Array(cropW * cropH * 3);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const denom = homography[6] * px + homography[7] * py + 1;
      const source = sampleRgb(rgb, width, height, (homography[0] * px + homography[1] * py + homography[2]) / denom, (homography[3] * px + homography[4] * py + homography[5]) / denom);
      const offset = (y * cropW + x) * 3;
      out[offset] = source[0];
      out[offset + 1] = source[1];
      out[offset + 2] = source[2];
    }
  }
  return { rgb: out, width: cropW, height: cropH };
}

function recTensor(crop, maxWidth) {
  const targetW = Math.min(maxWidth, Math.max(8, Math.round(crop.width * (48 / crop.height))));
  const resized = resizeRgb(crop.rgb, crop.width, crop.height, targetW, 48);
  const data = new Float32Array(3 * 48 * targetW);
  const plane = 48 * targetW;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) data[c * plane + i] = (resized[i * 3 + (2 - c)] / 255 - 0.5) / 0.5;
  }
  return { data, width: targetW };
}

function ctc(logits, time, classes, alphabet) {
  let text = '';
  let total = 0;
  let count = 0;
  let previous = -1;
  for (let step = 0; step < time; step++) {
    const row = step * classes;
    let best = 0;
    let max = -Infinity;
    for (let c = 0; c < classes; c++) {
      const value = logits[row + c];
      if (value > max) {
        max = value;
        best = c;
      }
    }
    if (best !== 0 && best !== previous && best < alphabet.length) {
      text += alphabet[best];
      let score = max;
      if (max > 1 || max < 0) {
        let normalizer = 0;
        for (let c = 0; c < classes; c++) normalizer += Math.exp(logits[row + c] - max);
        score = 1 / normalizer;
      }
      total += score;
      count++;
    }
    previous = best;
  }
  return { text, score: count ? total / count : 0 };
}

function bounds(quad) {
  const xs = quad.map((point) => point[0]);
  const ys = quad.map((point) => point[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  return { x0, x1, y0, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 };
}

// A two-column page has two clusters of line centers. Lines that cross the
// gutter, such as a heading or a detection blob that bridged the columns, stay
// in vertical order and are split before recognition. A sign or a photo has one
// cluster, so it is read top to bottom.
function columnCut(items) {
  if (items.length < 6) return null;
  const widths = items.map((item) => item.w).sort((a, b) => a - b);
  const median = widths[widths.length >> 1];
  const centers = items
    .filter((item) => item.w <= median * 1.2)
    .map((item) => item.cx)
    .sort((a, b) => a - b);
  if (centers.length < 6) return null;
  const low = centers[0];
  const high = centers.at(-1);
  const span = high - low;
  if (span < median) return null;
  const bins = 24;
  const hist = new Array(bins).fill(0);
  for (const center of centers) {
    const index = Math.min(bins - 1, Math.floor(((center - low) / span) * bins));
    hist[index]++;
  }
  let leftPeak = 0;
  let rightPeak = bins - 1;
  for (let i = 1; i < bins / 2; i++) if (hist[i] > hist[leftPeak]) leftPeak = i;
  for (let i = Math.ceil(bins / 2); i < bins - 1; i++) if (hist[i] >= hist[rightPeak]) rightPeak = i;
  if (rightPeak <= leftPeak + 1) return null;
  if (hist[leftPeak] < 3 || hist[rightPeak] < 3) return null;
  let valley = leftPeak + 1;
  for (let i = leftPeak + 1; i < rightPeak; i++) if (hist[i] < hist[valley]) valley = i;
  if (hist[valley] > Math.min(hist[leftPeak], hist[rightPeak]) * 0.35) return null;
  return { cut: low + ((valley + 0.5) / bins) * span, median };
}

export function arrange(items, pageWidth) {
  const layout = columnCut(items);
  const spanning = [];
  const columns = [[], []];
  for (const item of items) {
    const crosses = layout && item.w > layout.median * 1.15 && item.x0 < layout.cut && item.x1 > layout.cut;
    const full = item.w > pageWidth * 0.72;
    if (!layout || crosses || full) spanning.push(item);
    else columns[item.cx < layout.cut ? 0 : 1].push(item);
  }
  if (!layout) columns[0] = spanning.splice(0, spanning.length);
  for (const column of columns) column.sort((a, b) => a.cy - b.cy || a.x0 - b.x0);
  spanning.sort((a, b) => a.cy - b.cy);
  const lines = [];
  const takeAbove = (y) => {
    for (const column of columns) {
      const taken = [];
      while (column.length && column[0].cy < y) taken.push(column.shift());
      if (taken.length) lines.push(...joinLine(taken));
    }
  };
  for (const item of spanning) {
    takeAbove(item.cy);
    lines.push(item);
  }
  takeAbove(Infinity);
  return lines.map((line) => ({
    text: line.text.trim(),
    score: line.score,
    quads: line.quads ?? [line.quad],
  }));
}

function joinLine(items) {
  const lines = [];
  for (const item of items) {
    const line = lines.at(-1);
    const gap = line ? Math.max(8, 0.7 * Math.max(line.h, item.h)) : 0;
    if (line && Math.abs(item.cy - line.cy) < 0.55 * Math.max(line.h, item.h) && item.x0 <= line.x1 + gap) {
      line.parts.push(item);
      line.x1 = Math.max(line.x1, item.x1);
      line.cy = (line.cy + item.cy) / 2;
      line.h = Math.max(line.h, item.h);
    } else {
      lines.push({ ...item, parts: [item] });
    }
  }
  return lines.map((line) => {
    line.parts.sort((a, b) => a.x0 - b.x0);
    return {
      text: line.parts.map((part) => part.text.trim()).filter(Boolean).join(' '),
      score: line.parts.reduce((sum, part) => sum + part.score, 0) / line.parts.length,
      quads: line.parts.map((part) => part.quad),
    };
  });
}

export async function recognize(image, sessions, alphabet, model, hooks = {}) {
  const geometry = detGeometry(image.width, image.height);
  const tensor = new sessions.Tensor('float32', detTensor(image.rgb, image.width, image.height, geometry), [1, 3, geometry.resizeH, geometry.resizeW]);
  const detected = await sessions.det.run({ [sessions.det.inputNames[0]]: tensor });
  if (hooks.cancelled?.()) return null;
  const map = detected[sessions.det.outputNames[0]];
  const mapHeight = map.dims[2];
  const mapWidth = map.dims[3];
  const boxes = boxesFromMap(map.data, mapWidth, mapHeight, geometry.ratioW, geometry.ratioH, {
    thresh: 0.2,
    boxThresh: model.boxThresh,
    unclip: 1.4,
  });
  const kept = [];
  for (let index = 0; index < boxes.length; index++) {
    if (hooks.cancelled?.()) return null;
    const crop = warpRgb(image.rgb, image.width, image.height, boxes[index].quad);
    const encoded = recTensor(crop, 1280);
    const input = new sessions.Tensor('float32', encoded.data, [1, 3, 48, encoded.width]);
    const recognized = await sessions.rec.run({ [sessions.rec.inputNames[0]]: input });
    const output = recognized[sessions.rec.outputNames[0]];
    const decoded = ctc(output.data, output.dims[1], output.dims[2], alphabet);
    hooks.onLine?.(index + 1, boxes.length);
    if (decoded.score < 0.55 || !decoded.text.trim()) continue;
    kept.push({ ...decoded, ...bounds(boxes[index].quad), quad: boxes[index].quad });
  }
  return arrange(kept, image.width).filter((line) => line.text);
}
