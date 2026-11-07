// The segmenters the page offers. MediaPipe and MODNet come from Hugging Face at pinned
// revisions. RVM is the official Robust Video Matting v1.0.0 MobileNetV3 fp32 ONNX.
// That GitHub release does not send the cross-origin header a browser needs, so the page
// fetches a pinned Hugging Face copy of the same bytes.
// `mean` and `std` map RGB in [0, 1] to what each model was trained on. MODNet takes any size
// divisible by 32 and is given the camera's 16:9 shape: a square input stretches the face, and
// its matte breaks up on close-ups. The three MODNet entries are that one model at three sizes.
//
// RVM's downsample ratio follows the camera, using the project's rule of one: the long side of
// the frame its main network sees is 512, and a camera already smaller than that is not shrunk.
// That is `min(512 / max(width, height), 1)`. Auto uses this. A fixed factor in (0, 1] replaces it.
export const MODELS = [
  {
    id: 'mediapipe',
    name: 'MediaPipe 256 × 256',
    note: 'fast person segmentation, square model',
    repo: 'onnx-community/mediapipe_selfie_segmentation',
    revision: 'be49485c8e027524be38591817fc5cd31bd9d00e',
    file: 'onnx/model.onnx',
    bytes: 462352,
    width: 256,
    height: 256,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'mediapipe-landscape',
    name: 'MediaPipe 256 × 144',
    note: 'fast person segmentation, landscape model',
    repo: 'onnx-community/mediapipe_selfie_segmentation_landscape',
    revision: '2497d5bec26c626c7b3c4edc6e1fefc21b64f6c3',
    file: 'onnx/model.onnx',
    bytes: 462338,
    width: 256,
    height: 144,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'modnet',
    name: 'MODNet 512 × 288',
    note: 'alpha matting with finer edges',
    repo: 'Xenova/modnet',
    revision: 'fa2fa546052fba4c08921230a26cc69a333fca12',
    file: 'onnx/model.onnx',
    bytes: 25888640,
    width: 512,
    height: 288,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
  {
    id: 'modnet-fast',
    name: 'MODNet 448 × 256',
    note: 'alpha matting, faster',
    repo: 'Xenova/modnet',
    revision: 'fa2fa546052fba4c08921230a26cc69a333fca12',
    file: 'onnx/model.onnx',
    bytes: 25888640,
    width: 448,
    height: 256,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
  {
    id: 'modnet-384',
    name: 'MODNet 384 × 224',
    note: 'alpha matting, fastest',
    repo: 'Xenova/modnet',
    revision: 'fa2fa546052fba4c08921230a26cc69a333fca12',
    file: 'onnx/model.onnx',
    bytes: 25888640,
    width: 384,
    height: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
  {
    id: 'rvm',
    kind: 'rvm',
    name: 'RVM MobileNetV3',
    note: 'recurrent matting, scale follows the camera',
    repo: 'eafish/web-onnx',
    revision: '576473dca1b5792d8c30753b21325edc4063440b',
    file: 'rvm_mobilenetv3_fp32.onnx',
    bytes: 14975696,
    sha256: '88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828',
    source: 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx',
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
];

export const CAPTURES = [
  {id: '320x180', width: 320, height: 180},
  {id: '640x360', width: 640, height: 360},
  {id: '960x540', width: 960, height: 540},
  {id: '1280x720', width: 1280, height: 720},
  {id: '1920x1080', width: 1920, height: 1080},
];

export function model(id) {
  const found = MODELS.find(candidate => candidate.id === id);
  if (!found) throw new Error(`Unknown model ${id}`);
  return found;
}

export function capture(id) {
  const found = CAPTURES.find(candidate => candidate.id === id);
  if (!found) throw new Error(`Unsupported capture size ${id}`);
  return found;
}

// Square is close to 1. Anything wider is landscape, anything taller is portrait.
export function frameShape(width, height) {
  if (!(width > 0) || !(height > 0)) throw new Error('A frame needs a size');
  const ratio = width / height;
  if (ratio > 0.9 && ratio < 1.1) return 'square';
  return width >= height ? 'landscape' : 'portrait';
}

// RVM reads the camera frame, so it fits every shape. The others are shown only when their
// input shape matches the capture, so a 16:9 camera is not offered a square model.
export function modelsFor(width, height) {
  const shape = frameShape(width, height);
  return MODELS.filter(entry => {
    if (entry.kind === 'rvm') return true;
    return frameShape(entry.width, entry.height) === shape;
  });
}

// Official RVM rule: keep the downsampled frame's long side at 512. Below that, use the frame as it is.
export function rvmRatio(width, height) {
  if (!(width > 0) || !(height > 0)) throw new Error('RVM needs a camera size');
  return Math.min(512 / Math.max(width, height), 1);
}

// `choice` is "auto" or a fixed factor in (0, 1]. Auto is the official long-side rule.
export function rvmScale(width, height, choice = 'auto') {
  if (choice === 'auto' || choice == null || choice === '') return rvmRatio(width, height);
  const ratio = Number(choice);
  if (!(ratio > 0) || ratio > 1) throw new Error('RVM scale must be between 0 and 1');
  return ratio;
}

// One decimal when that is the value, so 0.4 stays "0.4" and 1 stays "1.0". Anything else keeps three decimals.
export function formatR(ratio) {
  if (!(ratio > 0) || ratio > 1) throw new Error('RVM scale must be between 0 and 1');
  const one = Number(ratio.toFixed(1));
  if (Math.abs(ratio - one) < 1e-6) return one.toFixed(1);
  return ratio.toFixed(3).replace(/0+$/, '');
}

// The size RVM's main network sees. `ratio` defaults to Auto for this camera size.
export function rvmNetworkSize(width, height, ratio = rvmRatio(width, height)) {
  if (!(ratio > 0) || ratio > 1) throw new Error('RVM scale must be between 0 and 1');
  return {ratio, width: Math.round(width * ratio), height: Math.round(height * ratio)};
}

export const modelUrl = m => m.url ?? `https://huggingface.co/${m.repo}/resolve/${m.revision}/${m.file}`;
