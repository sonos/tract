// The segmenters the page offers, fetched from Hugging Face at pinned revisions.
// `mean` and `std` map RGB in [0, 1] to what each model was trained on. MODNet takes any size
// divisible by 32 and is given the camera's 16:9 shape: a square input stretches the face, and
// its matte breaks up on close-ups.
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

export const modelUrl = m => `https://huggingface.co/${m.repo}/resolve/${m.revision}/${m.file}`;
