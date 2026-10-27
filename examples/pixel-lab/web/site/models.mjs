// The segmenters the page offers, fetched from Hugging Face at pinned revisions.
// `mean` and `std` map RGB in [0, 1] to what each model was trained on.
export const MODELS = [
  {
    id: 'mediapipe',
    name: 'MediaPipe Selfie Segmentation',
    note: 'fast person segmentation, 256 × 256 mask',
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
    id: 'modnet',
    name: 'MODNet portrait matting',
    note: 'alpha matting with finer edges, 512 × 512 matte',
    repo: 'Xenova/modnet',
    revision: 'fa2fa546052fba4c08921230a26cc69a333fca12',
    file: 'onnx/model.onnx',
    bytes: 25888640,
    width: 512,
    height: 512,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
];

export const CAPTURES = [
  {id: '640x360', width: 640, height: 360},
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
