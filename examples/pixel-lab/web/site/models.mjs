// The segmenters the page offers. MediaPipe and the fp32 RVM come from Hugging Face at a pinned
// revision. prepare-models.py writes the webcam MODNet, the PP-HumanSegV2 person channel, the five
// Longpipe landscape mattes and the fp16 RVM into dist/models. Longpipe keeps an f32 camera and an
// f32 mask; `f16Body` runs the network in f16. The fp16 RVM file is the official graph with two
// Casts, so the camera and the matte are float32 and the recurrent state stays float16. The
// PP-HumanSegV2 slice is not published upstream, so it is fetched from tract-test-assets.
// `mean` and `std` map RGB in [0, 1] to what each model was trained on.
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
    id: 'pp-humanseg-v2',
    name: 'PP-HumanSegV2 256 × 144',
    note: 'portrait segmentation, steadier when the person is still',
    url: 'models/pp_humanseg_v2.onnx',
    bytes: 3831310,
    sha256: '6734ac175ca3c3d94bf1dfb7e253573c84a602b8b49d27499d256b73faa476f7',
    source: 'https://github.com/PaddlePaddle/PaddleSeg',
    width: 256,
    height: 144,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
  {
    id: 'modnet',
    name: 'MODNet 512 × 288',
    note: 'webcam matte, finer edges',
    url: 'models/modnet_webcam.onnx',
    bytes: 25969398,
    sha256: 'de03cc16f3c91f25b7c2f0b42ea1a8d34f40a752234f3887572655e744e55306',
    source: 'https://github.com/yakhyo/modnet/releases/download/weights/modnet_webcam.onnx',
    width: 512,
    height: 288,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  },
  {
    id: 'lp-xs',
    name: 'Longpipe xs 384 × 240',
    note: 'static matte, fixed canvas',
    f16Body: true,
    url: 'models/lp_xs.onnx',
    bytes: 3613647,
    sha256: '75749759daa4d7e839e8aa189ce6a4e95edc72253b1f5028e833455ebae2f792',
    source: 'https://huggingface.co/Longpipe/longpipe/blob/main/onnx/xs/matting_landscape.onnx',
    width: 384,
    height: 240,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'lp-small',
    name: 'Longpipe small 384 × 224',
    note: 'static matte, fixed canvas',
    f16Body: true,
    url: 'models/lp_small.onnx',
    bytes: 3613647,
    sha256: 'e6aca89d9df1cfe26670693eaf138d615b3b74979fe613f8f0195e9f5da14a90',
    source: 'https://huggingface.co/Longpipe/longpipe/blob/main/onnx/small/matting_landscape.onnx',
    width: 384,
    height: 224,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'lp-medium',
    name: 'Longpipe medium 512 × 320',
    note: 'static matte, fixed canvas',
    f16Body: true,
    url: 'models/lp_medium.onnx',
    bytes: 12959999,
    sha256: 'a0b385d6690b2d08c1e393ff2258b6f3b3065d8bee0f1ff0f0ce2d68a0ce0cf7',
    source: 'https://huggingface.co/Longpipe/longpipe/blob/main/onnx/medium/matting_landscape.onnx',
    width: 512,
    height: 320,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'lp-large',
    name: 'Longpipe large 640 × 400',
    note: 'static matte, fixed canvas',
    f16Body: true,
    url: 'models/lp_large.onnx',
    bytes: 14209298,
    sha256: '2dab0b195cd712477a1549d38465e28dcca4ad05efe407c9406f08caa6c6e4da',
    source: 'https://huggingface.co/Longpipe/longpipe/blob/main/onnx/large/matting_landscape.onnx',
    width: 640,
    height: 400,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'lp-xl',
    name: 'Longpipe xl 1280 × 768',
    note: 'static matte, fixed canvas',
    f16Body: true,
    url: 'models/lp_xl.onnx',
    bytes: 29198204,
    sha256: '9ee02f5a73df90d5b62af667eaf57ca85a4065e7fe676e845d4a28d410f56daa',
    source: 'https://huggingface.co/Longpipe/longpipe/blob/main/onnx/xl/matting_landscape.onnx',
    width: 1280,
    height: 768,
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'rvm',
    kind: 'rvm',
    name: 'RVM MobileNetV3 fp16',
    note: 'recurrent matting, scale follows the camera',
    url: 'models/rvm_mobilenetv3_fp16.onnx',
    bytes: 7503609,
    sha256: '84c01698359a946d2167152f378d27e59eaebc86666190ffd9b5642aba7e4e98',
    source: 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx',
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  {
    id: 'rvm-fp32',
    kind: 'rvm',
    name: 'RVM MobileNetV3 fp32',
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

// MediaPipe and MODNet classify each frame on its own. RVM already carries a
// recurrent state, so the uncertain-mask hold is not applied to it.
export function holdsUncertainMask(entry) {
  return entry.kind !== 'rvm';
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
