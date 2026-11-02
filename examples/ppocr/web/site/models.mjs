// PP-OCRv6 tiers published by PaddlePaddle. Each recognition model carries its
// character dictionary in inference.yml at the same revision. Tiny's dictionary
// is the smaller one (6906 classes with the CTC blank and a trailing space);
// Small and Medium share the larger dictionary.

export const TIERS = [
  {
    id: 'tiny',
    name: 'Tiny',
    note: 'fastest, the one to start with',
    boxThresh: 0.4,
    det: {
      repo: 'PaddlePaddle/PP-OCRv6_tiny_det_onnx',
      revision: '2ba1506c0380b8f0b03dd142459aac66d4421f6c',
      bytes: 1780590,
    },
    rec: {
      repo: 'PaddlePaddle/PP-OCRv6_tiny_rec_onnx',
      revision: '2612ab37152ae0a677521bae4e1e3d4fb4cf7c30',
      bytes: 4462639,
    },
  },
  {
    id: 'small',
    name: 'Small',
    note: 'sharper on small print',
    boxThresh: 0.45,
    det: {
      repo: 'PaddlePaddle/PP-OCRv6_small_det_onnx',
      revision: '28fe5895c24fd108c19eb3e8479f4ab385fbfc62',
      bytes: 9880512,
    },
    rec: {
      repo: 'PaddlePaddle/PP-OCRv6_small_rec_onnx',
      revision: 'b8f84f0b80c529de40b4fbb3544b84fa7233a513',
      bytes: 21159378,
    },
  },
  {
    id: 'medium',
    name: 'Medium',
    note: 'most accurate',
    boxThresh: 0.45,
    det: {
      repo: 'PaddlePaddle/PP-OCRv6_medium_det_onnx',
      revision: '61323801669c338b7891481ec7bac61ce31b576a',
      bytes: 62032837,
    },
    rec: {
      repo: 'PaddlePaddle/PP-OCRv6_medium_rec_onnx',
      revision: '50c7eacafc52fa7bcf4194e8cd08e46f8558504b',
      bytes: 76554979,
    },
  },
];

export const tier = (id) => {
  const found = TIERS.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`Unknown model ${id}`);
  return found;
};

export const weightUrl = (model) =>
  `https://huggingface.co/${model.repo}/resolve/${model.revision}/inference.onnx`;

export const dictUrl = (model) =>
  `https://huggingface.co/${model.rec.repo}/resolve/${model.rec.revision}/inference.yml`;

export const weightBytes = (model) => model.det.bytes + model.rec.bytes;

export const megabytes = (bytes) => (bytes / 1e6).toFixed(bytes >= 10e6 ? 0 : 1);
