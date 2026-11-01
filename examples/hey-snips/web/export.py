"""Puts a resampler and a log-Mel featurizer in front of the hey_snips v4 wavenet, as one NNEF graph
over 48 kHz samples in [-1, 1], as Web Audio gives them.

    python export.py <wavenet nnef dir> <out dir>

The resampler is a windowed-sinc low-pass, scaled to the int16 range the wavenet was trained on, as
a convolution of stride 3 down to 16 kHz. The featurizer: 25 ms Hamming windows every 10 ms,
centred in 512-point frames, power spectrum, 20 HTK mel bands over 0-8 kHz, natural log floored
at 1. The wavenet's own graph counts frames with the symbol S; here S counts input samples.
"""
import os
import re
import shutil
import struct
import sys

import numpy as np

RATE, FFT, WINDOW, HOP, BANDS = 16000, 512, 400, 160, 20
INPUT_RATE, TAPS, CUTOFF = 48000, 127, 7200
DECIMATION = INPUT_RATE // RATE
RESAMPLED = f'((S - {TAPS}) / {DECIMATION} + 1)'
FRAMES = f'(({RESAMPLED} - {FFT}) / {HOP} + 1)'


def mel_filters():
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(RATE / 2), BANDS + 2))
    bins = np.fft.rfftfreq(FFT, 1 / RATE)
    filters = np.zeros((BANDS, len(bins)))
    for band, (low, centre, high) in enumerate(zip(edges, edges[1:], edges[2:])):
        filters[band] = np.clip(np.minimum((bins - low) / (centre - low), (high - bins) / (high - centre)), 0, None)
    return filters


def low_pass():
    t = np.arange(TAPS) - (TAPS - 1) / 2
    taps = 2 * CUTOFF / INPUT_RATE * np.sinc(2 * CUTOFF / INPUT_RATE * t) * np.blackman(TAPS)
    return 32768 * taps / taps.sum()


def write_dat(path, array):
    array = np.ascontiguousarray(array, np.float32)
    dims = list(array.shape) + [0] * (8 - array.ndim)
    header = struct.pack('<4BII8III', 0x4E, 0xEF, 1, 0, array.nbytes, array.ndim, *dims, 32, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.ljust(128, b'\0') + array.tobytes())


FEATURIZER = f'''  audio = tract_core_external(shape = [S], datum_type = "F32");
  resampler_filter = variable<scalar>(label = "resampler/filter", shape = [1, 1, {TAPS}]);
  resampler_input = unsqueeze(audio, axes = [0, 1]);
  resampler = conv(resampler_input, resampler_filter, 0.0, border = "constant", padding = [(0, 0)], stride = [{DECIMATION}], dilation = [1], groups = 1);
  resampled = squeeze(resampler, axes = [0, 1]);
  featurizer_window = variable<scalar>(label = "featurizer/window", shape = [{WINDOW}]);
  featurizer_mel = variable<scalar>(label = "featurizer/mel", shape = [{FFT // 2 + 1}, {BANDS}]);
  featurizer_complex = pad(unsqueeze(resampled, axes = [1]), padding = [(0, 0), (0, 1)], value = 0.0);
  featurizer_stft = tract_core_stft(featurizer_complex, axis = 0, frame = {FFT}, stride = {HOP}, window = featurizer_window);
  featurizer_half = slice(featurizer_stft, axes = [1], begin = [0], end = [{FFT // 2 + 1}], stride = [1]);
  featurizer_power = squeeze(sum_reduce(sqr(featurizer_half), axes = [2]), axes = [2]);
  featurizer_energy = matmul(featurizer_power, featurizer_mel, transposeA = false, transposeB = false);
  input_node = log(max(featurizer_energy, 1.0));
'''

src, dst = sys.argv[1:]
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
write_dat(f'{dst}/resampler/filter.dat', low_pass().reshape(1, 1, TAPS))
write_dat(f'{dst}/featurizer/window.dat', np.hamming(WINDOW))
write_dat(f'{dst}/featurizer/mel.dat', mel_filters().T)
graph = open(f'{dst}/graph.nnef').read()
head, wavenet = graph.split('  input_node = tract_core_external(shape = [S, 20], datum_type = "F32");\n')
head, found = re.subn(r'graph network\(input_node\)', 'graph network(audio)', head)
assert found == 1, 'Unexpected wavenet graph signature'
wavenet = re.sub(r'\bS\b', FRAMES, wavenet)
open(f'{dst}/graph.nnef', 'w').write(head + FEATURIZER + wavenet)
