import urllib.request
import wave
from pathlib import Path

import numpy
import tract


def setup_module(module):
    for url, file in [
        ("https://tract-test-assets.tract.rs/hey-snips/v1/hey_snips.nnef.tar", "hey_snips.nnef.tar"),
        ("https://tract-test-assets.tract.rs/e2e/hey-snips-tts.wav", "hey-snips-tts.wav"),
    ]:
        if not Path(file).exists():
            urllib.request.urlretrieve(url, file)


def speech():
    with wave.open("hey-snips-tts.wav") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        data = wav.readframes(wav.getnframes())
    return numpy.frombuffer(data, dtype="<i2").astype(numpy.float32) / 32768.0


def scores(state, pulses):
    result = []
    for pulse in pulses:
        logits = state.run([pulse])[0].to_numpy().reshape(-1, 2)
        result.extend(logits[:, 1] - logits[:, 0])
    return numpy.array(result)


def test_hey_snips():
    model = tract.nnef().with_pulse().load("hey_snips.nnef.tar")
    pulse = model.input_fact(0).dim(0).to_int64()
    delay = model.property("pulse.delay").to_numpy()[0]
    state = model.into_runnable().spawn_state()
    assert state.input_count() == 1
    assert state.output_count() == 1
    audio = speech()
    pulses = [audio[start : start + pulse] for start in range(0, len(audio) - pulse + 1, pulse)]
    first, second = pulses[: len(pulses) // 2], pulses[len(pulses) // 2 :]
    head = scores(state, first)
    clone = state.clone()
    rest = scores(state, second)
    assert numpy.array_equal(scores(clone, second), rest)
    all_scores = numpy.concatenate([head, rest])[delay:]
    assert numpy.count_nonzero((all_scores[:-1] <= 2.0) & (all_scores[1:] > 2.0)) == 4
