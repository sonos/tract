# NVIDIA NemotronLabs VoiceChat

[VoiceChat](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) is NVIDIA's
full-duplex voice assistant: it listens and speaks at the same time, one 80 ms frame at a time.
It is a chain of networks, all of which run in tract here:

- **The speech encoder**, a FastConformer from
  [nemotron-speech-streaming](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
  with VoiceChat's weights and no lookahead: 16 kHz audio to one 1024-wide frame per 80 ms.
  Tract pulses it, mel spectrogram included, so each frame only costs its own compute.
- **The language model**, Nemotron-H 9B (the hybrid Mamba-2 / attention model of
  [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)): each frame
  it reads the encoder frame and its own previous text token and writes the agent's next text
  token, greedily.
- **EAR-TTS**, a Gemma-3 speech generator: each frame it turns the agent's text token into 31
  codec codes, sampling a Mixture-of-Gaussians head with classifier-free guidance.
- **The codec decoder**: 31 codes to 1764 samples of voice at 22.05 kHz, pulsed too.
- **An RNNT head** on the encoder, which transcribes the user and drives NVIDIA's turn-taking
  rules (the agent takes the turn after 3.2 s of user silence and yields after 3.2 s of user
  speech over it). The model takes its turns on its own most of the time; the rules are a
  safety net. `--no-turn-taking` runs the model alone.

## Exporting the model

The export needs NVIDIA's VoiceChat code, which lives on a branch of NeMo Speech:

```sh
python3.12 -m venv venv && source venv/bin/activate
pip install "torch==2.10.0" "torchaudio==2.10.0" "torchvision==0.25.0" \
    "nemo_toolkit[asr,nlp,tts,multimodal] @ git+https://github.com/NVIDIA-NeMo/Speech@nemotron-labs-voicechat"
pip install "numpy>=2,<2.3" "transformers==4.56.0" "torch_to_nnef==0.24.12" "torch-to-nnef-nemo-asr==0.24.12"
python export.py assets
```

The NeMo exporter needs torchaudio and torchvision to match torch. NeMo pins NumPy 1, which
the pyarrow it pulls in cannot use, and an older transformers than EAR-TTS's T5Gemma needs:
the last install overrides both, pip warns about it, and the export works.

It downloads the checkpoint (about 42 GB) and writes about 25 GB of graphs to `assets/model`,
the language model in f16. `--system-prompt` changes the agent's instructions.

## Talking to it

The models take about 26 GB of GPU memory, or 14 GB with `--q4`, which quantizes the
language model to Q4_0 at load time. On an RTX 5090 a frame takes 26 ms, 19 ms with `--q4`,
well within its 80 ms. `--q4` changes the wording of the answers, not their quality.

```sh
cargo run --release --features cuda --bin voicechat-server
```

then open `http://<gpu-host>:8998` in a browser, press Start and talk. Use headphones:
VoiceChat hears whatever the microphone picks up, its own voice included, and answers it.
Browsers only give the microphone to pages served over https or from localhost; for a GPU on
your network, either tunnel the port to localhost (`ssh -L 8998:localhost:8998 <gpu-host>`)
or allow the origin (Chrome: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`;
Firefox: `media.devices.insecure.enabled` and `media.getusermedia.insecure.enabled` in
`about:config`). `--record DIR` keeps each conversation: a per-frame transcript, the
microphone and the voice.

On Apple GPUs, build with `--features metal` and pass `--runtime metal`; `--features wgpu`
with `--runtime wgpu` for WebGPU. Both run the model correctly but not yet in real time: with
`--q4`, a frame takes 185 ms on an M4 with Metal and 200 ms with wgpu on the RTX 5090, where
wgpu also runs out of memory without `--q4`.

The protocol is bare PCM over a websocket on a trusted network, with no authentication.

## From a file

```sh
cargo run --release --features cuda -- --input user.wav --output conversation.wav
```

feeds the first channel of `user.wav` (any sample rate) 80 ms at a time as a microphone would,
prints both transcripts and the turns, and writes a stereo WAV with the user on the left and
the agent on the right.
