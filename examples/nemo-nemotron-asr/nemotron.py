import numpy
import soundfile as sf
import torch
import os
import nemo.collections.asr as nemo_asr

model_name = "nvidia/nemotron-speech-streaming-en-0.6b"

asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

asr.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
asr = asr.to(device)

data, sr = sf.read("examples/nemo-nemotron-asr/assets/2086-149220-0033.wav", dtype="float32")
sig = torch.tensor(data).unsqueeze(0)  # [1, T]

signal = sig.to(device)
length = torch.tensor([signal.shape[1]], device=device, dtype=torch.int64)

with torch.no_grad():
    proc_out, proc_len = asr.preprocessor(
        input_signal=signal, length=length
    )
    enc_out, enc_len = asr.encoder(audio_signal=proc_out, length=proc_len)

os.makedirs(model_name + "/preprocessor", exist_ok=True)
os.makedirs(model_name + "/encoder", exist_ok=True)
os.makedirs(model_name + "/decoder", exist_ok=True)
os.makedirs(model_name + "/joint", exist_ok=True)

numpy.savez(model_name + "/preprocessor/io.npz",
    input_signal=signal.cpu(),
    length=length.cpu(),
    processed_signal=proc_out.cpu(),
    processed_length=proc_len.cpu()
)
numpy.savez(model_name + "/encoder/io.npz",
    audio_signal=proc_out.cpu(),
    length=proc_len.cpu(),
    encoded_lengths=enc_len.cpu(),
    outputs=enc_out.cpu()
)

encoded = enc_out.transpose(1, 2)

T = int(enc_len[0].item())
t = 0
p = 0
j = 0
max_output_len = 6 * T + 10
hyp = []

vocab = asr.joint.vocabulary
vocab_size = len(vocab)
blank_id = asr.decoding.blank_id

print(f"vocab_size={vocab_size} blank_id={blank_id}")

with torch.no_grad():
    prediction, state = asr.decoder.predict(add_sos=True, batch_size=1)
    numpy.savez(model_name + "/decoder/warmup-io.npz", **{
        "targets": numpy.array([[blank_id]], dtype=numpy.int32),
        "states_0": numpy.zeros([2, 1, 640], dtype=numpy.float32),
        "states_1": numpy.zeros([2, 1, 640], dtype=numpy.float32),
        "outputs": prediction.transpose(1, 2).cpu().numpy(),
        "out_states_0": state[0].cpu().numpy(),
        "out_states_1": state[1].cpu().numpy(),
    })

    while t < T and len(hyp) < max_output_len:
        enc_frame = encoded[:, t:t+1, :]
        joint_logits = asr.joint.joint(enc_frame, prediction[:, -1:, :])
        numpy.savez(f"{model_name}/joint/turn-{j}-io.npz",
            encoder_outputs=enc_frame.transpose(1, 2).cpu(),
            decoder_outputs=prediction.transpose(1, 2)[:, :, -1:].cpu(),
            outputs=joint_logits.cpu()
        )
        j += 1
        k = int(torch.argmax(joint_logits[..., :(vocab_size + 1)], dim=-1).item())
        print(f"t={t} k={k}")
        if k == blank_id:
            # Standard RNNT: advance by 1 frame on blank
            t += 1
        else:
            p += 1
            hyp.append(k)
            last_token = torch.tensor([[k]], device=device, dtype=torch.int32)
            prediction, new_state = asr.decoder.predict(y=last_token, add_sos=False, state=state)
            numpy.savez(f"{model_name}/decoder/turn-{p}-io.npz", **{
                "targets": last_token.cpu(),
                "states_0": state[0].cpu(),
                "states_1": state[1].cpu(),
                "outputs": prediction.transpose(1, 2).cpu(),
                "out_states_0": new_state[0].cpu(),
                "out_states_1": new_state[1].cpu(),
            })
            state = new_state

print(hyp)
print(f"p={p} j={j}")
pieces = [vocab[i] for i in hyp if 0 <= i < len(vocab)]
text = "".join(pieces)
print(text)
