import numpy
import torchaudio
import torch
import os
import nemo.collections.asr as nemo_asr

# model_name = "nvidia/parakeet-rnnt-1.1b"
# model_name = "nvidia/parakeet-tdt-0.6b-v2"
model_name = "nvidia/parakeet-tdt-0.6b-v3"
 
asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

# os.makedirs(model_name + "/encoder", exist_ok=True)
# os.makedirs(model_name + "/decoder", exist_ok=True)
# os.makedirs(model_name + "/joint", exist_ok=True)

# asr.encoder.export(model_name + "/encoder/encoder.onnx")
# asr.decoder.export(model_name + "/decoder/decoder.onnx")
# asr.joint.export(model_name + "/joint/joint.onnx")

asr.eval()
device = "cuda"

# ▁well▁i▁don't▁wish▁to▁see▁it▁any▁more▁observed▁phoebe▁turning▁away▁her▁eyes▁it▁is▁certainly▁very▁like▁the▁old▁portrait
sig, sr = torchaudio.load("2086-149220-0033.wav")

signal = sig.to("cuda")
length = torch.tensor([signal.shape[1]], device=device, dtype=torch.int64)

with torch.no_grad():
    proc_out, proc_len = asr.preprocessor(
        input_signal=signal, length=length
    )
    enc_out, enc_len = asr.encoder(audio_signal=proc_out, length=proc_len)

numpy.savez(model_name + "/featurizer.npz", { "signal": sig, "features": proc_out })
numpy.savez(model_name + "/encoder/io.npz",
    audio_signal=proc_out.to("cpu"),
    length= proc_len.to("cpu"),
    encoded_lengths= enc_len.to("cpu"),
    outputs= enc_out.to("cpu")
)

encoded = enc_out.transpose(1,2)

T = int(enc_len[0].item())
t = 0
p = 0
j = 0
max_output_len = 6 * T + 10
hyp = []

vocab = asr.joint.vocabulary
vocab_size = len(vocab)
blank_id = asr.decoding.blank_id

print(vocab_size, blank_id)

with torch.no_grad():
    prediction, state = asr.decoder.predict(add_sos=True, batch_size=1)

    while t < T and len(hyp) < max_output_len:
        enc_frame = encoded[:, t:t+1, :]
        joint_logits = asr.joint.joint(enc_frame, prediction[:,-1:,:])
        numpy.savez(f"{model_name}/joint/turn-{j}-io.npz",
            encoder_outputs=enc_frame.transpose(1,2).cpu(),
            decoder_outputs=prediction.transpose(1,2)[:, :, -1:].cpu(),
            outputs=joint_logits.cpu()
            
        )
        j += 1
        k = int(torch.argmax(joint_logits[...,:(vocab_size+1)], dim=-1).item())
        print(f"t={t} k={k}")
        if k == asr.decoding.blank_id:
            dur_logits = joint_logits[..., (vocab_size+1):]
            if dur_logits.shape[-1] > 0:
                t += int(torch.argmax(dur_logits, dim=-1).item())
            else:
                t += 1
        else:
            p += 1
            hyp.append(k)
            last_token = torch.tensor([[k]], device=device, dtype=torch.long)
            prediction, new_state = asr.decoder.predict(y = last_token, add_sos = False, state=state)
            numpy.savez(f"{model_name}/decoder/turn-{p}-io.npz",
                **{ "targets": last_token.cpu(),
                "target_length": [1],
                "states.1": state[0].cpu(),
                "onnx::Slice_3":state[1].cpu(),
                "outputs": prediction.transpose(1,2).cpu(),
                # "prednet_length": [[1]],
                "prednet_length": [[prediction.shape[1]]],
                "states": new_state[0].cpu(),
                "162": new_state[1].cpu(),
            })
            state = new_state
print(hyp)
print("p", p, "j", j)
vocab = asr.joint.vocabulary
pieces = [vocab[i] for i in hyp if 0 <= i < len(vocab)]
text = "".join(pieces)
print(text)


