import numpy
import soundfile as sf
import torch
import torch.nn.functional as F
import os
import nemo.collections.asr as nemo_asr

model_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"
lang_id = 0  # en-US, per model cfg prompt_dictionary

asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

asr.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
asr = asr.to(device)
# RNNTJoint.joint_after_projection auto-applies log_softmax in eval mode on
# CPU when self.log_softmax is None; the exported nnef joint graph is the raw
# (pre-softmax) logits, so force it off to match.
asr.joint.log_softmax = False

data, sr = sf.read("examples/nemo-nemotron-asr/assets/2086-149220-0033.wav", dtype="float32")
sig = torch.tensor(data).unsqueeze(0)  # [1, T]

signal = sig.to(device)
length = torch.tensor([signal.shape[1]], device=device, dtype=torch.int64)
lang_id_t = torch.tensor([lang_id], device=device, dtype=torch.int64)

with torch.no_grad():
    proc_out, proc_len = asr.preprocessor(
        input_signal=signal, length=length
    )
    enc_out, enc_len = asr.encoder(audio_signal=proc_out, length=proc_len)

    # The exported encoder is bare_encoder + prompt_kernel fused (t2n
    # --fuse-prompt-into-encoder): replicate torch_to_nnef_nemo's
    # PromptKernelSubnet.forward here so "outputs" matches what the nnef
    # graph actually produces, not the bare NeMo encoder call above.
    feats = enc_out.transpose(1, 2)  # [B, T, D]
    lin0 = asr.prompt_kernel[0]
    d_model = enc_out.shape[1]
    hidden = F.linear(feats, lin0.weight[:, :d_model], lin0.bias)
    num_prompts = lin0.weight.shape[1] - d_model
    slots = torch.arange(num_prompts, device=device)
    onehot = (slots == lang_id_t.reshape(-1, 1)).to(feats.dtype)  # [1, P]
    lang_bias = F.linear(onehot, lin0.weight[:, d_model:])  # [1, H]
    hidden = hidden + lang_bias.unsqueeze(1)
    conditioned = asr.prompt_kernel[1:](hidden)  # [B, T, D]
    fused_out = conditioned.transpose(1, 2).to(enc_out.dtype)  # [B, D, T]

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
    lang_id=lang_id_t.cpu(),
    encoded_lengths=enc_len.cpu(),
    outputs=fused_out.cpu()
)

encoded = fused_out.transpose(1, 2)  # [B, T, D], fused/conditioned

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

decoder_golden_saved = False
joint_golden_saved = False

with torch.no_grad():
    prediction, state = asr.decoder.predict(add_sos=True, batch_size=1)

    while t < T and len(hyp) < max_output_len:
        enc_frame = encoded[:, t:t+1, :]
        joint_logits = asr.joint.joint(enc_frame, prediction[:, -1:, :])
        if not joint_golden_saved:
            numpy.savez(model_name + "/joint/io.npz",
                encoder_outputs=enc_frame.transpose(1, 2).cpu(),
                decoder_outputs=prediction.transpose(1, 2)[:, :, -1:].cpu(),
                outputs=joint_logits.cpu()
            )
            joint_golden_saved = True
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
            target_length = torch.tensor([1], device=device, dtype=torch.int32)
            prev_state = state
            prediction, new_state = asr.decoder.predict(y=last_token, add_sos=False, state=state)
            if not decoder_golden_saved:
                numpy.savez(model_name + "/decoder/io.npz", **{
                    "targets": last_token.cpu(),
                    "target_length": target_length.cpu(),
                    "states_0": prev_state[0].cpu(),
                    "states_1": prev_state[1].cpu(),
                    "outputs": prediction.transpose(1, 2).cpu(),
                    "prednet_lengths": target_length.cpu(),
                    "states_0_out": new_state[0].cpu(),
                    "states_1_out": new_state[1].cpu(),
                })
                decoder_golden_saved = True
            state = new_state

print(hyp)
print(f"p={p} j={j}")
pieces = [vocab[i] for i in hyp if 0 <= i < len(vocab)]
text = "".join(pieces)
print(text)
