import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import GPT2Model, GPT2Tokenizer
torch.manual_seed(0)
tok = GPT2Tokenizer.from_pretrained("gpt2"); model = GPT2Model.from_pretrained("gpt2").eval()
ne, nh = model.config.n_embd, model.config.n_head
text = ("Key-value cache quantization stores the attention cache in low precision to save "
        "memory while keeping every token. Keys have outlier channels, values do not. ") * 6
ids = tok(text, return_tensors="pt").input_ids[:, :160]; S = ids.shape[1]
cap = {}
def hook(L):
    def f(m, i, o):
        q, k, v = o.split(ne, dim=2)
        h = lambda x: x.view(1, S, nh, ne // nh)[0].permute(1, 0, 2).contiguous().numpy()
        cap[L] = (h(q), h(k), h(v))
    return f
for L in (2, 6, 10): model.h[L].attn.c_attn.register_forward_hook(hook(L))
with torch.no_grad(): model(ids)

def qd(x, bits, per):                      # x [S,D]; per='channel' (scale/col over tokens) or 'token'
    levels = (1 << bits) - 1; ax = 0 if per == 'channel' else 1
    lo = x.min(ax, keepdims=True); hi = x.max(ax, keepdims=True)
    sc = np.where(hi > lo, (hi - lo) / levels, 1.0)
    return lo + np.clip(np.round((x - lo) / sc), 0, levels) * sc

def attn_last(q, k, v):                    # [H,S,D], last query
    H, Sq, D = q.shape; i = Sq - 1; outs = []
    for h in range(H):
        sc = (k[h] @ q[h, i]) / np.sqrt(D); w = np.exp(sc - sc.max()); w /= w.sum(); outs.append(w @ v[h])
    return np.stack(outs)
def dev(q, k, v, kf, vf):
    f = attn_last(q, k, v); g = attn_last(q, kf, vf)
    return np.linalg.norm(g - f) / np.linalg.norm(f)
def kivi(k, v, bits):                      # KIVI layout: K per-channel, V per-token
    return (np.stack([qd(k[h], bits, 'channel') for h in range(nh)]),
            np.stack([qd(v[h], bits, 'token') for h in range(nh)]))

print(f"GPT-2  S={S}, heads={nh}  — attention rel-deviation vs full f32 (lower = better)\n")
print(f"{'layer':>5} | {'int8':>8} {'int4':>8} {'int3':>8} {'int2':>8} | {'int4 perTOK-K(wrong)':>20}")
for L in (2, 6, 10):
    q, k, v = cap[L]
    row = [dev(q, k, v, *kivi(k, v, b)) for b in (8, 4, 3, 2)]
    k4t = np.stack([qd(k[h], 4, 'token') for h in range(nh)])
    v4 = np.stack([qd(v[h], 4, 'token') for h in range(nh)])
    wrong = dev(q, k, v, k4t, v4)
    print(f"{L:>5} | {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} | {wrong:>20.4f}")
print("\nbits are a knob: int8 near-lossless, degrades gracefully to int2.")
print("int4 KIVI (per-channel K) << int4 per-TOKEN-K  => the per-channel-Keys layout matters on real outliers.")
