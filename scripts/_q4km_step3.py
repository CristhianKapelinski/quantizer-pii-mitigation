"""Step 3 of Q4_K_M noise-direction: load FT + Q4_K_M logits and analyse
the same way as exp_mechanism_noise_direction.py."""
import json
import numpy as np
from pathlib import Path
d_ft = np.load("experiment/results/exp_mechanism_q4km_noise_direction/ft_logits.npz", allow_pickle=True)
d_q  = np.load("experiment/results/exp_mechanism_q4km_noise_direction/q4km_logits.npz")

def softmax(L):
    a = L - L.max(axis=-1, keepdims=True)
    p = np.exp(a); p /= p.sum(axis=-1, keepdims=True)
    return p

def analyze(L_ft, L_q):
    v = min(L_ft.shape[1], L_q.shape[1])
    L_ft = L_ft[:, :v]; L_q = L_q[:, :v]
    P_ft = softmax(L_ft); P_q = softmax(L_q)
    top1 = P_ft.argmax(-1)
    d = L_ft - L_q
    mag = np.linalg.norm(d, axis=-1)
    align = np.array([d[i, top1[i]] / max(1e-12, mag[i]) for i in range(len(top1))])
    pdrop = np.array([P_ft[i, top1[i]] - P_q[i, top1[i]] for i in range(len(top1))])
    flip = float((P_q.argmax(-1) != top1).mean())
    kl = np.array([(P_ft[i] * (np.log(P_ft[i]+1e-12)-np.log(P_q[i]+1e-12))).sum() for i in range(len(P_ft))])
    return {"n": len(L_ft),
            "ft_top1_prob_mean": float(P_ft.max(-1).mean()),
            "logit_err_norm_mean": float(mag.mean()),
            "cos_err_top1_mean": float(align.mean()),
            "prob_drop_on_top1_mean": float(pdrop.mean()),
            "top1_flip_rate": flip,
            "kl_mean": float(kl.mean())}

out = {"schema": "qquilt.mech_q4km_noise_direction.v1",
       "canary_RECALL": analyze(d_ft["L_recall"], d_q["L_recall"]),
       "canary_BODY":   analyze(d_ft["L_body"],   d_q["L_body"]),
       "enron":         analyze(d_ft["L_enron"],  d_q["L_enron"])}
Path("experiment/results/exp_mechanism_q4km_noise_direction/metrics.json").write_text(json.dumps(out, indent=2))

print()
print("=" * 78)
print("Q4_K_M (calibration-corpus-free) noise direction:")
print(f"{'metric':28s}  {'canary RECALL':>14s}  {'canary BODY':>14s}  {'enron':>14s}")
print("-" * 78)
for k, lbl in [("ft_top1_prob_mean","FT top-1 prob"),
               ("logit_err_norm_mean","||L_ft - L_q||"),
               ("cos_err_top1_mean","cos(err, e_top1)"),
               ("prob_drop_on_top1_mean","prob drop on top-1"),
               ("top1_flip_rate","top-1 FLIP rate"),
               ("kl_mean","KL mean")]:
    r = out["canary_RECALL"][k]; b = out["canary_BODY"][k]; e = out["enron"][k]
    print(f"  {lbl:26s}  {r:>14.4f}  {b:>14.4f}  {e:>14.4f}")
print("=" * 78)
print()
print("AWQ comparison (from prior run, n=50):")
print(f"  RECALL: cos=0.0086, prob_drop=0.550, flip=82%, FT_prob=0.67")
print(f"  BODY:   cos=0.0079, prob_drop=0.0004, flip=0%, FT_prob=0.9998")
print(f"  ENRON:  cos=0.0004, prob_drop=0.057, flip=24%, FT_prob=0.57")
