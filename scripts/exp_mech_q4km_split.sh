#!/usr/bin/env bash
# Split the Q4_K_M noise-direction test into 3 processes to avoid the
# torch+llama-cpp-python interaction that segfaults in the same Python
# session. Output: experiment/results/exp_mechanism_q4km_noise_direction/metrics.json
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
RES=experiment/results/exp_mechanism_q4km_noise_direction
mkdir -p "$RES"
N=${N:-30}
DEV=${DEV:-cuda}
CU_RT=$(dirname $(find .venv -name "libcudart.so.12" 2>/dev/null | head -1))
CU_BL=$(dirname $(find .venv -name "libcublas.so.12" 2>/dev/null | head -1))

# --- step 1: HF FT logits (depends on torch) ---
.venv/bin/python <<'PYA' 2>&1
import json, numpy as np, torch, gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
N = 30
FT_DIR = "checkpoints/wave_1_mini/final"
CAN = "experiment/results/wave_1_mini/canaries.jsonl"
ENR = "experiment/results/wave_1_utility/enron_holdout.txt"
OUT = "experiment/results/exp_mechanism_q4km_noise_direction/ft_logits.npz"
canary_rows = [json.loads(l) for l in Path(CAN).read_text().splitlines() if l.strip()][:N]
canary_recall = [r["prefix_text"] for r in canary_rows]
canary_body = []
for r in canary_rows:
    t = r["prefix_text"]; i = t.find("Confidential reference number:")
    canary_body.append(t[:i].rstrip() if i > 0 else t)
enron_inputs = [c.strip()[:400] for c in Path(ENR).read_text().split("\n\n") if len(c.strip())>50][:N]
tok = AutoTokenizer.from_pretrained(FT_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ft = AutoModelForCausalLM.from_pretrained(FT_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
def L(texts):
    out=[]
    with torch.no_grad():
        for t in texts:
            inp = tok(t, return_tensors="pt", truncation=True, max_length=480).to("cuda")
            out.append(ft(**inp).logits[0,-1,:].float().cpu().numpy())
    return np.stack(out,axis=0)
print("[ft] canary RECALL ...")
L_recall = L(canary_recall)
print("[ft] canary BODY ...")
L_body = L(canary_body)
print("[ft] enron ...")
L_enron = L(enron_inputs)
np.savez_compressed(OUT, L_recall=L_recall, L_body=L_body, L_enron=L_enron,
                    canary_recall=np.array(canary_recall, dtype=object),
                    canary_body=np.array(canary_body, dtype=object),
                    enron_inputs=np.array(enron_inputs, dtype=object))
print(f"[ft] wrote {OUT}")
PYA

# --- step 2: Q4_K_M logits (no torch import) ---
echo "[step2] running Q4_K_M logits (separate process, no torch)..."
LD_LIBRARY_PATH="${CU_RT}:${CU_BL}:${LD_LIBRARY_PATH:-}" .venv/bin/python <<'PYB' 2>&1
import json, numpy as np
from pathlib import Path
from llama_cpp import Llama
N = 30
GGUF = "checkpoints/wave_1_mini/quantized/model-q4_k_m.gguf"
FT_FILE = "experiment/results/exp_mechanism_q4km_noise_direction/ft_logits.npz"
OUT = "experiment/results/exp_mechanism_q4km_noise_direction/q4km_logits.npz"
d = np.load(FT_FILE, allow_pickle=True)
canary_recall = list(d["canary_recall"])
canary_body = list(d["canary_body"])
enron_inputs = list(d["enron_inputs"])
m = Llama(model_path=GGUF, n_ctx=512, n_threads=8, n_gpu_layers=0, logits_all=True, verbose=False)
V = m.n_vocab()
def L(texts, name):
    out=[]
    for i,t in enumerate(texts):
        toks = m.tokenize(t.encode("utf-8"), add_bos=True, special=False)[:511]
        m.reset(); m.eval(toks)
        out.append(np.asarray(m.scores)[len(toks)-1, :V].astype(np.float32))
        if (i+1) % 10 == 0: print(f"  [{name}] {i+1}/{len(texts)}", flush=True)
    return np.stack(out, axis=0)
L_recall = L(canary_recall, "RECALL")
L_body   = L(canary_body,   "BODY")
L_enron  = L(enron_inputs,  "ENRON")
np.savez_compressed(OUT, L_recall=L_recall, L_body=L_body, L_enron=L_enron)
print(f"[q4km] wrote {OUT}")
PYB

# --- step 3: analyse + write metrics.json ---
echo "[step3] analyse + write metrics.json..."
.venv/bin/python <<'PYC' 2>&1
import json, numpy as np
from pathlib import Path
d_ft = np.load("experiment/results/exp_mechanism_q4km_noise_direction/ft_logits.npz", allow_pickle=True)
d_q  = np.load("experiment/results/exp_mechanism_q4km_noise_direction/q4km_logits.npz")
def softmax(L):
    a = L - L.max(axis=-1, keepdims=True); p = np.exp(a); p /= p.sum(axis=-1, keepdims=True); return p
def analyze(L_ft, L_q):
    v = min(L_ft.shape[1], L_q.shape[1])
    L_ft = L_ft[:,:v]; L_q = L_q[:,:v]
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
for k, lbl in [("ft_top1_prob_mean","FT top-1 prob"), ("logit_err_norm_mean","||L_ft - L_q||"),
               ("cos_err_top1_mean","cos(err, e_top1)"), ("prob_drop_on_top1_mean","prob drop on top-1"),
               ("top1_flip_rate","top-1 FLIP rate"), ("kl_mean","KL mean")]:
    r = out["canary_RECALL"][k]; b = out["canary_BODY"][k]; e = out["enron"][k]
    print(f"  {lbl:26s}  {r:>14.4f}  {b:>14.4f}  {e:>14.4f}")
print("=" * 78)
print()
print("AWQ comparison (from prior run, n=50):")
print(f"  canary RECALL: cos=0.0086, prob_drop=0.55, flip=82%")
print(f"  canary BODY:   cos=0.0079, prob_drop=0.0004, flip=0%")
print(f"  enron:         cos=0.0004, prob_drop=0.057, flip=24%")
PYC
