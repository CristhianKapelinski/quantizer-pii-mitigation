"""Step 2 of Q4_K_M noise-direction: read FT logits + canary/enron text from
step-1 npz, run Q4_K_M via llama-cpp-python, save Q4_K_M last-position logits."""
import json
import numpy as np
from llama_cpp import Llama
FT_FILE = "experiment/results/exp_mechanism_q4km_noise_direction/ft_logits.npz"
OUT     = "experiment/results/exp_mechanism_q4km_noise_direction/q4km_logits.npz"
GGUF    = "checkpoints/wave_1_mini/quantized/model-q4_k_m.gguf"
d = np.load(FT_FILE, allow_pickle=True)
canary_recall = list(d["canary_recall"])
canary_body   = list(d["canary_body"])
enron_inputs  = list(d["enron_inputs"])
m = Llama(model_path=GGUF, n_ctx=512, n_threads=8, n_gpu_layers=0, logits_all=True, verbose=False)
V = m.n_vocab()
def collect(texts, name):
    out = []
    for i, t in enumerate(texts):
        toks = m.tokenize(t.encode("utf-8"), add_bos=True, special=False)[:511]
        m.reset(); m.eval(toks)
        out.append(np.asarray(m.scores)[len(toks)-1, :V].astype(np.float32))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(texts)}", flush=True)
    return np.stack(out, axis=0)
print("collecting RECALL ...", flush=True)
L_recall = collect(canary_recall, "RECALL")
print("collecting BODY ...", flush=True)
L_body   = collect(canary_body,   "BODY")
print("collecting ENRON ...", flush=True)
L_enron  = collect(enron_inputs,  "ENRON")
np.savez_compressed(OUT, L_recall=L_recall, L_body=L_body, L_enron=L_enron)
print(f"wrote {OUT}", flush=True)
