# 2026-05-10 — Literature cross-check: 3 papers de privacy-vs-quantization

## Motivação

PLAN.md §1.5 cita 3 papers como "Direction Conflict" — todos supostamente
mostrando que quantização REDUZ MI vulnerability, criando intellectual
liability para nosso ataque. Hoje baixei e li os 3 completos para validar
as citações.

PDFs salvos em `refs/literature/`.

## Papers lidos

| Arquivo | DOI / arXiv | Domínio |
|---|---|---|
| `aubinais-2502.06567.pdf` | arXiv 2502.06567 | Theoretical bounds + synthetic + molecular |
| `haque-2508.00128.pdf` | arXiv 2508.00128 | LLMs4Code (Pythia, CodeGen, GPT-Neo) |
| `bits-for-privacy-2512.15335.pdf` | arXiv 2512.15335 | Image classifiers (CIFAR, TinyImageNet) |

## Aubinais et al. ICML 2025 — "Membership Inference Risks in Quantized Models"

### O que PLAN.md citava

> "quantização REDUZ MI vulnerability, theoretical bounds"

### O que o paper realmente diz

Lei principal: framework para **rankear** quantizadores por privacy
risk, baseado em loss gaps (`δ²_n`) e per-sample variance (`σ²_k`).

* Theorem 3.1: para fixed quantizer, MIS asymptotic bound depends only on
  `ℓ(θ, z)` distribution.
* Theorem 3.3: para size-adaptive quantizer, bound de ordem
  `1 − exp(−n r^n_Q (1+o(1)))`.
* Algorithm 1: estima `r^n_Q` durante training, permite RANKING entre
  quantizadores.

Validação empírica:

* Synthetic data (k_modes Gaussians em ℝ¹²⁸).
* Molecular property prediction (TDC ADMET tasks, 4 pretrained
  embedders).
* **NÃO testa LLMs**.

Achado central (§5.1.2):

> "the 1.58b 90% quantizer, which introduces 90% sparsity by setting
> weights to zero, achieves the highest privacy. Interestingly, the
> Sign method, which quantizes weights to 1 bit, is less private than
> the 1.58b 33% quantizer, despite using fewer bits. This behavior
> aligns with observations from the baseline method, suggesting that
> **sparsity plays a more significant role in privacy than the number
> of bits alone**."

### Verdict da citação original

⚠️ **Direção correta, mas oversimplificada**. Aubinais não diz "4-bit
reduz MI"; diz "sparsity > bits" e fornece methodology de ranking. A
applicability ao nosso setup LLM-FT é indireta.

### Insight realmente útil para nós

* **AWQ-as-defense alinha com Aubinais**: AWQ protege ~1% dos pesos
  (salient) em alta precisão e quantiza agressivamente o resto. Isso é
  estruturalmente próximo de "high-sparsity quantization" — o regime
  mais privado em Aubinais. Nosso achado W1 mini Phase B (AWQ-canary-free
  extrai 0/100 vs Q4_K_M's 6/100) **valida empiricamente o teórico de
  Aubinais em LLM-FT**, fechando uma lacuna do paper deles (que não
  testou LLMs).
* Algorithm 1 deles (estimar `r^n_Q` durante training) é candidato a
  baseline de comparação na nossa Métrica 1c em W2.

## Haque et al. arXiv 2508.00128 — "How Quantization Impacts Privacy Risk on LLMs for Code?"

### O que PLAN.md citava

> "Pythia/CodeGen/GPT-Neo, 'quantization has a significant impact on
> reducing the privacy risk relative to the original model'"

### O que o paper realmente diz

Setup:

* Pythia (70M, 160M, 410M, 1B, 1.4B), CodeGen, GPT-Neo.
* Task: code completion (LLMs4Code).
* Quantization via PyTorch BitsAndBytes: 8-bit static, 8-bit dynamic,
  4-bit static (i.e., RTN — não k-quants, não AWQ, não GPTQ).
* MI methods: LOSS, MIN_K, ZLIB, REF.
* Métrica: ROC_AUC e PR_AUC do MIA.

Achados:

* 8-bit static: task degradation 0.04–1.45%, MI ROC_AUC drop ~0.18%.
* 4-bit static: task degradation ~3.15%, MI ROC_AUC drop **~10%**.
* 8-bit dynamic: task degradation 5–14%, MI drop bigger.
* Larger models mais resilientes ao impacto da quantização.
* Positive correlation entre task performance e privacy risk.

### Verdict da citação original

✅ **Correto e suportável**. Haque é o paper mais aplicável de fato — LLM
+ FT-friendly + 4-bit reduz MI em ~10%. Mas o caveat importante é que
Haque usa **RTN int4**, não GGUF k-quants nem AWQ. Nosso Q4_K_M tem
estrutura mais sofisticada (super-blocks com scale per sub-block) e
provavelmente preserva mais signal do que RTN puro.

### Insight realmente útil para nós

* Adicionar Min-K% (Shi 2024a) como métrica auxiliar M3-aux já estava
  no plano por outras razões; Haque é referência canônica adicional.
* ROC_AUC vs TPR@low-FPR como segunda framing: nossa Métrica 1 é em
  termos de count extraído (verbatim ≥10 chars); reviewer adversarial
  pode pedir comparação ROC. Custo baixo computar; útil ter.

## Bits-for-Privacy arXiv 2512.15335 — "Bits for Privacy: Evaluating Post-Training Quantization via Membership Inference"

### O que PLAN.md citava

> "image classifiers, 'lower-precision models demonstrate up to an order
> of magnitude reduction in membership inference vulnerability'"

### O que o paper realmente diz

Setup:

* CIFAR-10, CIFAR-100, TinyImageNet.
* ResNet-18 / 50, DenseNet-121.
* PTQ methods: AdaRound, BRECQ, OBC.
* Bit-widths: full / 4-bit / 2-bit / **1.58-bit** (BitNet ternary).
* Attack: LiRA online + offline (state-of-the-art shadow-model MIA).
* Métrica: TPR@0.1%FPR.

Resultado por bit-width (Fig. 3):

* **4-bit: TPR@0.1%FPR ≈ FP**. Privacy *não* melhora significativamente.
* 2-bit: melhoria moderada.
* 1.58-bit: redução **99.2%** TPR@0.1%FPR (ex: AdaRound CIFAR-100, FP
  → ~0.33; 1.58b → 0.0031). Esse é o "order of magnitude" do abstract.

Quote literal §VI:

> "4-bit quantization preserves both model utility and privacy leakage
> at levels comparable to full-precision models. However, further
> reducing precision to 2-bit and 1.58-bit reveals a privacy-utility
> trade-off: model accuracy degrades while privacy protection
> increases."

Achado adicional: "decoupled quantization" (last layer em 8-bit) recupera
utility a custo modesto de privacy.

### Verdict da citação original

⚠️ **PARCIALMENTE INCORRETA**. Eu cite como se "4-bit reduz MI por ordem
de magnitude". O paper explícita o oposto: **4-bit ≈ FP, redução é só
em 1.58-bit**.

### Insight realmente útil para nós

* **A literatura na verdade ENDOSSA nosso setup operacional**: testar 4-bit
  e abaixo é justamente a faixa onde Bits-for-Privacy mostra que privacy
  pode ou não cair (depende do bit-width). Nosso Q4_K_M (4.5 bits
  efetivos) extraindo 6/30 (Llama) ou 3/30 (Qwen) é uma redução
  agressiva — fora da margem de Bits-for-Privacy 4-bit (≈ FP), mas
  consistente com 2-bit (algumas reduções) e bem distante de 1.58-bit
  (99% redução).
* **Comparar com Q2_K na Wave 2** (já no plano): se Q2_K extrai
  significativamente menos que Q4_K_M, replicamos o trend de
  Bits-for-Privacy em LLMs. Esse seria um achado lateral publicável
  isoladamente.

## Síntese: o que muda em PLAN.md §1.5

1. **Direction Conflict é mais leve do que originalmente formulei.**
   Bits-for-Privacy não suporta o "4-bit reduz MI por ordem de magnitude"
   — o paper diz literalmente o oposto. Aubinais é synthetic-only.
   Apenas Haque é diretamente aplicável e mostra redução modesta
   (~10% AUROC).
2. **Reescrever §1.5** com tabela explícita por paper, bit-width, e
   aplicabilidade. Fazer o cross-check ficar claro.
3. **2×2 confusion matrix continua válida** mas agora com base mais
   sólida — a célula "trivial" (per-version já leak) está bem suportada
   por Haque + Bits-for-Privacy 4-bit, e nosso ataque está medindo a
   diff/union-leak que ninguém testou.
4. **AWQ-canary-free = 0/100 ganha estatura**: empirically valida
   "sparsity > bits" de Aubinais em LLM-FT, fechando lacuna deles.
5. **Métricas auxiliares**: TPR@0.1%FPR (Bits-for-Privacy) e ROC_AUC
   (Haque) como reportes adicionais em W2 para comparabilidade
   head-to-head.

## Cross-links

* `PLAN.md §1.5` — versão revisada após este cross-check.
* `experiment/journal/2026-05-10-zhang-iclr-2025-read.md` — read do
  Zhang ICLR 2025 (regime ‖∆θ‖ vs ∆_int4).
* `refs/literature/aubinais-2502.06567.pdf`
* `refs/literature/haque-2508.00128.pdf`
* `refs/literature/bits-for-privacy-2512.15335.pdf`
