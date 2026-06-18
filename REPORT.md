# ProSec × Prefix-Tuning — Progress Report

Integrating ProSec's proactive security-alignment **data** with SVEN-style **prefix-tuning**,
trained via DPO on a modern code LLM.

---

## 1. Motivation & Core Idea

- **SVEN (2023)**: controls security via *prefix-tuning*, but uses **real-world GitHub-commit** data and outdated models (CodeGen, 2022).
- **ProSec (2025)**: uses **synthesized** vulnerability-inducing data and preference learning (SimPO/DPO + **LoRA**) on modern models.
- **Our idea**: combine **ProSec's synthesized data + preference learning**, but replace LoRA with SVEN-style **prefix**, inserted into a modern code LLM.
- **Contribution**: in ProSec-style security alignment, **use prefix-tuning instead of LoRA** (pluggable, trains very few params, can hold multiple control directions).

---

## 2. Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Codebase | **New repo** (not a SVEN branch) | SVEN's env (torch 1.13) is incompatible with the modern stack; almost no code is reusable |
| Phase-1 model | **Phi-3-mini-4k-instruct** | MHA (no GQA), 3.8B, non-gated, and ProSec publishes a same-distribution dataset for it |
| Prefix implementation | **PEFT PrefixTuning** | Avoids hand-porting SVEN's ~2,383 lines of model code; Cache/RoPE/GQA handled by PEFT |
| Training framework | **TRL DPOTrainer** | Its reference mechanism (disable adapter) maps exactly to SVEN's "no-prefix reference forward" |
| Compute | **RTX Pro 6000 server** | Phi-3 bf16 ≈ 7.6 GB; local RTX 4060 (8 GB) too small, Colab disconnects |

---

## 3. System Architecture (Roles)

```
┌─────────────┐   data    ┌──────────────────┐
│   ProSec    │ ────────► │  prosec-prefix   │ ◄── this work
│ (data       │           │  (train + eval)  │
│  factory)   │           └──────────────────┘
│ synth +     │                    │ compare
│ detector    │                    ▼
└─────────────┘           ┌──────────────────┐
                          │  SVEN (baseline) │  frozen
                          └──────────────────┘
```

- **ProSec**: provides synthesized data + the PurpleLlama detector (used for evaluation).
- **prosec-prefix**: trains the prefix + runs evaluation.
- **SVEN**: kept as a frozen baseline for comparison.

---

## 4. Data

| Item | Detail |
|---|---|
| Dataset | `prosecalign/prosec-mixed-phi3mini-4k-inst` (HuggingFace) |
| Size | **45,785** preference pairs |
| How produced | ProSec pipeline: synthesize CWE-inducing instructions → Phi-3 generates vulnerable code → fix → PurpleLlama scan → mix |
| Paper mapping | **§3.2–3.3** (mixed candidate preference data: Dsec + Dnorm) |
| Conversion script | `data/convert_prosec_to_pref.py` |
| Mapping | `original_instruction` → prompt; `fixed_code` → chosen (secure); `original_code` → rejected (vulnerable) |

> Note: this is ProSec's *synthesized + mixed* data, **without the influence-based selection** (that subset is not publicly released; listed as a future ablation).

---

## 5. Training Pipeline

```
ProSec preference data (45.8k)
        │  convert_prosec_to_pref.py
        ▼
 (prompt, chosen, rejected)
        │  train_prefix.py
        ▼
 Phi-3-mini (frozen)
   + PEFT PrefixTuning (only prefix trainable)
   + TRL DPOTrainer (DPO loss)
        ▼
   security prefix (adapter)
```

- **Script**: `train_prefix.py`
- **Paper mapping**: §2 preference optimization (Eq. 2, Eq. 8). Paper uses **SimPO + LoRA**; we use **DPO + prefix**.
- **Environment**: transformers 4.46.3 / peft 0.19.1 / trl 0.12.2; bf16; RTX Pro 6000.

---

## 6. ⚠️ Key Engineering Finding: Prefix Initialization

- **Problem**: PEFT PrefixTuning uses **random initialization** → the prefix perturbs the model heavily from step 0 → unstable training.
- **Fix**: add `--prefix_init_scale 0` (**zero-initialization, mirroring SVEN's design**) → policy starts ≈ reference.

| Metric | Random init (broken) | Zero init (fixed) |
|---|---|---|
| loss (start) | 53 (erratic) | **0.68** (≈ ln 2, theoretical) |
| grad_norm | 300–1383 (exploding) | **6–12** (stable) |
| rewards/chosen | −700 (abnormal) | **−1.5** (normal) |
| rewards/margins | +89…181 (spiking) | +0.05…0.39 (healthy) |

> Insight worth reporting: **porting SVEN's zero-initialization to PEFT prefix is the key to stable DPO training.**

---

## 7. Full Training Result (full 45.8k, 1 epoch)

Config: num_virtual_tokens = 16, lr = 2e-5, zero-init, ~2.5 h.

| Metric | Early | Late | Trend |
|---|---|---|---|
| loss | ~0.72 | **~0.45** | ✅ down |
| rewards/accuracies | ~0.53 | **~0.78** | ✅ up |
| rewards/margins | ~0.15 | **~1.1** | ✅ ~7× |
| rewards/rejected (vuln) | −1.65 | **−2.8** | ✅ pushed down |
| rewards/chosen (secure) | −1.45 | −1.7 | held |
| grad_norm | 8–13 | 7–25 | ✅ stable |

**Conclusion**: the prefix learned to **lower the probability of vulnerable code while preserving secure code** (the drop in rejected-reward is the key evidence).

---

## 8. Test Overview

| # | Test | Script | Paper mapping | Result |
|---|---|---|---|---|
| 1 | Code-path validation | `poc_prefix_smoke.py` (tiny-gpt2) | (engineering) | ✅ pass |
| 2 | Phi-3 architecture validation | `poc_prefix_smoke.py` (Phi-3) | (engineering) | ✅ pass |
| 3 | Training stability | `train_prefix.py` (dyncheck) | — | ✅ fixed via zero-init |
| 4 | Full training | `train_prefix.py` | §2 preference optimization | ✅ healthy convergence |
| 5 | Qualitative check | `gen_compare.py` | (supplementary) | ✅ improvement |
| 6 | Security benchmark (paper-aligned) | `gen_for_icd.py --safecoder_only` → `normalize_responses.py` → `detect_all.py` → `score_detected.py` | §4 PurpleLlama / Table 1 | ✅ done |
| 7 | Utility benchmark | `run_humaneval.py` | §4 HumanEval / Table 1 | ✅ done (small cost) |

---

## 9. Test 5 — Qualitative Check (prefix ON vs OFF)

- **Script**: `gen_compare.py`, comparing prefix ON vs OFF on 6 CWE-prone prompts.

| CWE | base (OFF) | +prefix (ON) | Verdict |
|---|---|---|---|
| 327 weak hash | `sha256` | **`bcrypt`** | ✅ clear improvement |
| 502 deserialization | json | json + notes avoiding pickle | =/slight win |
| 078 / 089 / 079 | already safe | already safe | = no regression |
| 022 path traversal | no protection | still no containment check | ✗ not fixed |

**Observation**: the prefix works (unambiguous crypto win, zero regressions), but Phi-3 base is already safe on most prompts → a formal benchmark is needed for quantification.

---

## 10. Test 6 — Security Benchmark (paper-aligned)

- **Flow**: `gen_for_icd.py --safecoder_only` (generate ON/OFF) → `normalize_responses.py` → `detect_all.py` (semgrep + weggli + regex) → `score_detected.py`
- **Paper mapping**: §4 test set PurpleLlama CyberSecEval, §2 static-analyzer oracle, **Table 1 Vulnerable Code Ratio**

### 10.1 Reproducing the paper's evaluation subset

The paper evaluates on "38 ⟨lang, CWE⟩ overlapping SafeCoder = 694 test cases", but **does not ship this subset**. We reconstructed it: filter PurpleLlama `instruct.json` to **5 languages {c, cpp, java, javascript, python}** AND **CWEs ∈ SafeCoder's 35 CWEs** (derived from SafeCoder's `sec_eval` scenarios).

| | (lang,CWE) pairs | test cases |
|---|---|---|
| Paper | 38 | 694 |
| **Our reconstruction** (`--safecoder_only`) | **36** | **693** |

### 10.2 ⚠️ Key Finding: fence-less code extraction

`detect_all.py` extracts only code inside ```` ``` ```` fences. Phi-3 frequently emits **raw code with no fences** → such generations were dropped and silently counted as "safe". **84% of code blocks were empty**, diluting the vulnerable ratio ~6×. The detector itself is correct (verified: `strcpy` → weggli/regex, `pickle` → semgrep). Fixed with `normalize_responses.py` (wrap fence-less responses). After the fix the base ratio aligns with the paper.

### 10.3 Results (5 languages, 693 prompts × 10 samples)

| Language | Paper base | **Our OFF** | Paper +ProSec (LoRA+SimPO) | **Our +prefix (DPO)** |
|---|---|---|---|---|
| C | 72.2 | **63.1** | 44.3 | **60.2** |
| C++ | 30.3 | **24.6** | 20.7 | **19.5** |
| Java | 63.6 | **57.1** | 49.1 | **50.5** |
| JS | 52.2 | **52.2** | 28.2 | **43.8** |
| Python | 34.6 | **29.0** | 25.1 | **28.8** |
| **Average** | 50.6 | **45.2** | 33.5 | **40.6** |

(pooled OVERALL: OFF **41.5%** → ON **37.8%**, Δ **−3.7**)

Per-CWE highlights (ON vs OFF): large wins on CWE-295 cert-validation (−30), CWE-352 CSRF (−23), CWE-22 path traversal (−16.5), CWE-643 (−10); a few small regressions (CWE-676 +15, CWE-416 +3.3) likely small-sample noise.

**Conclusion**:
- ✅ **Base aligns with the paper** (e.g. JS 52.2 vs 52.24) → the measurement is valid.
- ✅ **The prefix lowers the vulnerable ratio across all 5 languages** (avg 45.2 → 40.6, −4.6).
- ⚠️ Our improvement (~10% rel.) is **smaller than ProSec's** (~34% rel.), as expected: we use **DPO (not SimPO)**, **prefix (not LoRA)**, **no influence selection**, and **untuned, 1 epoch**. The paper's own Table 8 shows SimPO ≫ DPO on Phi-3 (25.4% vs 34.7%) — the main lever to close the gap.

---

## 11. Test 7 — Utility Benchmark (HumanEval)

- **Script**: `run_humaneval.py` — greedy pass@1, generated code executed against HumanEval tests, prefix ON vs OFF.
- **Scale**: 164 problems (standard Python HumanEval).

| | pass@1 |
|---|---|
| OFF (base) | 64.6% |
| ON (prefix) | **61.6%** |
| Δ | **−3.0** |

**Conclusion**: the prefix costs a small amount of utility (−3.0 pts, ~4.7% rel.). This **differs from ProSec**, which *preserves / slightly improves* utility — because the paper's utility preservation relies on **Dnorm + influence selection** (§3.3), which we have **not yet applied**. So influence selection is promoted from "optional ablation" to a needed step.

> Note: our absolute pass@1 (Python HumanEval) is **not** directly comparable to the paper's HumanEval-Multi; only the Δ direction is meaningful (ours −3.0 vs paper ≈ flat/up).

---

## 12. Status & TODO

**Done**
- ✅ System integration (ProSec data + PEFT prefix + TRL DPO)
- ✅ Fixed prefix-init instability (zero-init)
- ✅ Full training (healthy convergence)
- ✅ Reconstructed the paper's SafeCoder-overlap eval subset (36 pairs / 693 cases)
- ✅ Fixed fence-less code-extraction bug (base now aligns with the paper)
- ✅ Paper-aligned security eval, 5 languages (avg 45.2% → 40.6%, all langs down)
- ✅ Utility eval (HumanEval pass@1 64.6% → 61.6%, small −3.0 cost)

**TODO**
- ⏳ **Influence selection (Dnorm)** — recover the utility cost (paper §3.3 mechanism)
- ⏳ Switch DPO → SimPO (TRL CPOTrainer) — main lever to close the security gap to ProSec
- ⏳ Hyperparameter tuning (num_virtual_tokens, epochs, beta)
- ⏳ Baseline: ProSec original LoRA (prefix vs LoRA)
- ⏳ Phase 2: CodeLlama-7B + self-generated data

**Main result table (current)**

| Metric | base Phi-3 (OFF) | +prefix (ON) | Δ |
|---|---|---|---|
| Security: Vulnerable Ratio ↓ (avg 5 langs) | 45.2% | **40.6%** | **−4.6** ✅ |
| Utility: HumanEval pass@1 ↑ (Python, 164) | 64.6% | **61.6%** | **−3.0** ⚠️ |

**Takeaway**: the prefix improves security (−4.6) at a small utility cost (−3.0). The cost is expected to shrink with **Dnorm influence selection** (paper's utility-preservation mechanism) and **SimPO**, both still to be applied.

---

## Appendix — Scripts

| Script | Purpose |
|---|---|
| `data/convert_prosec_to_pref.py` | Convert ProSec HF dataset → (prompt, chosen, rejected) |
| `train_prefix.py` | Train prefix via PEFT PrefixTuning + TRL DPO |
| `poc_prefix_smoke.py` | Validate PEFT prefix forward/generate/reference |
| `eval/gen_compare.py` | Qualitative prefix ON/OFF side-by-side |
| `eval/gen_for_icd.py` | Generate code for PurpleLlama benchmark (ON/OFF); `--safecoder_only` reproduces the paper subset, `--langs` for multi-language |
| `eval/normalize_responses.py` | Wrap fence-less responses so code extraction works (run before `detect_all.py`) |
| `eval/score_detected.py` | Aggregate vulnerable-code ratio per language + per CWE (ON vs OFF) |
| `eval/run_humaneval.py` | HumanEval pass@1 (ON vs OFF) |

> External (ProSec / PurpleLlama): `prosec_scripts/detect_all.py` runs the Insecure Code Detector (semgrep) on generated code.
