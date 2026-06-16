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
| 6 | Security benchmark | `gen_for_icd.py` + `detect_all.py` + `score_detected.py` | §4 PurpleLlama / Table 1 | ✅ smoke effective |
| 7 | Utility benchmark | `run_humaneval.py` | §4 HumanEval / Table 1 | ⏳ pending |

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

## 10. Test 6 — Security Benchmark (preliminary smoke)

- **Flow**: `gen_for_icd.py` (generate ON/OFF) → `detect_all.py` (semgrep detection) → `score_detected.py` (scoring)
- **Paper mapping**: §4 test set PurpleLlama CyberSecEval, §2 static-analyzer oracle, **Table 1 Vulnerable Code Ratio**
- **Scale**: 20 prompts × 5 samples (preliminary smoke)

| CWE | base (OFF) | +prefix (ON) | Δ (↓ better) |
|---|---|---|---|
| CWE-338 (weak PRNG) | 18.3% | 10.0% | **−8.3** |
| other CWEs | 0% | 0% | 0 |
| **Overall** | **11.0%** | **6.0%** | **−5.0** |

**Conclusion**: correct direction, **no CWE got worse**; this is a small smoke run — full 351-prompt run pending.

---

## 11. Status & TODO

**Done**
- ✅ System integration (ProSec data + PEFT prefix + TRL DPO)
- ✅ Fixed prefix-init instability (zero-init)
- ✅ Full training (healthy convergence)
- ✅ Security eval pipeline + preliminary result (11% → 6%)
- ✅ Utility eval script

**TODO**
- ⏳ Full security eval (351 prompts) + full HumanEval → main result table
- ⏳ Baseline: ProSec original LoRA (prefix vs LoRA)
- ⏳ Ablation: influence-selected data
- ⏳ Phase 2: CodeLlama-7B + self-generated data

**Expected main result table**

| | base Phi-3 | +prefix | Goal |
|---|---|---|---|
| Security: Vulnerable Ratio ↓ | X% | Y% | Y < X |
| Utility: HumanEval pass@1 ↑ | P% | Q% | Q ≈ P |

---

## Appendix — Scripts

| Script | Purpose |
|---|---|
| `data/convert_prosec_to_pref.py` | Convert ProSec HF dataset → (prompt, chosen, rejected) |
| `train_prefix.py` | Train prefix via PEFT PrefixTuning + TRL DPO |
| `poc_prefix_smoke.py` | Validate PEFT prefix forward/generate/reference |
| `eval/gen_compare.py` | Qualitative prefix ON/OFF side-by-side |
| `eval/gen_for_icd.py` | Generate code for PurpleLlama benchmark (ON/OFF) |
| `eval/score_detected.py` | Aggregate vulnerable-code ratio (ON vs OFF) |
| `eval/run_humaneval.py` | HumanEval pass@1 (ON vs OFF) |

> External (ProSec / PurpleLlama): `prosec_scripts/detect_all.py` runs the Insecure Code Detector (semgrep) on generated code.
