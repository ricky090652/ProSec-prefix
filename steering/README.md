# steering/ —— DuoSteer 移植到 Phi-3 + ProSec

來源：Yan & Yao, *Interpreting and Steering for Safe and Correct Code Generation*
（DuoSteer，`../DuoSteer-Safe-Correct-Code-Gen`）。這裡只移植 Stage 2（定位）與
之後的 Stage 3（steering）；資料用 ProSec 的 D_sec，評測沿用本 repo 的 ICD + HumanEval
+ MultiPL-E，不用 DuoSteer 的 CodeQL / GPT-4.1。

## 目前的問題

**用全部 D_sec 合併起來的 safe − vuln 方向，在 Phi-3 上有沒有訊號？**
（ProSec 的全域設定：推論時不知道 CWE，一個向量要涵蓋所有 CWE 與語言。）

## 在 server 上跑

```bash
# 0. 產生 pairs（D_sec → intra / held-out，依指令切分）
python steering/build_pairs.py --train_file data/train_pref.jsonl \
    --out_dir data/steering --heldout_ratio 0.05 --max_pairs 4000

# 1. 抽表徵（layer + head，response_mean；約 8,000 次 forward）
python steering/extract_representations.py \
    --input_file data/steering/intra.jsonl --cwe_id all --bf16

REP=data/representations/microsoft_Phi-3-mini-4k-instruct/intra/all/response_mean

# 2. layer probe（32 個，幾分鐘）
python steering/train_probe.py --rep_dir $REP --mode layer

# 3. head probe（1,024 個 × 200 epochs，預估數小時，可以背景跑）
python steering/train_probe.py --rep_dir $REP --mode head

# 4. causal head knockout（probe 前 256 名 × val split 的 pairs）
nohup python steering/head_causal_analysis.py --rep_dir $REP --bf16 > causal.log 2>&1 &
```

### Steering（§3.4，Table 2 的 LayerMD / ProbeMD / CausalMD）

```bash
IJ=$ICD/CybersecurityBenchmarks/datasets/instruct/instruct.json

# 5. steering 向量（MD + PD、layer + head；用 train split，與 knockout 的 val split 不重疊）
python steering/extract_steering_vector.py --rep_dir $REP

# 6. 在 held-out D_sec 上選設定（論文 Appendix I）：502 有 118 條指令
SEL=outputs/steer/cwe-502-heldout
python steering/steer_generate.py --pairs_file data/steering/heldout.jsonl --cwe 502 --num_gen 3 \
    --setting baseline head --head_results $REP/causal/head_causal_results.json \
    --vector_dir $REP/vectors --top_k_list 32 --alpha_list 1 2 3 5 10 --out_dir $SEL
ICD=$ICD bash steering/score_steer.sh $SEL

# 7. 選定的設定在評測集（693 題子集裡的 502，77 題 × 10 樣本）上跑；OFF 直接用既有的
python steering/steer_generate.py --instruct_json $IJ --safecoder_only --cwe 502 \
    --setting head --head_results $REP/causal/head_causal_results.json \
    --vector_dir $REP/vectors --top_k_list <k> --alpha_list <α> --out_dir outputs/steer/cwe-502
ICD=$ICD bash steering/score_steer.sh outputs/steer/cwe-502 outputs/full_shared.off.norm.jsonl.detected.jsonl
```

功能性（HumanEval + MultiPL-E，與 LoRA 各臂同一支程式；ON = steered、OFF = 原版）：

```bash
STEER="--steer_setting head --steer_vector_dir $REP/vectors \
  --steer_head_results $REP/causal/head_causal_results.json --steer_top_k 32 --steer_alpha 10"
python eval/run_humaneval.py $STEER --max_new_tokens 2048 --skip_off --out outputs/humaneval_<name>.json
python eval/run_multipl_e.py $STEER --langs js,cpp --max_new_tokens 2048 --skip_off --out outputs/multipl_e_<name>.json
```

`--skip_off` 省掉 OFF：greedy 解碼下 base 不變（HumanEval 70.73、js 59.63、cpp 46.58）。
`--adapter` 與 `--steer_*` 可以同時給（ON = adapter + steering）。hook 在 `steering/hooks.py`。

- ProbeMD：把 `--head_results` 換成 `$REP/probes/plots/head_accuracy_results.json`。
- LayerMD：`--setting layer --layer_list 26`（最佳 probe 層）。
- PD 版本：加 `--method probe`。
- 生成協定與 `eval/gen_for_icd.py` 相同（抽樣、逐題 seed），**baseline 與既有的 OFF 逐字相同**
  （tiny Phi-3 驗證），所以評測集上不用重跑 baseline，steered 與 OFF 是逐題配對的。
- 中斷後同一行指令重跑會接著做。

結果在 `$REP/probes/plots/`：`layer_accuracy_results.json`、`head_accuracy_results.json`、
`best_per_layer.png`、`head_heatmap.png`；knockout 在 `$REP/causal/`：
`head_causal_results.json`、`head_causal_delta.png`、`head_causal_delta_vs_base.png`。
knockout 每做完一個 head 就存 checkpoint，中斷後同一行指令重跑會接著做。

**knockout 不用 cross pairs**：DuoSteer 做 cross pairs 只是因為 p^b 底下漏洞碼太少
（§3.2），knockout 時兩邊都在 p^b 底下打分數（Eq. 1）。D_sec 每個 CWE 有上千筆
同一指令下的 safe/vuln，直接用；pairs 取 probe 的 val split（§3.3「validation pairs」），
與算向量的 train split 不重疊。

## 怎麼讀（對照論文 §4）

| 看什麼 | 論文（Llama，per-CWE） |
|---|---|
| layer probe 最佳準確率 | 71–87%，最佳層多在 ℓ∈[9,11] |
| head probe ≥0.65 的數量 | CWE-022：0 個；CWE-295：960 個 |

合併向量若 layer probe 接近 0.5，代表合併起來沒有可分的方向，改走 per-CWE
（`--cwe_id cwe-338` 等；只有 338 / 502 / 78 / 22 的資料量夠）。

## 與 DuoSteer 原版的差異

- prompt 用 ProSec 指令本身（誘發漏洞的指令），不套 DuoSteer 的 benign 模板。
- 斷詞對齊本 repo：prompt 同 `eval/gen_for_icd.py`，response 同 TRL `tokenize_row`；
  response mask 只蓋程式碼，不含 template 結尾的 `<|end|><|endoftext|>`。
- probe 依指令（`src_id`）切分；原版在沒有 `src_id` 的資料上會退回按 pair 切。
- `--token_agg` 預設 `response_mean`、probe `--epochs` 預設 200（皆為論文設定）。

## 已驗證（本機，tiny Phi-3：真 config、2 層、隨機權重、真 tokenizer）

- batch（有 padding）vs 單筆 vs 手算 response 平均：差異 ≤ 5e-7（float32 誤差）
- head 抓取 shape (b, L, 32, 96)；1,024 個 head 檔與 metadata 格式同原版
- 隨機權重的 probe 準確率 ≈ 0.5（沒有訊號時學不到東西，切分沒有洩漏）

## ⚠️ 資料分布

D_sec 27,400 筆的 CWE 極度不均：338 占 47%（12,846），502 / 78 / 22 各 3~5 千，
295 只有 110、377 只有 38。合併向量會被 CWE-338 主導。
