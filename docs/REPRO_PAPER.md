# 論文復現：ProSec 原始設定（LoRA + SimPO + §3.3 influence selection）

分支：`repro/prosec-lora-simpo` ｜ 入口：`scripts/repro_paper_lora_simpo.sh`

這份文件的用途是把「論文寫死的」「論文沒寫、我們自己決定的」「結構上做不到的」
三類分開，之後回報數字時才知道哪些差異可以歸因、哪些不行。

---

## 1. 四階段流程

```
train_pref.jsonl (45,785)
   │  ① train_prefix.py --peft_method lora --subset dsec --max_steps 1000 --save_steps 100
   ▼
warmup-dsec/checkpoint-{100..1000}          10 個 checkpoint（論文 §5「warm-up training」）
   │  ② data/collect_training_dynamics.py   對每筆算 r(x,y,θ_t) = (1/|y|)·log p
   ▼
dynamics/checkpoint-*.jsonl
   │  ③ data/select_dnorm.py                Kendall τ → 每個 D_sec 取 top-2 → 丟最低 20%
   ▼
train_pref_selected.jsonl
   │  ④ train_prefix.py --peft_method lora --max_steps 1500
   ▼
lora-simpo/  →  既有 eval 管線（不需修改）
```

評測腳本走 `PeftModel.from_pretrained` + `disable_adapter()`，
對 LoRA 與 Prefix 完全通用，**這次不用動 `eval/`**。

---

## 2. 直接照論文抄的設定

| 項目 | 值 | 出處 |
|---|---|---|
| 目標函數 | SimPO（TRL `CPOTrainer`, `cpo_alpha=0`）| §4 Eq.(8) |
| lr | **5e-6** | Appendix C Table 6 |
| β / γ | **1.5 / 0.5** | Appendix C Table 6 |
| steps | **1500**（Phi-3-mini）| Appendix C Table 6 |
| total batch size | **64** | Appendix C |
| PEFT | **LoRA r=8, α=16** | Appendix C |
| warm-up | D_sec、**1000 steps**、每 **100** 步存 checkpoint | §4「Warm-up Training for Influence Score」|
| 影響力量測 | r(x,y,θ) = (1/\|y\|)·log π_θ(y\|x)，長度歸一化 | Eq.(5)(6) |
| f / g | f = r(x_n, y_n)、g = −r(x_v, y_f) | Eq.(5)(6)、Table 3 第一列 |
| 相關係數 | Kendall Tau | Eq.(7) |
| D_norm 選擇 | 每個 D_sec 取 **top-2**，再丟掉全域最低 **20%** | §5 |
| system prompt | `You are helpful coding assistant.` | ProSec `influence_score/data_utils_refactored.py::encode_oneturn_phi3` |

### f/g/corr 的設定是怎麼確定的

論文本文沒寫死 corr 的種類與取向，`sample_refactored.py` 的**預設值**
（spearman / `onof` / `ofif`）也不是論文最終用的那組。
但作者上傳的最終資料集叫

```
prosecalign/top2-cds-0.8-kendall-on-neg_if-corr-max-2
```

而 `sample_refactored.py` 的命名規則是
`{prefix}-cds-{downsample_ratio}-{corr_method}-{obj0}-{obj1}-corr-{flag}-{top_n}`，
反推得到：**kendall / f=`on` / g=`neg_if` / flag=`max` / top_n=2 / ratio=0.8**。
`on`+`neg_if` 這條分支算的正是 `corr(r(x_n,y_n), −r(x_v,y_f))`，
與論文 Eq.(5)(6) 及 Table 3 第一列（Vul 23.02 最安全的那組）完全吻合。
`data/select_dnorm.py` 的預設值即為此組。

---

## 3. 論文沒寫、我們自己決定的（改動時要一起改這張表）

| 項目 | 我們的選擇 | 理由 / 風險 |
|---|---|---|
| LoRA target modules | Phi-3 的 all-linear：`qkv_proj, o_proj, gate_up_proj, down_proj` | 論文只寫 r/α。他們用的 `SeCAlign-llama-factory` 未公開；LLaMA-Factory 預設是 all-linear。PEFT 的內建 mapping 沒有 `phi3` 條目，不指定會直接報錯 |
| lr schedule / warmup | cosine、warmup_ratio 0.1 | SimPO 官方 recipe。論文未提 |
| max_grad_norm | 1.0 | LLaMA-Factory 預設。**注意主線 prefix 實驗用的是 0.3**，兩者不可混談 |
| max_length | 2048（prompt 1024）| ProSec `training_dynamics_refactored.py` 的 `MAX_LENGTH = 2048`。主線用的 1024 會截斷 12% 的樣本（S1-loc） |
| batch 拆法 | 4 × grad_accum 16 = 64 | 論文用 2×A100-40G；我們單卡，只要乘積是 64 就等價 |
| 截斷方向 | 超長時砍 prompt 的頭，不砍 response 的尾 | response 被截斷會同時扭曲長度歸一化的分子與分母 |
| logits shift | 標準 causal shift（位置 t 預測 t+1）| ProSec `scores.py::get_sequence_logps` **沒有做這個 shift**，直接拿 `logits[:,i,:]` 去 gather `labels[:,i]`。那是錯的；我們照正確定義做。影響：他們算出來的 r 有一格偏移，但因為對所有樣本一致，排序相關係數受到的影響有限 |
| attention_mask | 有傳 | ProSec 的 `checkpoint_stats` 只傳 `input_ids` 和 `labels`，沒傳 mask，left-padding 的 pad token 會被 attend 到 |

---

## 4. ⚠️ 結構上做不到的：D_norm 候選池已經被壓平

**這是這次調查最重要的發現，會直接影響 §3.3 復現的意義。**

論文的 selection 是「每個 D_sec 從**很多個** D_norm 候選裡挑 top-2」。
候選池由 `influence_score/data_utils.py::prepare_selection_dataset_new` 產生：
每個 D_sec 條目 × 該 (正常指令, lang, cwe) 底下**所有**可解析的正常碼，做笛卡兒積
（再隨機下採樣到 35%）。

我們手上的 `prosecalign/prosec-mixed-phi3mini-4k-inst` **不是那個候選池**。實測結構：

| 統計 | 值 |
|---|---|
| D_sec | 27,400（`fixed_code` 全部相異） |
| D_norm | 18,385 |
| 有 D_norm 夥伴的 D_sec | **11,306 / 27,400** |
| 每個 D_sec 的 D_norm 候選數 | **1 個：4,227 組；2 個：7,079 組；3 個以上：0 組** |

三項證據都指向「這份資料已經套過 §3.3 的選擇」：

1. **候選數硬性卡在 2** —— 這正是 `top_n=2` 的痕跡，沒有任何群組有 3 個以上。
2. **`mix_and_upload_original_w_fixed_batch.py` 產不出這個形狀** —— 那支腳本對每個
   D_sec 只用 `np.random.choice` 取**恰好 1 個**候選，且 `original_code`
   （= 該筆的 `fixed_code`）應全部相異。實測是 11,306 個相異值配 18,385 列，矛盾。
3. **列的順序是「D_sec 全部在前、D_norm 全部在後」** —— `mix_and_upload` 結尾會
   `all_ds.shuffle(seed=42)`，而 `sample_refactored.py` 結尾是
   `concatenate_datasets([ifv_ds, onf_ds])` **不洗牌**。實測第一筆 benign=True
   出現在 index 27,400，正好是 D_sec 的總數。

**推論：`prosec-mixed-phi3mini-4k-inst` 雖然叫 "mixed"，實際是 selection 腳本的輸出。**

因此在這份資料上：

- **top-2 那一層是 no-op**（沒有任何群組的候選數 > 2）。`select_dnorm.py` 偵測到
  這個情況會印出警告。
- **只有「丟掉全域最低 20%」那一層真的會動**：18,385 → 14,708，
  最終 27,400 + 14,708 = **42,108**（D_norm / D_sec = 0.54）。

### 這件事推翻了 REPORT.md §4.3 的一個說法

`REPORT.md` §4.3 主張我們訓練在「未經 §3.3 篩選的候選池」上，並把 utility 掉幅
歸因於此。就 D_norm 而言這個歸因**站不住腳**——選擇很可能已經做過了。
另外論文的「約 10k / 7× SafeCoder」對得上的是 **D_sec 的相異指令數 11,939**，
不是偏好對數 45,785。這條歸因需要重寫。

### 要真的復現 top-2，只有一條路

重建候選池，需要兩份東西：

1. `prosecalign/secalign-haiku-new-fix-new-dsec-phi3m-all-pairs`（D_sec all-pairs）
   → HF 上是 **private（401）**，拿不到。
2. `infer-ret-ori-task-haiku.phi3m-inst-filtered.jsonl`（Phi-3 對正常指令的多次取樣）
   → 未釋出，但**可以自己重生**：拿
   `prosecalign/haiku-vul-inducing-instructions-clustered` 的 `original_prompt`
   餵 Phi-3-mini 多次取樣即可（純生成，不需要靜態分析器）。

也就是說：(2) 我們做得到，(1) 做不到，除非同時重生 D_sec 的 all-pairs
（那等於重跑 ProSec Step 2–4 的整條合成管線）。

### 由此得到的操作建議

如果這份資料已經是 selection 的輸出，那麼**再跑一次我們自己的 selection 不是復現，
而是第二次篩選**（top-2 那層是 no-op，但 0.8 那層會變成第二次砍 20%，
最壞情況是砍掉論文原本要保留的樣本）。無法從資料本身判斷 `cds-0.8` 有沒有被套過。

因此第 1 節的四階段有兩種跑法：

| | 做法 | 適用時機 |
|---|---|---|
| **A（建議先跑）** | 跳過階段 1–3，直接用 45,785 筆跑階段 4 | 把這份資料當成論文的最終訓練集。這其實比再篩一次**更接近**論文 |
| **B** | 跑完整四階段 | 想拿到自己的 influence score 做 ablation（E3「只有 D_sec」、不同 D_norm 比例）時 |

A 的成本只有階段 4（1500 steps）。B 要多付「暖身 1000 steps + 10 個 checkpoint ×
45,785 筆 × 2 側的前向」，是數小時到十小時等級的額外算力。

跑法 A：`STAGE=4 FINAL_TRAIN_FILE=data/train_pref.jsonl bash scripts/repro_paper_lora_simpo.sh`

---

## 5. 已知不可比的地方

- 我們的 base OFF 數字與論文 Table 1 不同（45.21 vs 50.57），主因是評測子集
  重建為 36 組 / 693 題（論文 38 組 / 694 題），以及 `eval/gen_for_icd.py`
  取樣未設 seed。**只比 Δ，不比絕對值。**
- utility 我們跑 Python HumanEval + MultiPL-E，論文跑 HumanEval-Multi + MXEval，
  絕對值不可對照。
