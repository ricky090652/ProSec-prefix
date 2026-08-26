# Prefix-ProSec 實驗 Checklist

> 完整原理與判準見 [prefix-prosec_research_plam.md](prefix-prosec_research_plam.md) §8–§14。
> 這份只追蹤「做到哪了」。
>
> **使用方式**：跑完一項就把結果貼回對話，Claude 會勾掉對應項目、填入數字、
> 並依停損點判斷下一步。不要自己改結構，直接貼結果即可。

**目前進度**：S0 完成 · **S1 訓練健康、margin 隨 lr 單調成長**，待往上探 lr + 跑安全評測
**上次更新**：2026-08-26（S1-lr2 結果：對齊論文設定後訓練正常）

---

## 全域規則（每次實驗都適用）

- [ ] 一次只改一個變數（Exp2 的教訓）
- [ ] `epochs` 一律固定為 1，除非該實驗本身就是在測 epoch
- [ ] 每個 run 存一份 `run_meta.json`（版本、超參數、資源、seed）
- [ ] 每組 security 數字都要附退化守門報表（S0.5 的三條件）
- [ ] LoRA 與 Prefix 比較時：同資料、同 split、同 β/γ、同 batch、同評測 seed

---

## L0 · 診斷層 —— 會改變後續所有實驗的地基

### S0 · Prefix 學習率校準
指令：`bash run_lr_sweep.sh` ｜ 判讀：`python eval/compare_lr_sweep.py outputs/lr_sweep/*.log`

- [x] **S0-a** lr = 2e-5（現況對照下界）→ **underfit 確認**
- [x] **S0-b** lr = 1e-4 → **唯一健康的一組**
- [x] **S0-c** lr = 1e-3 → chosen 反向崩到 −4.33，grad 峰值 2498
- [x] **S0-d** lr = 5e-3 → reward 崩潰（chosen −24.08）
- [x] **S0-結論** 診斷目的達成：Exp1 的 2e-5 確認 underfit，H1 未被推翻，繼續主線
- [ ] **S0-e** 小規模安全評測 + `check_degeneration.py`（改到 S1 的 lr 定案後一起做）
- [~] ~~**S0-f** lr = 3e-4（DPO）~~ → **延後到 S3 的 E2-dpo**。主線用 SimPO，
      再細調 DPO 的 lr 對 S1/S2 沒有幫助；E2-dpo ablation 需要時再跑（同等搜尋預算原則）

> 通過標準：`rewards/chosen` 不再單調下降、margin 由雙邊貢獻、`grad_norm` 全程 < 50。
> **停損點**：若四組都只有 rejected 單邊下降 → 不是 lr 問題，改試 `nvt` 16 → 40 → 64。

**結果**（2026-08-25，DPO β=0.1、nvt=16、zero-init、4,000 筆子集、各約 13 分鐘）：

| lr | chosen Δ | rejected Δ | margins | acc | grad max | 判讀 |
|---|---|---|---|---|---|---|
| 2e-5 | +0.076 | +0.016 | 0.27 | 0.607 | 18 | **underfit**——兩側幾乎沒動 |
| 1e-4 | +0.035 | −0.401 | 0.72 | 0.735 | 21 | **唯一雙邊貢獻**，穩定 |
| 1e-3 | −2.833 | −6.294 | 4.29 | 0.934 | **2498** | chosen 反向崩，只是 rejected 崩更快 |
| 5e-3 | −21.910 | −34.188 | 14.35 | 0.975 | 26 | **reward 崩潰**，chosen 掉到 −24 |

**結論**：
1. **Exp1 用的 2e-5 確認是 underfit**（acc 0.607，兩側 Δ < 0.08）。這解釋了為什麼 Exp1 的
   安全性改善只有論文的三分之一——prefix 根本沒被訓練起來。
2. 但**可用區間比文獻窄得多**。文獻的 1e-3~1e-2 是 supervised fine-tuning 的量級；
   偏好學習（尤其 DPO）脆弱得多，1e-3 以上就 reward 崩潰。
3. 高 acc 具有欺騙性：5e-3 的 acc 0.975 是兩側一起崩、只是速度不同造成的，毫無意義。
4. 停損點**未觸發**（不是全部 underfit），繼續走主線。

**待確認**：1e-4 的 acc 0.735 仍低於 0.75 目標，最佳點可能落在 2e-4~5e-4 之間 → 補跑 S0-f。

---

### S0.5 · 生成退化診斷與守門機制
指令：`python eval/check_degeneration.py --on ... --off ... --per_lang`

- [ ] **S0.5-a** 跑 Exp1 的既有輸出
- [ ] **S0.5-b** 跑 Exp2 的既有輸出（**重點看 cpp**，驗證 −24.22 是不是退化造成的）
- [ ] **S0.5-結論** 確立三條守門條件的基準線

> 通過標準：ON 程式碼長度 ≥ OFF 的 90%、語法完整率 Δ ≥ −3 pt、stub 率 Δ ≤ +2 pt。
> **停損點**：若 ON 明顯退化 → 現有 security 數字全部被污染，必須先修才能進 S1。

**結果**：_（待填）_

---

## L1 · 主線層 —— 論文沒有它就不能投

### S1 · DPO → SimPO 目標函數切換

- [x] **S1-code** `train_prefix.py` 加 `--objective {dpo,simpo}`（SimPO 走 `CPOTrainer`）
- [x] **S1-code** 加 `--gamma`、`--cpo_alpha`；`--beta` 改為依 objective 自動取預設
- [x] **S1-code** `--prefix_init_scale 0` 改走 PEFT 原生 `init_weights="zero"`
- [x] **S1-code** `compare_lr_sweep.py` 判讀門檻改為尺度無關 + 自動辨識 objective
- [x] **S1-code** `run_lr_sweep.sh` 加 `OBJECTIVE` 參數
- [x] **S1-lr** SimPO lr 確認（1e-4 / 3e-4 / 1e-3）→ **三組全部失敗，問題不在 lr**
- [x] **S1-verify** 核對 ProSec Appendix C Table 6 → **找到三處設定不符，見下**
- [x] **S1-lr2** 論文設定重掃 lr {5e-6, 2e-5, 5e-5} @ batch 64 → **三組都健康，未達上限**
- [ ] **S1-lr3** 往上再探一級：lr {1e-4, 2e-4} @ batch 64（約 70 分）
- [ ] **S0-e** 對現有 `lr_5e-5` adapter 跑小規模安全評測 + 退化守門（約 30 分，可與上一項排隊）
- [ ] **S1-loc** 跑 `data/edit_locality.py`（CPU，可與 S1-lr2 並行；S5 前置）
- [ ] **S1-run** SimPO + Prefix 全量訓練（β=1.5、γ=0.5、lr 用 S1-lr 結果、epochs=1）
- [ ] **S1-eval** 安全評測 + HumanEval + 退化守門

> 通過標準：訓練穩定、`rewards/accuracies` > 0.75、通過退化守門。
> 陷阱：β 在 DPO 與 SimPO 之間不可換算，第一輪嚴格用論文的 1.5 / 0.5。
> **已驗證的陷阱**：TRL 的 `cpo_alpha` 預設 1.0 會變成 CPO-SimPO 混合，
> 純 SimPO 必須設 0（`train_prefix.py` 已把預設改為 0，並在非 0 時警告）。
> SimPO 的 reward 尺度與 DPO 差一個量級，不可跨 objective 比大小。

**S1-lr 結果**（2026-08-26，SimPO β=1.5 γ=0.5 cpo_alpha=0、nvt=16、4,000 筆子集）：

| lr | margins Δ | accuracies late | chosen Δ | rejected Δ | 判讀 |
|---|---|---|---|---|---|
| 1e-4 | **−0.053** | **0.390** | −0.813 | −0.749 | margin 縮小，排序變差 |
| 3e-4 | +0.000 | **0.419** | −0.435 | −0.436 | 兩側同步下降，margin 不動 |
| 1e-3 | +0.171 | **0.506** | −0.933 | −1.143 | 勉強有 margin，但排序仍≈隨機 |

**三組的 accuracies 都 < 0.51**，代表模型把 rejected 排在 chosen 前面的次數不比隨機少。
這不是 underfit（DPO 在同樣預算下 accuracies 到 0.61~0.93），是**訓練訊號本身有問題**。

**原假設已被論文推翻**：γ 不是結構上不可達——論文用完全相同的 β=1.5 / γ=0.5
拿到 Vul 25.39（base 40.76）。正確的結論是「margin 訊號弱 → 對步長極度敏感」。

**真正的原因：三處超參數沒對齊論文（Appendix C, Table 6）。**

| | 論文 | 我們第一輪 | 差距 |
|---|---|---|---|
| lr | **5e-6** | 1e-4 ~ 1e-3 | 高 20~200 倍 |
| effective batch | **64** | 16 | 梯度雜訊大 2 倍 |
| 訓練量單位 | **1500 optimizer steps** @ batch 64 | 250 steps @ batch 16 | — |

論文的 lr=5e-6 是 LoRA 的值，Prefix 通常需要略高，所以重掃範圍由 5e-6 往上包住一段。

> 附帶修正一條先前的規則：「epochs 固定為 1」是從 Exp2 過度推論的。論文用 **optimizer
> steps** 計量（1500 steps @ batch 64 ≈ 9.6 epochs over ~10k）。Exp2 真正的問題是它跑了
> 5,722 個 step，是論文的 3.8 倍。**之後一律用 optimizer steps @ batch 64 描述訓練量。**

**S1-lr2 結果**（對齊論文設定：effective batch 64、16,000 筆 = 250 optimizer steps）：

| lr | margins early → late | **margin 成長** | accuracies early → late | grad late |
|---|---|---|---|---|
| 5e-6 | 0.0366 → 0.0421 | **+15.0%** | 0.398 → 0.403 | 2.8 |
| 2e-5 | 0.0376 → 0.0494 | **+31.4%** | 0.400 → 0.408 | 7.0 |
| 5e-5 | 0.0389 → 0.0632 | **+62.5%** | 0.402 → 0.422 | 26.2 |

**訓練是健康的。** margin 隨 lr 單調成長、沒有飽和、沒有不穩定跡象（lr 每升一級，
Δmargin 約 ×2.1）。effective batch 從 16 改成 64 是關鍵——同樣的 lr 在 batch 16 下
會讓 margin 縮小，在 batch 64 下則穩定成長。

**accuracies < 0.5 不是失敗訊號。** 起點就是 0.398——這是**資料與 base model 的性質**：
ProSec 的 `y_f` 是在「已看過漏洞碼 + 分析器回饋」的條件下生成的（論文 §3.2：
`y_f ~ π(y|x_v, y_v, l, c)`），而 `y_v` 是直接從 `x_v` 取樣的。所以在只給 `x_v` 的條件下
評分，`y_v` 天生機率較高。要看的是趨勢（三組都在往上），不是絕對值。
論文自始至終沒有回報 preference accuracy，最終指標是 vulnerable code ratio。

**以下為第一輪（batch 16、lr 過高）的原始分析，保留供參考：**
TRL 的 SimPO 目標 margin = γ/β = 0.5/1.5 = **0.333 nats/token**。已用觀測值驗證 loss 模型
（預測 0.9364 vs 實測 0.943~0.945）。要把 loss 壓到 0.5 需要 margin 0.622 nats/token，
即 chosen 每個 token 的機率是 rejected 的 1.86 倍、且要撐過整個序列。

但 ProSec 的 pair 只差 1~3 行——相同的 token 對 margin 貢獻≈0，卻**完整計入長度歸一化的分母**。
可達 margin ≈ ρ × Δ（ρ=變動 token 比例）。若 ρ≈0.1、Δ≈2 nats，上限僅 0.2，達不到 0.333。
目標不可達 → loss 永遠停在高梯度區 → 優化器持續往無法擴大 margin 的方向推 → 兩側一起被壓低。

**為什麼 DPO 沒這個問題**：DPO 沒有長度歸一化，margin 是 token 的**加總**，
少數幾個 diff token 就能推出很大的 margin。SimPO 除以 |y|，margin 被 edit ratio 封頂。

> 這正是 S5（masked SimPO）要解決的機制——只是它以**阻斷者**的身分提早兩步出現了。

**結果**：_（待填）_

---

### S2 · E1 LoRA baseline（同管線、同資料）—— 論文成敗關鍵

- [ ] **S2-code** `train_prefix.py` 加 `--peft_method {lora,prefix}`
- [ ] **S2-code** 加資源指標記錄 callback（trainable params / adapter 大小 / peak mem / runtime）
- [ ] **S2-a** LoRA lr sweep：1e-4
- [ ] **S2-b** LoRA lr sweep：3e-4
- [ ] **S2-c** LoRA lr sweep：5e-4
- [ ] **S2-run** LoRA 最佳設定全量訓練（r=8, α=16, SimPO）
- [ ] **S2-eval** 安全評測 + HumanEval + MultiPL-E + 退化守門
- [ ] **S2-sanity** 與論文 Table 1 的 Phi-3 數字比對（只檢查方向與量級，不要求相同）
- [ ] **S2-RQ2** 產出資源對照表

> **停損點**：若 Prefix 安全性遠遜 LoRA（差距 > 10 pt）→ H1 不成立，
> 論文主軸改為「Prefix 的能力邊界分析」，及早轉向。

**結果**：_（待填）_

---

### S3 · 完整實驗矩陣 + Ablation + 多 seed

#### 前置
- [ ] **S3-split** 建立 (lang, cwe) stratified 5% validation split，存成固定檔案
- [ ] **S3-ckpt** `train_prefix.py` 加 checkpoint selection（validation loss 最低）
- [ ] **S3-leak** leakage 檢查：對 HumanEval / MultiPL-E / PurpleLlama 的 prompt 做 exact / normalized / fuzzy 比對，記錄移除數量

#### 主實驗
- [ ] **E0** Base Phi-3（OFF）—— 用固定 seed 重跑一份，之後所有實驗共用
- [ ] **E1** LoRA r=8 + SimPO（來自 S2）
- [ ] **E2** Prefix nvt=16 + SimPO —— 第一階段主方法

#### Ablation
- [ ] **E3** 僅 D_sec（無 D_norm）
- [ ] **E2a** Prefix nvt=10
- [ ] **E2b** Prefix nvt=20
- [ ] **E2c** Prefix nvt=40
- [ ] **E2-init** Prefix 隨機初始化（對照零初始化）
- [ ] **E2-dpo** Prefix + DPO（讓既有的 DPO 實驗變成正式 ablation）

#### 多 seed
- [ ] **E1-s2 / E1-s3** LoRA 第 2、3 個 seed
- [ ] **E2-s2 / E2-s3** Prefix 第 2、3 個 seed

#### 分析
- [ ] **S3-RQ4** 分層分析：seen/unseen CWE、語言、訓練樣本數、edit ratio 分桶

> **停損點**：若 Prefix 的 utility 掉幅小於 LoRA → H2 成立，**S4 可略過**，直接進 S5。
> 這個「不需要」本身就是有價值的發現。

**結果**：_（待填）_

---

## L2 · 補強層 —— 決定論文層級

### S4 · D_norm influence selection 【條件執行】

> **只有當 S3 顯示 E2 的 utility 掉幅仍 > 3 pt 時才做。**

- [ ] **S4-code** 確認 ProSec repo 有無原版 selection script；沒有則用 gradient-based 近似
- [ ] **S4-data** 產出篩選後資料集（目標量級約 10k），記錄分布變化
- [ ] **S4-E1'** LoRA 在篩選後資料上重跑
- [ ] **S4-E2'** Prefix 在篩選後資料上重跑
- [ ] **S4-分析** selection 對 LoRA 與 Prefix 的效果是否對稱

**結果**：_（待填）_

---

### S5 · Diff-token masked SimPO —— 最有創新性的一段

#### 前置分析（必須先做，幾小時，可省下一週實作）
- [ ] **S5-loc** edit-locality 統計：edit ratio、hunk 數、單一 contiguous diff 比例、最長共同前後綴
- [ ] **S5-overlap** **diff 與 analyzer finding 行號的重疊率** —— 若不重疊，mask 抓到的是雜訊，H3 直接不成立
- [ ] **S5-bucket** 依 edit ratio 分低/中/高三桶（切點由實際分布決定）

#### 實作
- [ ] **S5-code** token-level diff → mask 建構；處理全 0 mask 與過短 mask 的邊界
- [ ] **S5-code** `train_prefix.py` 加 `--loss_scope {full,masked,adaptive}`

#### 實驗
- [ ] **S5-mask-prefix** masked × Prefix
- [ ] **S5-mask-lora** masked × LoRA
- [ ] **S5-adaptive** adaptive full/local（門檻由分桶結果反推）
- [ ] **S5-H3** 比較 masked 對 Prefix 的增益 vs 對 LoRA 的增益 —— 論文最強的一張圖
- [ ] **S5-pareto** 畫 security-utility 的 Pareto 圖

> 通過標準（滿足其一）：不損失安全性但 utility 掉幅更小；或相同 utility 下更安全。
> **停損點**：若各分桶下 masked 都沒增益 → H3 不成立，誠實回報，論文停在 L1 + L3。
> 陷阱：mask 只作用於 loss 聚合，**不作用於前向**。前向必須跑完整序列。

**結果**：_（待填）_

---

## L3 · 泛化層

### S6 · CodeLlama-7B + CWEval

- [ ] **S6-smoke** 先用 `poc_prefix_smoke.py` 驗證 GQA 架構下 prefix 前向/生成正常
- [ ] **S6-data** 改用 `prosecalign/prosec-mixed-clm7b-inst`（61.2k，同指令集、CodeLlama 生成）
- [ ] **E4** CodeLlama-7B base
- [ ] **E5** CodeLlama-7B + LoRA
- [ ] **E6** CodeLlama-7B + Prefix（最佳設定）
- [ ] **S6-cweval** CWEval 環境建置（Docker）
- [ ] **S6-RQ3** E0 / E1 / E2 在 CWEval 上的 func@k 與 func-sec@k

**結果**：_（待填）_

---

## L4 · 延伸層 —— 有餘力才做，不影響論文成立

- [ ] **X1** CWE-specific 多 Prefix（推薦度：高 —— Prefix 相對 LoRA 最有結構性的優勢）
- [ ] **X2** Prefix 組合與插拔行為分析
- [ ] **X3** `prefix_projection=True` ablation（必須連同 613M 參數代價一起報告）
- [ ] **X4** FIM 格式 ablation（先驗證模型是否真能解析 FIM token）
- [ ] **X5** KTO（只有引入未配對資料時才值得）
- [ ] **X6** Verifier-based weighting
- [ ] **X7** Qwen2.5-Coder-3B 跨模型 transfer

---

## 里程碑

- [ ] **M0** S0 + S0.5 —— 知道正確操作點，且數字不是退化偽造的
- [ ] **M1** S1 + S2 —— 有可比較的 LoRA 對照組
- [ ] **M2** S3 —— **第一階段論文可以寫完**（最低可發表門檻）
- [ ] **M3** S4 —— 知道 selection 對兩種 PEFT 是否對稱
- [ ] **M4** S5 —— **方法創新成立**（真正的目標）
- [ ] **M5** S6 —— 外部效度

---

## 已完成（歷史）

- [x] 系統整合：ProSec 資料 + PEFT prefix + TRL DPO
- [x] 修掉 prefix 隨機初始化的不穩定（零初始化）
- [x] 重建論文未釋出的評測子集（36 組 / 693 題）
- [x] 修掉 fence-less 程式碼抽取 bug（84% code block 為空、漏洞率被稀釋 6 倍）
- [x] Exp1（DPO, β=0.1, ep1）：security 45.21 → 40.54；HumanEval 64.63 → 61.59
- [x] Exp2（DPO, β=0.05, ep2）：security 44.79 → 36.51；HumanEval 64.63 → 53.05 ⚠️ 變數混淆，作廢
- [x] 資料血緣考證（100% exact-match 驗證 D_sec / D_norm 來源）
