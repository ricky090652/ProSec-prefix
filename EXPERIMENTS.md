# Prefix-ProSec 實驗 Checklist

> 完整原理與判準見 [prefix-prosec_research_plam.md](prefix-prosec_research_plam.md) §8–§14。
> 這份只追蹤「做到哪了」。
>
> **使用方式**：跑完一項就把結果貼回對話，Claude 會勾掉對應項目、填入數字、
> 並依停損點判斷下一步。不要自己改結構，直接貼結果即可。

**目前進度**：**四臂 utility 完成，結論清楚** —— LoRA qkv **+1.22**（與論文 +1.42 一致）· prefix nvt=64 **−32.93**（擾動代價隨長度暴增）· 待四臂安全性評測
**上次更新**：2026-09-03（P-match utility）

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

### S1 · ~~DPO → SimPO 目標函數切換~~ → **2026-08-28 反轉：主線改回 DPO**

> SimPO 在 prefix（S1-full，step ~450 斷崖）與 LoRA（S2-repro，step ~350 平滑失控）
> **兩次都崩**。機制已定位（見 S2-repro）。主線改用 DPO，理由：
> 論文 Appendix D.3 Table 8 自己驗證過 DPO 在這份資料上有效（Vul 40.76 → 34.65、
> utility 42.78 → 44.20），Table 6 也給了 DPO 超參數；而核心主張是 prefix vs LoRA 的
> **相對**比較，目標函數是控制變因，兩臂一致即可。
> SimPO 的失敗改列為論文的一節，並由 S5（masked SimPO）承接為提出的修正。
>
> **新主線**：`scripts/run_dpo_arms.sh`（兩臂唯一差別是 `--peft_method`）
> - [x] **D-lr-prefix** → **定案 5e-5**（比值 6.30；1e-4 比值掉到 2.62 且 grad 峰值 3722）
> - [x] **D-lr-lora** → **定案 5e-6**（也正是論文 Table 6 的值）
> - [x] **D-run** 兩臂全長 800 steps @ batch 64、β=0.05、不帶 system prompt → 見下方結果
> - [ ] **D-eval** 兩臂全套 693 題 × 10 samples × 5 語言 + 退化守門 + HumanEval/MultiPL-E
>       （100 題×5 的標準誤 2.2pt 分不出兩臂差距，最終對照不可用便宜篩選）
>
> **D-lr 完整對照**（2026-08-28~30，DPO β=0.05、batch 64、16,000 筆 = 250 steps、
> seed 42、max_length 2048。**兩臂掃同一張網格、同樣條件，唯一差別是 `--peft_method`**）
>
> LoRA（每組約 87 分鐘）。`rewards/*` 是 β·log(π訓練後/π_base) 的 token 總和，
> 0 = 與 base 相同、負 = 機率被壓低；括號內是 late − early 的變化量：
>
> | lr | rewards/chosen early→late (Δ) | rewards/rejected early→late (Δ) | 壓rej/壓chosen | margin early→late | acc early→late | grad 峰值 | chosen 保留/token |
> |---|---|---|---|---|---|---|---|
> | **5e-6** ✅ | -0.0026 → -0.1500 (-0.147) | -0.0127 → -0.6952 (-0.682) | **4.63** | 0.010 → 0.545 | 0.534 → 0.724 | 58 | 99.3% |
> | 2e-5 | -0.0330 → -3.4503 (-3.417) | -0.1687 → -8.2426 (-8.074) | 2.36 | 0.136 → 4.792 | 0.628 → 0.975 | 106 | 86.0% |
> | 5e-5 | -0.1972 → -17.2034 (-17.006) | -1.0295 → -34.0913 (-33.062) | 1.94 | 0.832 → 16.888 | 0.706 → 0.989 | 366 | 47.0% |
> | 1e-4 | -1.0352 → -36.6601 (-35.625) | -2.9977 → -69.9023 (-66.905) | 1.88 | 1.962 → 33.242 | 0.764 → 0.995 | 569 ⚠️ | 20.0% |
> | 2e-4 | -3.2846 → -35.3859 (-32.101) | -7.5961 → -83.7929 (-76.197) | 2.37 | 4.311 → 48.407 | 0.814 → 0.997 | 608 ⚠️ | 21.2% |
>
> Prefix（每組約 55 分鐘）。`rewards/*` 是 β·log(π訓練後/π_base) 的 token 總和，
> 0 = 與 base 相同、負 = 機率被壓低；括號內是 late − early 的變化量：
>
> | lr | rewards/chosen early→late (Δ) | rewards/rejected early→late (Δ) | 壓rej/壓chosen | margin early→late | acc early→late | grad 峰值 | chosen 保留/token |
> |---|---|---|---|---|---|---|---|
> | 5e-6 | -0.7780 → -0.7383 (+0.040) | -0.8421 → -0.8575 (-0.015) | ∞ | 0.064 → 0.119 | 0.521 → 0.554 | 45 | 96.8% |
> | 2e-5 | -0.7735 → -0.7394 (+0.034) | -0.8484 → -0.9466 (-0.098) | ∞ | 0.075 → 0.207 | 0.532 → 0.612 | 34 | 96.8% |
> | **5e-5** ✅ | -0.7670 → -0.8250 (-0.058) | -0.8613 → -1.2268 (-0.365) | **6.30** | 0.094 → 0.402 | 0.550 → 0.692 | 67 | 96.4% |
> | 1e-4 | -0.7637 → -1.0707 (-0.307) | -0.8921 → -1.6975 (-0.805) | 2.62 | 0.129 → 0.627 | 0.578 → 0.730 | 3722 ⚠️ | 95.4% |
> | 2e-4 | -0.8065 → -4.2191 (-3.413) | -1.0106 → -5.2037 (-4.193) | 1.23 | 0.204 → 0.985 | 0.612 → 0.647 | 3672 ⚠️ | 83.1% |
>
> **∞ 是哨兵值不是計算結果**：比值定義為 `rejected 下降量 ÷ chosen 下降量`，
> 但 chosen 若沒有下降（Δ ≥ 0），相除會得到負數、失去意義，
> 所以 `compare_lr_sweep.py` 直接回傳 ∞ 表示「完全沒有犧牲 chosen」這個最佳情況。
> **⚠️ 但 ∞ 不代表該 lr 最好** —— 比值只衡量「選擇性」，不衡量「學了多少」。
> 5e-6 / 2e-5 的 chosen 上升是因為模型幾乎沒動（acc 0.554 / 0.612，
> 對照 S0 已診斷為 underfit 的 0.607），所以選擇規則是
> **「比值 ≥ 3 且梯度未爆的最高 lr」**，不是「比值最大」。
>
> **兩臂的結構差異（值得寫進論文）**：
>
> 1. **起步位置**：兩者都零初始化，但 prefix 在前 50 步就跳到 chosen ≈ −0.77 然後幾乎不動
>    （5 個 lr 的 early 值都在 −0.76~−0.81），LoRA 則從 −0.003 慢慢漂。
>    **Prefix 是「一次大跳然後飽和」，LoRA 是「持續累積」** —— 插入虛擬 token 會立刻
>    改變所有位置的 attention，但可調的方向有限。
> 2. **動態範圍差一個量級**：同樣掃 5e-6→2e-4，LoRA 的 chosen Δ 從 −0.15 走到 −35.6（240 倍），
>    prefix 只從 +0.04 走到 −3.4（85 倍，且變化集中在最後一格）。Prefix 對 lr 反應遲鈍得多。
> 3. **崩潰模式不同**：LoRA 是**漸進**的（比值 4.63 一路滑到 1.88，無斷點）；
>    prefix 有**明確斷點**（5e-5→1e-4 之間，比值 6.30→2.62、梯度 67→3722，55 倍）。
> 4. **acc 天花板**：LoRA 最高 0.997，prefix 最高只到 0.730。這與參數量 4 倍差距一致
>    （3.1M vs 12.6M），是 P-match 系列要釐清的核心問題。
>
> **D-lr-lora 結果**（2026-08-28，DPO、batch 64、16,000 筆 = 250 steps、每組約 87 分鐘）：
>
> 判讀工具對 5 組全部標 ❌「reward 崩潰」，但**那個門檻對 DPO 過度敏感**——
> DPO 的 reward 是 β·log(π/π_ref) 的 **token 總和**、不做長度歸一化，
> 絕對值會隨序列長度放大，不能沿用 SimPO 的絕對門檻。
> 換算成每 token（β=0.05、chosen 中位數 456 token）後才看得懂：
>
> | lr | chosen 每 token 漂移 | 機率保留 | **壓rej/壓chosen** | acc |
> |---|---|---|---|---|
> | **5e-6** | −0.006 | **99.4%** | **4.63** ✅ | 0.724 |
> | 2e-5 | −0.150 | 86.1% | 2.36 | 0.975 |
> | 5e-5 | −0.746 | 47.4% | 1.94 | 0.989 |
> | 1e-4 | −1.562 | 21.0% | 1.88 | 0.995 |
> | 2e-4 | −1.408 | 24.5% | 2.37 | 0.997 |
>
> **關鍵是最後一欄的比值**（尺度無關，已加進 `compare_lr_sweep.py` 成為固定欄位）：
> 兩側同時下降是偏好學習的常態，重點在**不對稱程度**。
> 5e-6 壓 rejected 的幅度是壓 chosen 的 4.63 倍、chosen 幾乎沒動；
> lr 一往上比值就掉到 ~1.9，變成兩側等速下沉，那才是真的崩。
>
> **定案 lr = 5e-6**，與論文 Table 6 相同。
> ⚠️ 全長訓練的 β 必須與這輪 sweep 相同，跑之前先確認
> `grep -h "^objective=" outputs/lr_sweep_dpo_lora/*.log | head -1`。
>
> **D-run 結果**（2026-08-30，DPO β=0.05、800 steps @ batch 64）：
>
> | | LoRA lr=5e-6 | Prefix lr=5e-5 | *Exp1（參照）* |
> |---|---|---|---|
> | chosen 每 token 保留 | 88.8% | 94.6% | *96.3%* |
> | acc | **0.975** | 0.753 | *0.78* |
> | margin | **4.11** | 0.82 | *1.1* |
> | grad 峰值 | 87 | **18,657** | *25* |
>
> prefix 的梯度尖峰共 5 次（step 245/410/530/535/770），量級遞減、且 chosen 每次都回升
> （step 245 一度 −3.24，最終回到 −1.27）→ 偶發壞 batch，非失控。
> **這正好展示 DPO 的 reference 錨在做什麼**：同樣的尖峰在 SimPO 下是不可逆的。
>
> **⚠️ D-screen 發現量測缺陷：退化守門一直被 `max_new_tokens=512` 截斷污染。**
>
> 便宜篩選（100 題×5，c/cpp）的原始數字：
>
> | | 漏洞率 Δ | 語法完整率 Δ | 程式碼長度 |
> |---|---|---|---|
> | LoRA ck800 | **−25.60** | **−46.00** | +75.6% |
> | LoRA ck300 | — | −6.60 | +24.4% |
> | Prefix ck800 | −4.40 | −4.60 | +1.6% |
>
> 直接看生成內容，發現兩臂的樣本都**斷在單字/參數列中間**——是撞到
> `max_new_tokens=512`（1679 字元 ÷ ~3.5 ≈ 480 token）。括號配不起來 →
> 被判「語法不完整」。長度增幅與完整率下降呈**非線性**（+2%→−4.6、+24%→−6.6、
> +75%→−46.0），正是硬上限的特徵。
>
> **所以 LoRA 的 −25.60 不可信**：破碎的程式碼讓靜態分析器比對不到規則，被算成「安全」。
> **影響範圍不只這次** —— 歷史上所有 ON 比 OFF 長的評測（Exp1、Exp2、S1-ck）都可能被污染。
>
> **已修**：`gen_for_icd.py` 現在把 `truncated` 旗標寫進輸出，
> `check_degeneration.py` 改用「未截斷者的語法完整率」做判定，並在截斷率 > 20% 時警告。
> - [x] **D-screen2** 帶 `--max_new_tokens 1024` 重跑 → 截斷混淆確認，LoRA 語法 −46.00 → **+0.25**
> - [ ] **D-eval** 兩臂全套 693 題 × 10 samples（等 D-screen2 過關）
>
> **D-screen2 結果**（2026-08-30，`--max_new_tokens 1024`、100 題×5、c/cpp）：
>
> | | 漏洞率 Δ | 語法（未截斷）Δ | 截斷率 ON | 程式碼長度 | 註解率 OFF→ON |
> |---|---|---|---|---|---|
> | **LoRA** | **−18.80** | **+0.25** ✅ | 14.4% | **226%** | 18.9% → **48.6%** |
> | **Prefix** | −6.20 | −4.74 ❌ | 2.6% | 103% | 18.9% → 20.6% |
>
> 把 `max_new_tokens` 從 512 提到 1024 之後，LoRA 的語法完整率從 −46.00 翻成 **+0.25**
> —— 證實先前的守門失敗確實是截斷造成的量測缺陷，不是模型退化。
>
> ### ⚠️ S2 停損點觸發
>
> 差距 12.6 pt > 門檻 10 pt。100 題×5 的標準誤約 2.2pt，方向可信。
> **但轉向前有兩個疑點必須先排除**，否則結論會是「prefix 在沒調到位的設定下輸了」，
> 那不是能力邊界分析：
>
> **疑點一：LoRA 可能在鑽偵測器的漏洞。** 註解率從 18.9% 暴增到 **48.6%**（近半數輸出是註解），
> 程式碼長度 226%。抽樣可見它**在註解裡寫安全建議、然後照樣寫出漏洞碼**：
> 一個樣本的 `is_jpeg()` 直接 `return true`（假驗證），並在註解寫
> 「'system' is considered insecure... Consider using execvp」之後仍然呼叫 `system()`。
> 同一段還有 `"..." + input_filename`（C 的 char* 不能用 + 串接，根本編不過），
> 但括號配對的啟發式判它「語法完整」。
> 這是 ProSec 論文 Figure 8「不觸發偵測器 ≠ 安全」的變體。
> - [x] **D-untrunc** 成對移除截斷後重算 → **幾乎沒變**：LoRA −18.80 → **−17.80**、
>       Prefix −6.20 → **−5.82**，差距 12.0 pt。**截斷不是原因，疑點一排除。**
>
> **疑點二：兩臂操作點不對等。** LoRA acc 0.975 / margin 4.11，Prefix 只有 0.753 / 0.82。
> Prefix 明顯訓練強度較低，停在 −6.20 可能是「lr 保守」也可能是「nvt=16 容量不足」，
> 只有後者才叫能力邊界。
> **⚠️ 疑點二有了精確的檢驗方式：目前的比較在參數量上差 4 倍。**
>
> | 設定 | 可訓練參數 |
> |---|---|
> | prefix **nvt=16**（現用） | **3,145,728** |
> | LoRA r=8 **all-linear**（現用） | **12,582,912** |
>
> 而且數字剛好可以精確配對（Phi-3：H=3072、L=32、prefix = nvt×L×2×H）：
>
> | 配對 | Prefix | LoRA | 參數 |
> |---|---|---|---|
> | 高容量 | **nvt=64** | all-linear ✅已有 | 12,582,912 |
> | 低容量 | nvt=16 ✅已有 | **只掛 qkv_proj** | 3,145,728 |
>
> - [ ] **P-match-hi** prefix nvt=64（對上現有 LoRA all-linear），lr 5e-5、800 steps
> - [ ] **P-match-lo** LoRA 只掛 qkv_proj（對上現有 prefix nvt=16），lr 5e-6、800 steps
> - [ ] **P-match-eval** 兩組都跑便宜篩選；這同時產出 S2-RQ2 要的資源對照表
>
> 兩組跑完才有資格宣告 H1 不成立——否則結論只是「prefix 在四分之一的參數預算下輸了」。
>
> **兩項都做完才有資格宣告 H1 不成立。**
>
> **D-humaneval 結果**（2026-09-03，`--max_new_tokens 2048`、greedy、164 題）：
>
> | | OFF(base) | ON | Δ |
> |---|---|---|---|
> | **LoRA** | 70.73% | 66.46% | **−4.27** |
> | **Prefix** | 70.73% | 59.76% | **−10.98** |
>
> 截斷 0/164（兩臂、ON/OFF 皆是），數字乾淨。
>
> ### ⚠️ H2 被推翻：prefix 傷 utility 傷得**更重**
>
> 原假設是「prefix 的 utility 掉幅小於 LoRA」（S3 停損點的其中一條）。實測相反：
> prefix 掉 10.98 pt，是 LoRA（4.27 pt）的 **2.6 倍**。
>
> 加上 D-screen2 的安全性（prefix −5.82 vs LoRA −17.80），
> **prefix 在兩個軸上都輸**：學得比較少，卻傷得比較重。
>
> **退化守門沒有預測到這件事**：prefix 的程式碼長度保留 103%、註解率幾乎不變，
> LoRA 則是 226% / 48.6%——守門看起來 prefix 完好無損。
> **教訓：三條守門條件是「有沒有大幅退化」的篩子，不能代替 pass@1。**
> 程式碼長度正常、語法完整，不代表功能正確。
>
> **可能的機制（待驗證，勿當結論）**：prefix 把 16 個虛擬 token 插進**每一層**的 KV cache，
> 會改變**所有位置**的 attention——是全域且非針對性的擾動；LoRA 的低秩權重修正則可以
> 學成比較外科手術式的。這與 D-lr 觀察到的「prefix 在前 50 步就跳到 chosen ≈ −0.77
> 然後飽和，五個 lr 都一樣」一致：**prefix 在學到任何東西之前就先付了一筆固定的 utility 代價。**
>
> ### 附帶確認：舊的 utility 數字確實被截斷污染
>
> 同一個 base model、同一支腳本：
>
> | max_new_tokens | base HumanEval pass@1 |
> |---|---|
> | 512（Exp1/Exp2 用的） | **64.63%** |
> | 2048（現在） | **70.73%** |
>
> **光是 base 就被低估 6.1 pt。** 這證實 Exp1（−3.05）、Exp2（−11.59）、
> MultiPL-E cpp（−24.22）的 utility 數字全部不可信，
> 連帶「epochs=2 會過度訓練」這條結論的證據強度要打折。
>
> - [x] **D-humaneval** 兩臂 HumanEval → LoRA −4.27、Prefix −10.98
> - [x] **D-hedebug** 逐題檢查 prefix 的 27 題退步 → **不是 bug，是真的能力退步**
>
> 擔心過的可能性是「prefix 與生成時的 KV cache 互動有問題」——那類 bug 的症狀
> 正好是「訓練指標健康、生成品質掉」，而本 session 已經踩過一次
> （gradient checkpointing 重算把 prefix 串接兩次）。逐題檢查排除了它：
>
> | 題目 | ON 的錯誤 | 驗證方式 |
> |---|---|---|
> | HumanEval/102 | 只判斷端點、沒在區間內搜尋；`choose_num(12,15)` → 12（正解 14） | 實際執行 |
> | HumanEval/106 | 括號不配對的巢狀 comprehension | `SyntaxError` |
> | HumanEval/11 | `str(int(x) ^ y)` 漏了 `int(y)` | `TypeError` |
>
> 全部是格式正常、註解清楚、但寫錯的程式碼。沒有重複、亂碼、截斷或抽取失敗
> → **生成路徑與 `extract_code` 都正確**。
>
> **行為變化**：ON 傾向寫更壓縮的一行式、並加上多餘的 `# Example usage` /
> `if __name__ == "__main__":`。第 11 題最露骨——它**把 docstring 裡的預期輸出
> 從 `'100'` 改成 `'000'`**，改寫規格去遷就自己的錯誤答案。這是推理與指令遵循的退化。
>
> 規模：164 題中 27 題退步、約 9 題進步，淨掉 18 題。
>
> **共用程式碼因此也被反證是好的**：兩臂走同一支 `train_prefix.py`、同一份資料、
> 同一個 chat template、同一套評測腳本，只有 peft config 那幾行不同。
> LoRA 行為合理（−4.27 / −17.80、曲線健康），所以資料前處理與訓練管線沒有問題。
> - [ ] **D-multipl-e** 兩臂 MultiPL-E js/cpp（記得帶 `--max_new_tokens 2048`）
>
> **P-match 訓練結果**（2026-09-03，DPO β=0.05、800 steps @ batch 64，
> 除 `--peft_method` 與各自 sweep 的 lr 外設定全同）：
>
> | | 參數 | rewards/chosen early→late | 保留/token | margin late | 成長% | acc late | grad 峰值 |
> |---|---|---|---|---|---|---|---|
> | LoRA all-linear | 12.6M | −0.031 → −2.707 | 88.8% | **4.113** | +3385 | **0.975** | 87 |
> | LoRA qkv_proj | 3.1M | −0.014 → −0.530 | **97.7%** | 1.611 | +3732 | 0.885 | 71 |
> | prefix nvt=16 | 3.1M | −0.770 → −1.270 | 94.6% | 0.822 | +340 | 0.752 | 18657 |
> | prefix nvt=64 | 12.6M | **−4.441** → −2.930 | 87.9% | 1.323 | **+66** | 0.740 | 282 |
>
> ### 容量混淆排除：同參數量下 LoRA 在兩軸都勝
>
> | 參數預算 | LoRA | prefix |
> |---|---|---|
> | 3.1M | qkv_proj：acc **0.885** / 保留 **97.7%** | nvt=16：acc 0.752 / 保留 94.6% |
> | 12.6M | all-linear：acc **0.975** / 保留 **88.8%** | nvt=64：acc 0.740 / 保留 87.9% |
>
> 更強的一點：**LoRA 在四分之一的參數（qkv_proj, 3.1M）就勝過 prefix 的全額參數
> （nvt=64, 12.6M）**——acc 0.885 vs 0.740、保留 97.7% vs 87.9%。
> **所以「prefix 較弱是因為只拿到四分之一預算」這個辯護不成立，H1/H2 的推翻站得住。**
>
> ### prefix 加大容量反而更差（兩軸同時退步）
>
> | nvt 16 → 64 | 變化 |
> |---|---|
> | acc | 0.752 → 0.740（**−0.012**） |
> | chosen 保留/token | 94.6% → 87.9%（**−6.6 pt**） |
> | margin 成長 | +340% → **+66%** |
>
> **機制在數字裡看得很清楚**：nvt=64 的 `chosen early` 是 **−4.441**，
> 而 nvt=16 只有 −0.770 —— 初始擾動大了 **5.8 倍**。
> 它最終 margin 1.323 裡有 0.796 在 early 就存在了（學習之前），
> 之後只成長 +66%。**加長 prefix 主要是加大了那筆一次性的擾動，而不是增加學習能力。**
>
> 兩側 Δ 也證實這點：nvt=64 的 chosen (+1.511) 與 rejected (+0.983) **同時上升**，
> 是「從初始擾動回升」而非「選擇性壓低」。
> （`compare_lr_sweep.py` 原本對這種情況印 ∞，會誤讀成最佳形狀，已改印 `↑↑`。）
>
> 附帶：nvt=16 的 grad 峰值 18,657、nvt=64 只有 282——**更長的 prefix 梯度反而更穩**，
> 但表現更差。所以 nvt=16 那些尖峰不是它表現不好的原因。
>
> ⚠️ **這些都還只是訓練期的代理指標（acc / 保留率）**，必須由實際評測確認。
> 但方向很一致。
>
> **一個可測的預測**：`lora_qkv` 的保留率 97.7% 遠高於 all-linear 的 88.8%，
> 所以它的 utility 掉幅應該明顯小於 all-linear 的 −4.27，可能接近論文的 +1.42。
> 若成立，就同時解釋了「我們的 utility 為什麼跟論文反向」——論文很可能沒有掛 all-linear。
>
> **異常記錄**：`lora_qkv` 跑了 20,801 秒，比參數多 4 倍的 all-linear（12,890 秒）
> 還久 61%。參數更少不該更慢，推測是當時 GPU 上有其他工作，非程式問題，但要留意。
>
> - [x] **P-match-hi** prefix nvt=64 訓練完成
> - [x] **P-match-lo** LoRA qkv_proj 訓練完成
> - [ ] **P-match-eval** 四臂評測（安全性統一用 `--max_new_tokens 2048`；
>       HumanEval 只需跑兩個新臂，舊兩臂已是 2048）
>
> **P-match utility 結果**（2026-09-03，HumanEval、`--max_new_tokens 2048`、greedy）：
>
> | | 3.1M 參數（佔全模型 0.082%） | 12.6M 參數（0.329%） |
> |---|---|---|
> | **LoRA** | qkv_proj：**+1.22** ✅ | all-linear：−4.27 |
> | **prefix** | nvt=16：−10.98 | nvt=64：**−32.93** ❌ |
>
> （base OFF = 70.73%，四臂完全相同。）
>
> ### 兩個乾淨的結論
>
> **① LoRA 只掛 qkv_proj 時 utility 不降反升（+1.22），與論文的 DPO（+1.42）幾乎一致。**
> 這證實了 `docs/REPRO_PAPER.md` §3 那條假設猜錯了——**論文很可能沒有掛 all-linear**。
> all-linear 是 LLaMA-Factory 新版的預設，我照著補上，結果多出來的
> `o_proj / gate_up_proj / down_proj` 正是 utility 掉 4.27 的原因。
> 「我們的 utility 跟論文反向」這個落差就此解釋掉了。
>
> **② prefix 的 utility 代價隨長度暴增，而 LoRA 不會。**
>
> | 容量 3.1M → 12.6M | utility 變化 |
> |---|---|
> | LoRA（qkv → all-linear） | +1.22 → −4.27（**−5.5 pt**） |
> | prefix（nvt 16 → 64） | −10.98 → −32.93（**−22.0 pt**） |
>
> 同樣把參數放大 4 倍，prefix 的損害是 LoRA 的 **4 倍**。
> 與訓練期觀察到的機制一致：nvt=64 的 `chosen early` 是 −4.441（nvt=16 只有 −0.770），
> 初始擾動大 5.8 倍；**加長 prefix 主要在加大那筆一次性的全域擾動，不是增加學習能力。**
>
> 逐題數：nvt=64 有 **59 題退步、5 題進步**（nvt=16 是 27/9）；
> lora_qkv 是 **10 題退步、12 題進步**——真正的雙向改善，不是零和。
>
> ### 保留率無法跨方法預測 utility（方法上的教訓）
>
> | | chosen 保留/token | HumanEval Δ |
> |---|---|---|
> | LoRA qkv | 97.7% | **+1.22** |
> | prefix nvt=16 | 94.6% | −10.98 |
> | LoRA all-linear | 88.8% | −4.27 |
> | prefix nvt=64 | 87.9% | −32.93 |
>
> 在同一個方法內部單調（LoRA 97.7% > 88.8%、prefix 94.6% > 87.9%），
> **但跨方法完全失效**——prefix nvt=16 的保留率(94.6%)高於 LoRA all-linear(88.8%)，
> utility 卻掉 2.6 倍。**訓練期的代理指標只能在同一方法內比較。**
> 這是繼「退化守門預測失敗」之後第二次同樣的教訓。
>
> **⚠️ 小瑕疵**：nvt=64 有 6/164（3.7%）撞到 `max_new_tokens=2048`，
> 所以 −32.93 略被低估（最多回補 3.7 pt → 仍在 −29 附近），不影響結論。
>
> - [x] **P-match-eval（utility）** 四臂 HumanEval 完成
> - [ ] **P-match-eval（安全性）** 四臂統一用 `--max_new_tokens 2048` 重跑
>
> 以下為 SimPO 時期的紀錄，保留供論文的「為什麼不用 SimPO」一節引用。

### S1（歷史）· DPO → SimPO 目標函數切換

- [x] **S1-code** `train_prefix.py` 加 `--objective {dpo,simpo}`（SimPO 走 `CPOTrainer`）
- [x] **S1-code** 加 `--gamma`、`--cpo_alpha`；`--beta` 改為依 objective 自動取預設
- [x] **S1-code** `--prefix_init_scale 0` 改走 PEFT 原生 `init_weights="zero"`
- [x] **S1-code** `compare_lr_sweep.py` 判讀門檻改為尺度無關 + 自動辨識 objective
- [x] **S1-code** `run_lr_sweep.sh` 加 `OBJECTIVE` 參數
- [x] **S1-lr** SimPO lr 確認（1e-4 / 3e-4 / 1e-3）→ **三組全部失敗，問題不在 lr**
- [x] **S1-verify** 核對 ProSec Appendix C Table 6 → **找到三處設定不符，見下**
- [x] **S1-lr2** 論文設定重掃 lr {5e-6, 2e-5, 5e-5} @ batch 64 → **三組都健康，未達上限**
- [x] **S1-lr3** 往上探 lr {1e-4, 2e-4} → **2e-4 進入崩潰區，定案 5e-5**
- [x] **S0-e** `lr_5e-5` 小規模安全評測 → **OVERALL −5.00，退化守門三項全過**
- [x] **S1-loc** edit-locality 量測完成 → **ρ 遠高於預期，H3 需重新評估**
- [x] **S1-full** 全長訓練 1500 steps → **step ~450 崩塌**，chosen −0.51 → −5.17
- [x] **S1-ck** checkpoint-300 vs 1500 比較 → **採用 300**，1500 未過退化守門
- [x] **S1-seed** `gen_for_icd.py` 加 `--seed`（預設 42），且 ON/OFF **同題共用 seed**（配對比較）
- [~] ~~**S1-anchor** `--cpo_alpha 0.05` 錨點~~ → 主線改用 DPO 後不再需要；
      要做也是列為 SimPO 失敗分析的 ablation
- [ ] **S1-eval** 勝出設定跑全套評測（693 題 × 10 samples × 5 語言）+ HumanEval + MultiPL-E
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

**S1-lr3 結果**（延伸到 1e-4 / 2e-4，同樣 batch 64 / 250 steps）：

| lr | margin 成長 | chosen Δ | rejected Δ | grad late | 形狀 |
|---|---|---|---|---|---|
| 5e-6 | +15% | +0.007 | +0.002 | 2.8 | 太慢 |
| 2e-5 | +31% | +0.019 | +0.007 | 7.0 | 太慢 |
| **5e-5** | **+63%** | **+0.024** | −0.000 | 26.2 | ✅ **靠拉高 chosen 擴大 margin，最健康** |
| 1e-4 | +97% | +0.002 | −0.038 | 26.7 | chosen 持平，靠壓低 rejected |
| 2e-4 | +145% | **−0.230** | **−0.291** | **199** | ❌ 兩側一起崩，進入退化區 |

**定案 lr = 5e-5。** 1e-4 的 margin 更大但 chosen 已經持平——在 250 步就這樣，
拉到 1500 步很可能翻轉成 2e-4 那種兩側崩塌。5e-5 是唯一「靠拉高安全碼機率」
而非「靠壓垮漏洞碼」來擴大 margin 的設定。

**S0-e 結果**（`lr_5e-5`，僅 250 steps、16k 樣本，100 題 × 5 samples，涵蓋 c/cpp）：

| | OFF | ON | Δ |
|---|---|---|---|
| C | 63.53 | 58.04 | **−5.49** |
| C++ | 26.53 | 22.04 | **−4.49** |
| OVERALL | 45.40 | 40.40 | **−5.00** |

對照 Exp1（DPO、全量 45.8k、2,861 steps）：C −2.94、C++ −5.13。
**這一輪只跑了 250 步、看過 16k 筆，C 的改善就已經超過 Exp1。**

退化守門**三項全過**：程式碼長度保留 100.6%、語法完整率持平、stub 率反而下降 1.2pt。
安全性數字可單獨回報。

（附帶觀察：prefix 讓模型更常使用 code fence，無 fence 率 82.2% → 76.8%。不影響計分。）

**S1-full + checkpoint 比較結果**（2026-08-27）：

全長 1500 步在 **step ~450 相變**：chosen −0.513 → −5.167（每 token 機率 71% → 3.2%），
rejected −0.564 → −6.197，grad_norm 峰值 4438。margin 確實成長到 1.03（超過目標 0.333），
但是靠「把兩側一起壓垮、只是 rejected 壓更快」達成的。

| | checkpoint-300 | checkpoint-1500 |
|---|---|---|
| OVERALL（c+cpp，100 題×5） | 44.40 → **41.00**（−3.40） | 44.60 → 41.40（−3.20） |
| 語法完整率 Δ | −3.00 ✅ | **−4.20 ❌** |
| 退化守門 | **全過** | **未過** |

**多跑的 1,200 步什麼都沒買到**：安全性沒有更好，語法完整率反而掉破門檻。
→ **SimPO + prefix 在 reward 崩塌之後的訓練是純粹的損害。** 這是可寫進論文的發現。

**⚠️ 但這兩個 checkpoint 的安全性數字分不出高下。** OFF（完全沒動的 base）在三次評測
拿到 45.40 / 44.60 / 44.40，雜訊約 1.0 pt，而兩者差距只有 0.2 pt。**真正做出判決的是
退化守門，不是漏洞率。** 100 題×5 的快速評測標準誤約 2.2 pt，只能看方向、不能比小差距。

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

- [x] **S2-code** `train_prefix.py` 加 `--peft_method {lora,prefix}`（分支 `repro/prosec-lora-simpo`；Phi-3 的 LoRA target 需自己列，PEFT 沒有 phi3 條目）
- [ ] **S2-code** 加資源指標記錄 callback（trainable params / adapter 大小 / peak mem / runtime）
- [~] ~~**S2-a/b/c** LoRA lr sweep 1e-4 / 3e-4 / 5e-4~~ → 由 **D-lr-lora** 取代
      （同一張 5 點網格 5e-6~2e-4、DPO、batch 64，定案 5e-6）
- [x] **S2-run** LoRA 全量訓練 → 由 **D-run** 完成（DPO 而非 SimPO，理由見 S1 反轉）
- [ ] **S2-eval** 安全評測 + HumanEval + MultiPL-E + 退化守門
- [ ] **S2-sanity** 與論文 Table 1 的 Phi-3 數字比對（只檢查方向與量級，不要求相同）
- [ ] **S2-RQ2** 產出資源對照表

> **停損點**：若 Prefix 安全性遠遜 LoRA（差距 > 10 pt）→ H1 不成立，
> 論文主軸改為「Prefix 的能力邊界分析」，及早轉向。

**S2-repro 結果**（2026-08-28，分支 `repro/prosec-lora-simpo`，完全照論文 Appendix C：
LoRA r=8 α=16、SimPO β=1.5 γ=0.5、lr=5e-6、batch 64、1500 steps、45,785 筆未再篩選）：

**❌ 崩塌，而且比 prefix 那次嚴重。** 不是相變，是 **step ~350 起的平滑失控**：

| step | chosen | rejected | margin | grad | 判讀 |
|---|---|---|---|---|---|
| 5–305 | −0.466 → −0.450 | −0.506 → −0.654 | 0.04 → 0.21 | 0.7 → 4.0 | ✅ 理想形狀：chosen 守住、rejected 下降 |
| 355–455 | −0.548 → −1.016 | −0.829 → −2.190 | 0.28 → 1.17 | 4.2 → 8.1 | ⚠️ chosen 開始跟著掉 |
| 505–955 | → −14.8 | → −20.1 | → 5.23 | 峰值 **143.9** | ❌ 加速下墜 |
| 1005+ | −17 ~ −19 | −23 ~ −25 | ≈ 5.6 | 40–147 | ❌ 飽和，模型已毀 |

`chosen = −17.69` 換算成每 token 機率 e^(−11.8) ≈ **7.6e-6**。可用的 checkpoint 只有 **300**
（chosen −0.450，與起點幾乎相同）；600 邊緣（每 token 機率剩起點的 0.39 倍）；900 以上全毀。

**機制（已定位，且推翻先前對 S1-loc 的解讀）**：
step 1455 的平均 margin 5.584，照 SimPO loss 應該是 0.006，實測 **0.271** —— 差 45 倍，
代表**約三分之一的樣本從未越過目標**、梯度從未停過。是哪些樣本？
先前看 ρ 中位數 0.358 > γ/β = 0.333 就判「目標可達」，**該看的是左尾**：
p25 的 ρ 只有 0.141，這些 pair 要撐出 0.333 nats/token 需要每個差異 token 拉開 2.4 nats，做不到。
低 ρ 尾巴永遠停在高梯度區，而 SimPO **reference-free、沒有東西釘住絕對 log-prob**，
加上共同 token 佔 64%，優化器唯一的出路就是把兩側一起壓垮。
DPO 兩次都沒崩，正是因為有 reference model 當錨。

**⚠️ 不可據此下「prefix 較差」的結論** —— LoRA 在同一組設定下崩得更嚴重，
這是**管線層級**問題，不是 PEFT 方法比較。

**待驗的三個候選原因**：
- [ ] **S2-ck300** checkpoint-300 便宜篩選（100 題×5，c/cpp）+ 退化守門 —— 資訊量最大，先跑
- [ ] **S2-tgt** `--lora_target_modules qkv_proj`（只掛注意力）重跑 —— 論文只寫 r/α，
      all-linear 是我們自己補的假設；Phi-3 沒有 `q_proj/v_proj`，融成 `qkv_proj`。**不偏離論文**
- [ ] **S2-anchor** `--cpo_alpha 0.05` 錨點 —— 直接測「無錨」假設，但會變成 CPO-SimPO 混合

**S2-eval-fix**（2026-08-28）：`gen_for_icd.py` / `run_humaneval.py` / `run_multipl_e.py`
原本只送 user 訊息、沒有 system prompt，但復現用的 adapter 是帶
`You are helpful coding assistant.` 訓練的 → 評測時 adapter OOD。三支都加了 `--system_prompt`。
**早期 prefix 實驗訓練時沒有 system prompt，那些評測要維持不給。**

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

**S1-loc 結果**（3,000 筆抽樣，Phi-3 tokenizer）：

| 統計量 | p25 | p50 | mean | p75 | p95 |
|---|---|---|---|---|---|
| edit ratio ρ | 0.141 | **0.358** | 0.463 | 0.835 | 0.933 |
| edit hunk 數 | 9 | **17** | 21 | 28 | 48 |
| chosen 長度 | 279 | 443 | 496 | 661 | 1032 |
| rejected 長度 | 387 | 545 | **576** | 743 | 1019 |

- 單一 contiguous diff（hunk=1）：**2/3000 (0.1%)**
- 共同前後綴只佔 chosen 平均長度的 **14.3%**
- 超過 `max_length=1024`：chosen 11.6%、rejected 12.7%

**⚠️ 這推翻了 H3 的前提。** 計畫假設 pair「只差 1~3 行、差異高度集中」，
實際是**中位數 36% 的 token 不同、散成 17 個區塊、幾乎沒有單一連續 diff**。
S5 的處置見下方 S5 區塊的修訂。

**副作用（好消息）**：ρ 中位數 0.358 > γ/β = 0.333，所以 SimPO 的目標 margin
**是可達的**——這解釋了為什麼對齊論文設定後訓練就正常了，也正式推翻我先前
「γ 結構上不可達」的假設。

**已知缺陷**：12% 的樣本被 `max_length=1024` 截斷。SimPO 除以 |y|，對截斷敏感。
S3 應把 max_length 提到 1536（p95 約 1032）並記錄影響。
另注意 **rejected 平均比 chosen 長**（576 vs 496），與「修補版比較長」的直覺相反。

---

## L2 · 補強層 —— 決定論文層級

### S4 · D_norm influence selection 【❌ 2026-08-27 取消】

> **REPORT §4.3 已考證出 45,785 筆資料已經經過 §3.3 篩選**（嚴格分區、`y_f` 相似度
> p50 0.146、Dnorm 每組上限 2）。再跑一次是**第二道過濾，不是復現**。
> 「我們用未篩選資料」這個說法已撤回，不得再用來解釋 utility 掉幅。
> 直接訓練這 45,785 筆才是較忠實的復現。
>
> 保留下方內容僅供紀錄。若真要做，定位必須改為「額外的資料篩選 ablation」。
>
> **⚠️ 2026-08-27 前提變更**：`prosec-mixed-phi3mini-4k-inst` 很可能**已經是 selection 的輸出**，
> 不是候選池。三項證據：(1) 每個 D_sec 的 D_norm 候選數硬性卡在 2（=`top_n=2`）；
> (2) `mix_and_upload` 的邏輯產不出 11,306 個相異 y_f 配 18,385 列；
> (3) 列的順序是 D_sec 全在前、D_norm 全在後 —— `mix_and_upload` 會 shuffle，
> `sample_refactored.py` 不會。完整推導見 `docs/REPRO_PAPER.md` §4。
>
> **後果**：top-2 那層在我們的資料上是 no-op，只有「丟最低 20%」會動
> （18,385 → 14,708，總數 42,108）。再跑一次可能是**第二次篩選**，不是復現。
> 這也讓 `REPORT.md` §4.3「我們用的是未篩選候選池」這條歸因需要重寫。

- [x] **S4-code** 確認 ProSec repo 有無原版 selection script → **有**，`ProSec/influence_score/{training_dynamics,sample}_refactored.py`；已照它重寫成 `data/collect_training_dynamics.py` + `data/select_dnorm.py`
- [ ] **S4-data** 產出篩選後資料集（目標量級約 10k），記錄分布變化
- [ ] **S4-E1'** LoRA 在篩選後資料上重跑
- [ ] **S4-E2'** Prefix 在篩選後資料上重跑
- [ ] **S4-分析** selection 對 LoRA 與 Prefix 的效果是否對稱

**結果**：_（待填）_

---

### S5 · Diff-token masked SimPO —— 最有創新性的一段

> **⚠️ 2026-08-28 再修訂：S2-repro 的崩塌把 S5 從「加分項」推回主線。**
> LoRA 在論文原始設定下平滑失控，機制定位為「**ρ 左尾（p25 = 0.141）的 margin 目標不可達
> → 那些樣本永不收斂 → reference-free 的 SimPO 只能把兩側一起壓垮**」。
> 論述重心因此改變：不再是「梯度浪費在相同 token 上」（那個說法弱），
> 而是「**長度歸一化讓低 ρ 樣本的目標結構上不可達**」——masking 直接移除分母裡的共同 token，
> 正是對症的解法。`S5-bucket` 的低 ρ 桶從「分析」升級為「問題本體」。
>
> **⚠️ 2026-08-27 修訂：實測後 H3 的前提大幅減弱。**
> ρ 中位數 0.358、17 個 hunk、單一連續 diff 僅 0.1%。原本設想的「差異高度集中、
> 梯度大量浪費在相同 token 上」並不成立——共同前後綴只佔 14.3%。
>
> **但 masking 沒有死**：ρ=0.358 代表仍有 64% 的 token 是相同的，把它們排除
> 仍是實質改動。變的是論述重心：
> - 主線從 **H3（局部性對 Prefix 幫助更大）** 轉為 **H4（adaptive）**——
>   只有 p25 以下（ρ < 0.141，佔 25%）的 pair 才真正「局部」
> - `S5-overlap`（diff 與 analyzer 行號重疊率）從「加分項」升為**決定性檢查**：
>   17 個散落 hunk 裡有多少是安全修補、多少是無關重寫？若多為無關重寫，
>   masking 抓到的是雜訊，S5 應直接放棄，論文停在 L1 + L3
> - 若 S5 放棄，方法創新改由 §10.1（CWE-specific 多 Prefix）承擔

#### 前置分析（必須先做，幾小時，可省下一週實作）
- [x] **S5-loc** edit-locality 統計完成（見 S1-loc 結果）
- [ ] **S5-overlap** **diff 與 analyzer finding 行號的重疊率** —— **現在是 S5 的成敗關鍵**
- [ ] **S5-bucket** 依 ρ 分桶：低 <0.141（25%）／中 0.141~0.835／高 >0.835（25%）

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
