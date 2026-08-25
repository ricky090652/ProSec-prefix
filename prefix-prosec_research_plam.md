# Prefix-ProSec 研究計畫最終定案（詳細版）

## 1. 最終研究定位、範圍與方法層級

### 1.1 暫定題目
Prefix-ProSec: Lightweight and Extensible Proactive Security Alignment for Code Large Language Models

中文暫定題目：
> Prefix-ProSec：可擴充的輕量化 Code LLM 主動安全對齊方法

### 1.2 最終主線
第一階段的核心方法為：
> PROSEC 公開資料 + 完整序列 SimPO + Prefix Tuning + 安全／功能／效率聯合評估

這一階段確實主要是將原始 PROSEC 的 LoRA 改為 Prefix Tuning。它的目的不是宣稱新的 preference loss，而是建立受控且可重現的基礎：固定資料、模型、loss 與評估，只改變可訓練參數的形式，回答 Prefix 是否足以承載安全對齊，以及能節省多少參數、儲存與部署成本。

純粹完成這項替換，方法創新屬於「最低程度的增量式創新」。因此完整研究分成三層：
1. **必要基線層**：Prefix-ProSec。LoRA 改為 Prefix，完整序列 SimPO 不變。
2. **主要增強層**：Locality-Aware Prefix-ProSec。根據 repair pair 的局部性，選擇完整序列或 FIM-localized SimPO。
3. **延伸比較層**：LPO、KTO、CWE-specific prefixes、verifier-based weighting 等，依資源與實驗結果選做。

### 1.3 最終建議
不應一開始同時實作所有方法。最穩健的順序是：
1. 重現 PROSEC-LoRA。
2. 完成 Prefix-ProSec。
3. 跑完整 benchmark 與 edit-locality analysis。
4. 若結果與資料特性支持，再實作簡化 FIM。
5. 若 FIM 有效，再升級為 adaptive full/local 方法。

KTO 不放入第一階段，因為 PROSEC 已提供成對資料；只有後續加入大量未配對的 SAST／test-labelled samples 時才值得使用。

---

## 2. 研究動機、核心假設與研究問題

### 2.1 研究動機
PROSEC 已證明高品質的主動式安全資料與 preference optimization 可以大幅降低漏洞率，並透過 $D_{	ext{norm}}$ 維持一般程式碼能力。但原方法使用 LoRA，每一個安全任務、模型版本或安全政策仍需保存一組較大的 adapter。

Prefix Tuning 凍結完整 Code LLM，只在各 transformer layer 注入少量 continuous prefix parameters，具有以下潛在優勢：
* Adapter 體積小，容易保存、交換與停用。
* 同一個 Base Code LLM 可掛載不同安全 Prefix。
* 對既有 post-trained model 的干預較小。
* 適合研究「安全性控制」而不只是重新訓練整個模型。

限制是 Prefix 容量小，可能無法吸收完整序列中分散的安全訊號。因此後續才考慮 FIM／localized preference，使有限的 Prefix 參數更集中學習安全修補區域。

### 2.2 核心假設
* **H1：參數效率假設**。Prefix-ProSec 能以遠少於 LoRA 的 trainable parameters，取得大部分安全性改善。
* **H2：功能維持假設**。保留 PROSEC 的 $D_{	ext{norm}}$ 後，Prefix-ProSec 不會顯著降低 HumanEval、MXEval 與 CWEval functionality。
* **H3：局部性假設**。若 secure/insecure pair 的差異高度集中，localized/FIM SimPO 對 Prefix 的幫助可能高於對 LoRA 的幫助，因為 Prefix 的可訓練容量較有限。
* **H4：非普遍性假設**。Localized loss 不一定對所有資料有效；大範圍、多區塊或具有長距離依賴的安全修補，可能仍適合完整序列 SimPO。因此最終增強方法應考慮 adaptive full/local，而不是所有資料一律 mask。

### 2.3 研究問題
* **RQ1**：在相同 PROSEC 資料與 SimPO 下，Prefix 是否能接近 LoRA 的安全性與功能性？
* **RQ2**：Prefix 相對 LoRA 能減少多少 trainable parameters、checkpoint size、optimizer memory、GPU-hours 與部署儲存？
* **RQ3**：Prefix-ProSec 在 CWEval 上的 func@k 與 func-sec@k 是否優於 Base model，且不以失去功能性換取表面安全？
* **RQ4**：edit ratio、CWE、語言、程式語言長度與修改區塊數量是否會影響 Prefix-ProSec 的效果？

---

## 3. 方法來源、文獻對應與創新邊界

| 元件 | 本研究用途 | 主要參考來源 |
| :--- | :--- | :--- |
| 主動漏洞誘發指令與 repair pairs | 治用資料生成概念與公開資料 | PROSEC |
| $D_{	ext{sec}}$、$D_{	ext{norm}}$ | 安全學習與功能維持 | PROSEC |
| 完整序列 preference objective | 第一階段主要 loss | SimPO、PROSEC |
| Continuous security prefix | 取代 LoRA、建立可插拔安全模組 | Prefix-Tuning、SVEN |
| 單一 property prefix 的 comparative training | 支持用偏好資料學單一 Prefix | Comparative Prefix-Tuning |
| FIM localized preference | 第二階段集中修補區域梯度 | Alignment with FIM / StructureCoder |
| Diff-token localized loss | 重要 related work 與 baseline | LPO、DISCo |
| Binary desirable/undesirable alignment | 未配對資料延伸 | KTO |
| Joint functionality-security | 主要安全功能聯合評估 | CWEval |

### 3.1 創新邊界
本研究不應宣稱：
* 發明 Prefix Tuning。
* 發明 PROSEC 資料合成。
* 發明 SimPO。
* 發明 FIM DPO 或 token mask。

可以主張的貢獻是：
1. 將 proactive security alignment 系統化地轉換為 Prefix-only 的可插拔方法。
2. 在相同資料與 objective 下，公平比較 LoRA 和 Prefix 的安全、功能與資源 trade-off。
3. 使用 CWEval 補充 PROSEC 原本分離式安全／功能評估。
4. 分析 pair locality 與 Prefix 容量之間的關係。
5. 若進入第二階段，提出僅對符合條件的 repair pairs 使用 local FIM 的 adaptive Prefix alignment。

若只完成第一階段，適合作為學位論文的低難度增量式研究；若要提高方法創新性，優先完成第 9 節的 adaptive FIM，而不是只增加更多無關 benchmark。

---

## 4. 資料集策略、資料格式與品質檢查

### 4.1 是否重新合成？
第一階段不重新執行完整 PROSEC 合成。使用順序為：
1. 優先取得 PROSEC 公開的最終 $D_{	ext{sec}} \cup D_{	ext{norm}}$。
2. 若只有公開 fix pairs，沿用 $x_v, y_v, y_f$，補齊 normal outputs 與 $D_{	ext{norm}}$。
3. 若缺少 $y_n$，才使用相同目標模型與官方 script 生成 normal code。
4. 不重新付費使用 Claude 合成 vulnerability-inducing instructions。
5. 不更換 seed instruction dataset。
6. 只有新增未提供 target-specific data 的模型時，才重跑 vulnerable generation、fix generation 與 analyzer validation。

如此能確保 LoRA 與 Prefix 的差異來自 PEFT，而不是資料重新生成。

### 4.2 安全資料
$$D_{	ext{sec}} = \{(x_v, y_f, y_v)\}$$

* $x_v$：漏洞誘發指令。
* $y_v$：目標模型產生且被 analyzer 偵測出漏洞的程式碼。
* $y_f$：根據漏洞 feedback 修補，且重新檢測後通過的程式碼。
* Preference：$y_f \succ y_v$。

### 4.3 功能維持資料
$$D_{	ext{norm}} = \{(x_n, y_n, y_f)\}$$

* $x_n$：原本的一般程式設計指令。
* $y_n$：一般情境下的正常回答。
* $y_f$：安全修補版本，但不一定適合一般 prompt。
* Preference：$y_n \succ y_f$。

$D_{	ext{norm}}$ 避免模型無條件加入 sanitization、wrapper、try/except 或只使用特定安全 API。

### 4.4 訓練、驗證與 leakage
* 使用作者提供的 train split；若無 validation split，從 train 依 CWE stratified sampling 建立 5%~10% validation。
* 不使用 PurpleLlama、CWEval、HumanEval 或 MXEval 的測試輸出作 functionality anchor。
* 對 prompt 做 exact match、normalized match 與 fuzzy similarity 檢查。
* 若資料含 benchmark task 或近似改寫，移除後記錄移除數量。
* 固定 dataset revision 與 preprocessing hash。
* 重新組合 FIM middle 後是否可 parse / compile。

edit ratio 定義：
$$
ho_i = rac{	ext{changed tokens between } y_f^{(i)} 	ext{ and } y_v^{(i)}}{\max(|y_f^{(i)}|, |y_v^{(i)}|)}$$

後續將資料分為低、中、高 edit-ratio buckets，分析 localized loss 是否只在高度相似 pair 上有效。

### 4.5 必做資料統計
* $D_{	ext{sec}}$、$D_{	ext{norm}}$ 數量與混合比例。
* CWE 與語言分布。
* chosen/rejected token 長度與長度差。
* AST parse / compile 成功率。
* $y_f$ analyzer pass rate 與 $y_v$ detection rate。
* token-level edit distance、edit ratio 與 edit hunk 數量。
* longest common prefix/suffix 長度。
* 單一 contiguous diff 的比例。
* diff 是否與 analyzer finding 行號重疊。

---

## 5. 模型、程式碼基礎與實驗環境

### 5.1 主模型
使用 PROSEC 的 Phi3-mini-Instruct，並沿用其 config 中的確切 model revision。所有開發、超參數搜尋、ablation 與失敗分析先在此模型完成。

選擇原因：
* 模型較小，成本較低。
* PROSEC 有可比較結果。
* 公開 Phi3 pair data 較容易取得。
* 適合測試 Prefix 是否足以承載安全 alignment。

### 5.2 第二模型
主方法定案後，使用 CodeLlama-7B-Instruct 進行 generalization validation，只跑 Base、ProSec-LoRA 與最佳 ProSec-Prefix / enhanced method，不做完整超參數搜尋。

### 5.3 其他模型
Qwen2.5-Coder-3B 可作額外實驗，但若沒有對應 target-specific data，需重新生成 code pairs。此時應清楚區分：
* target-specific PROSEC training；或
* 使用 Phi3 data 的跨模型 transfer experiment。

兩者不能混為同一設定。

### 5.4 以 PROSEC repo 為修改基礎
保留：
* CWE elicitors 與 instruction data。
* Vulnerable／normal inference scripts。
* PurpleLlama detection 與 repair pipeline。
* Pair construction、mixing、selection 與 evaluation format。

新增：
* `peft_method = lora | prefix`
* Prefix config 與 adapter save/load。
* Experiment config、seed 與 logging。
* CWEval generation adapter。
* Edit-locality preprocessing 與報表。
* 第二階段 FIM dataset builder。

若公開 repo 沒有完整訓練 entry point，可用 Hugging Face TRL 建立最小 training wrapper，但不得改變 prompt formatting、chosen/rejected 欄位與 evaluation protocol。

### 5.5 必須固定的版本
* PROSEC commit hash。
* PurpleLlama submodule commit。
* Dataset revision。
* Base model revision。
* Transformers、PEFT、TRL、PyTorch、CUDA 版本。
* Tokenizer 與 chat template。
* Static analyzer／CWEval Docker image 版本。

---

## 6. 第一階段主方法：Prefix-ProSec

### 6.1 原始 PROSEC
原始 PROSEC 以 LoRA 做 preference optimization，論文設定為 rank $r=8$、$lpha=16$，預設使用完整序列 SimPO。

### 6.2 Prefix-ProSec
凍結 Base Code LLM，只更新插入各 transformer layers 的 continuous prefix：

```text
Frozen Base Code LLM
       +
Trainable Continuous Security Prefix
       |
       v
Secure Code Generation
```

推論時：
* `Base model + security prefix` -> 安全對齊模式
* `Base model without prefix` -> 原始模型模式

這個設計使一份 Base model 可以共用多個小型 safety modules。

### 6.3 SimPO objective
第一階段保留完整序列：

$$\mathcal{L}_{	ext{SimPO}}=-\mathbb{E}_{(x,y_w,y_l)} \left[\log\sigma\left(rac{eta}{|y_w|}\log\pi_{	heta}(y_w|x) -rac{eta}{|y_l|}\log\pi_{	heta}(y_l|x) -\gamma
ight)
ight]$$

$$D = D_{	ext{sec}}^{	ext{sec}} \cup D_{	ext{norm}}^{	ext{norm}}$$ (註：原文依 $D = D_{	ext{sec}} \cup D_{	ext{norm}}$)

第一輪治用 PROSEC 的 $eta=1.5$、$\gamma=0.5$ 與 effective batch size 64。除 PEFT-specific learning rate 外，其他設定盡量一致。

### 6.4 Prefix 設定
初始設定：
```text
task_type            = CAUSAL_LM
num_virtual_tokens   = 20
prefix_projection    = False
```

小型搜尋：
```text
learning rate: 1e-3, 5e-3, 1e-2
prefix length: 10, 20, 40
```

搜尋程序：
1. 固定 20 tokens，先選 learning rate。
2. 固定最佳 learning rate，比較 10／20／40 tokens。
3. 開發使用一個 seed。
4. 最佳設定使用至少三個 seeds。
5. 若 Prefix 明顯 underfit，才測 `prefix_projection=True`，不一開始加入完整 grid。

### 6.5 第一階段能主張什麼？
第一階段不是新 loss，核心問題是：
* Prefix 可否保留 PROSEC 的安全收益？
* 可訓練參數與 checkpoint 能減少多少？
* Prefix 是否比 LoRA 更容易維持 Base model utility？
* Prefix 是否在部分 CWE／語言失敗？
* Prefix 是否適合作為可插拔安全 patch？

---

## 7. Baselines、實驗矩陣、超參數與消融

### 7.1 必做實驗

| ID | 模型 | 資料 | Objective | PEFT | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| E0 | Phi3-mini | 無 | 無 | 無 | Base model |
| E1 | Phi3-mini | PROSEC | SimPO | LoRA | 原始 baseline |
| E2 | Phi3-mini | PROSEC | SimPO | Prefix | 第一階段主方法 |
| E3 | Phi3-mini | 僅 $D_{	ext{sec}}$ | SimPO | Prefix | $D_{	ext{norm}}$ ablation |
| E4 | CodeLlama-7B | 無 | 無 | 無 | 第二模型 base |
| E5 | CodeLlama-7B | PROSEC | SimPO | LoRA | 第二模型 baseline |
| E6 | CodeLlama-7B | PROSEC | SimPO | Prefix | 第二模型驗證 |

### 7.2 Prefix 消融
* Prefix length：10、20、40。
* 有／無 $D_{	ext{norm}}$。
* Random initialization；若 PEFT 版本支援，再比較 near-zero initialization。
* 必要時比較 `prefix_projection=False/True`。
* Seen／unseen CWE。
* Low／medium／high edit ratio。

### 7.3 公平比較
LoRA 與 Prefix 保持：
* 相同 Base model/tokenizer revision。
* 相同資料與 train/validation split。
* 相同 SimPO $eta, \gamma$。
* 相同 effective batch、generation settings 與 evaluator。
* 相同 checkpoint selection criterion。
* 相同 evaluation prompts 與 random seeds。

---

## 8. 執行順序總覽（本節取代 §1.3 的順序建議）

### 8.1 分層原則

實驗依「**是否為論文主張的必要條件**」分成三層，而不是依技術難度或新穎度分層。判準是一句話：
> 如果這一步不做，論文裡的哪一句話會站不住？

| 層級 | 定義 | 內容 | 不做的後果 |
| :--- | :--- | :--- | :--- |
| **L0 診斷層** | 便宜到不做沒道理，且會改變後續所有實驗的地基 | S0、S0.5 | 可能把「學習率設錯」誤判成「Prefix 容量不足」，整個第二階段建立在錯誤結論上 |
| **L1 主線層** | 支撐 RQ1／RQ2／H1／H2 的必要實驗，論文沒有它就不能投 | S1 ~ S3 | 無法回答「Prefix 能否取代 LoRA」 |
| **L2 補強層** | 修復 L1 暴露出的問題，並支撐 H3／H4 | S4、S5 | 論文降級為「單純把 LoRA 換成 Prefix」的最低度增量研究 |
| **L3 泛化層** | 支撐 RQ3 與外部效度 | S6 | 結論只在單一模型、單一 benchmark 上成立 |
| **L4 延伸層** | 有餘力才做，任何一項都不影響主線成立 | §10 全部 | 無 |

### 8.2 順序總表

| # | 步驟 | 層級 | 主要支撐 | 粗估成本 | 前置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | Prefix 學習率校準 | L0 | 全部實驗的地基 | 半天 | — |
| **S0.5** | 生成退化診斷與守門機制 | L0 | 所有 security 數字的效度 | 1 小時 | — |
| **S1** | DPO → SimPO 目標函數切換 | L1 | 與 PROSEC 對齊 | 1 天 | S0 |
| **S2** | E1：LoRA baseline（同管線同資料） | L1 | **RQ1、RQ2** | 1 天 | S1 |
| **S3** | 完整實驗矩陣 + Ablation + 多 seed | L1 | **RQ1、RQ2、RQ4、H1、H2** | 3~5 天 | S2 |
| **S4** | Dnorm influence selection | L2 | **H2**（utility 維持） | 2~3 天 | S3 |
| **S5** | Diff-token masked SimPO + locality 分析 | L2 | **H3、H4、RQ4** | 5~7 天 | S3 |
| **S6** | CodeLlama-7B 泛化 + CWEval | L3 | **RQ3** | 5~7 天 | S5 |

> **關鍵順序原則**：S0 必須在 S1 之前。理由見 §9.1 —— 目前 REPORT 的訓練動態顯示 Prefix 可能處於嚴重 underfit，若不先排除，S2 的 LoRA vs Prefix 比較會系統性地低估 Prefix，而這正是本研究的核心比較。

### 8.3 停損點（每一步的「該不該繼續」判準）

| 步驟 | 若結果為 X，則 | 動作 |
| :--- | :--- | :--- |
| S0 | 調高 lr 後 `rewards/chosen` 仍不上升 | 這是 Prefix 容量問題而非最佳化問題 → H1 需修正措辭，並把 §10.4 的 `prefix_projection` ablation 提前 |
| S0.5 | ON 的生成長度／parse rate 顯著低於 OFF | 現有所有 security 數字都被污染 → 必須先修，不得進 S1 |
| S2 | Prefix 的安全性遠遜於 LoRA（差距 > 10 個百分點） | H1 不成立 → 論文主軸改為「Prefix 的能力邊界分析」，這仍是可發表的負面結果，但要及早轉向 |
| S3 | Prefix utility 掉幅 < LoRA | H2 成立且 S4 可略過 → 這本身是重要發現，直接進 S5 |
| S5 | 各 edit-ratio 分桶下 masked 都沒有增益 | H3 不成立 → 誠實回報，論文停在 L1 + L3 的完整版本 |

---

## 9. 主線步驟詳解

每一步都用同一個格式說明：**為什麼要做（原理）→ 怎麼做（操作）→ 產出什麼 → 通過標準**。

---

### 9.1 S0：Prefix 學習率校準 【L0，最優先】

#### 為什麼要做（原理）

**目前的 `lr = 2e-5` 對 Prefix Tuning 而言低了 2~3 個數量級。**

這不是調參偏好問題，而是梯度尺度問題。LoRA 與 Prefix 的參數進入計算圖的方式根本不同：

* **LoRA**：學到的 `ΔW = BA` 直接加到權重矩陣上，乘在 activation 上，且有 `α/r` 的縮放放大（論文設定 α/r = 16/8 = 2）。梯度沿著整條 residual path 回傳，等效步長被放大。慣用 lr 在 `1e-4 ~ 5e-4`。
* **Prefix**：學到的是每層 attention 的 `past_key_values`，被**串接**在 key/value 序列前面。它只能透過 softmax 的注意力權重間接影響輸出——當前綴只佔 16 個 token、而 context 有數百 token 時，注意力質量分配給前綴的比例很小，回傳到 prefix 的梯度天然就小。這就是為什麼 Li & Liang (2021)、SVEN、PTC 用的 lr 都在 `1e-3 ~ 1e-2`。

換句話說，`2e-5` 是**全量微調**的量級，套在 prefix 上等於幾乎沒在訓練。

#### 支持這個懷疑的既有證據

REPORT §7 的訓練動態（全量 45.8k、1 epoch）：

| 指標 | 早期 | 後期 | 解讀 |
| :--- | :--- | :--- | :--- |
| `rewards/rejected`（漏洞碼） | −1.65 | **−2.8** | 被大幅壓低 ✅ |
| `rewards/chosen`（安全碼） | −1.45 | −1.7 | **幾乎沒動，甚至略降** ⚠️ |

健康的 preference learning 應該讓 margin 由**兩側共同貢獻**：chosen 上升、rejected 下降。目前只有單邊移動，代表模型只學會「壓低某類輸出的機率」，沒學會「提高安全寫法的機率」。這正是階段二 SVEN+DPO 時期遇到的同一個病徵（「Chosen reward 沒有顯著增加」），當時歸因於 DPO 的損失設計，但**在 lr 未校準的前提下，這個歸因是不可靠的**——underfit 與 DPO 病理會產生幾乎相同的曲線。

S0 的目的就是把這兩個原因**分離**。

#### 怎麼做

固定所有其他變數，只掃 lr：

```text
固定：num_virtual_tokens=16, init_weights=zero, epochs=1,
      objective=DPO(beta=0.1)   ← 先不動目標函數，維持與 Exp1 可比
      資料：45.8k 的固定隨機子集 4,000 筆（--max_samples 4000 --seed 42）
掃描：lr ∈ {2e-5(現況), 1e-4, 1e-3, 5e-3}
```

用子集而非全量的理由：4 個 run × 全量 2.5h = 10h，用 4k 子集每個約 15 分鐘，而 underfit 與否在前幾百步的動態上就看得出來，不需要跑完。

每個 run 記錄：`loss`、`rewards/chosen`、`rewards/rejected`、`rewards/margins`、`rewards/accuracies`、`grad_norm` 的完整曲線。

接著對表現最好的 2 組做一次**小規模安全評測**確認訓練動態的改善有反映到真實生成上：

```text
eval/gen_for_icd.py --safecoder_only --limit 100 --num_gen 5 --seed 42
```

#### 產出

* `outputs/lr_sweep/` 下 4 組訓練曲線與比較圖。
* 一個結論句：「Prefix + preference learning 的可用 lr 區間為 ___，後續所有 Prefix 實驗採用 ___。」
* 這張圖本身可以進論文的 implementation details 或 appendix——**「Prefix 的最佳 lr 比 LoRA 高兩個數量級」對想複現本方法的人是有價值的資訊**，而多數 PEFT 論文不會報這個。

#### 通過標準

* `rewards/chosen` 不再單調下降，margin 由雙邊貢獻。
* `grad_norm` 全程穩定（< 50，無尖峰）。
* 小規模安全評測的改善幅度 ≥ Exp1 的同規模結果。

#### 若不通過

代表問題不在最佳化步長，而在 **Prefix 的表達容量**。此時：
1. 把 §10.4 的 `prefix_projection=True` ablation 從延伸層提前到主線（但必須連同參數量代價一起報告，見 §13.1）。
2. 先試 `num_virtual_tokens` 16 → 40 → 64，這比開 MLP 投影便宜得多。
3. H1 的措辭要從「能取得大部分安全性改善」下修為附帶條件的版本。

---

### 9.2 S0.5：生成退化診斷與守門機制 【L0】

#### 為什麼要做（原理）

**靜態分析器對「沒寫的程式碼」一律回報 safe。** 這代表「降低漏洞率」可以被兩種完全不同的機制達成：

1. 真正學會安全寫法（我們要的）
2. 生成更短、更保守、功能不完整的程式碼（偽造的）

而**偏好學習的目標函數本身就會獎勵第 2 種**：

* **SimPO** 的獎勵是 `(β/|y|)·log π(y|x)`——長度歸一化的分母 `|y|` 讓「寫短一點」成為提高平均 log-likelihood 的捷徑。SimPO 原論文 Appendix 也承認這種 alternative implementation 的存在。
* **DPO** 沒有長度歸一化，但有相反方向的已知 length bias。

兩者都會讓「安全性提升」與「功能性下降」在數據上**同時發生且互相解釋**，如果不主動量測，你無法區分。

Exp2 已經有可疑訊號：MultiPL-E C++ 從 46.58 掉到 22.36（**−24.22**）。這種崩幅不太像單純的 overfit，比較像生成退化。

> 注意：§4.5 列的長度統計是針對**訓練資料**（chosen/rejected token 長度差），這裡要做的是針對**生成輸出**。兩者是不同的東西，原計畫沒有涵蓋後者。

#### 怎麼做

新增 `eval/check_degeneration.py`，吃現有的 `outputs/icd_*.on.jsonl` / `.off.jsonl`，對 ON / OFF 兩側各算：

| 指標 | 意義 |
| :--- | :--- |
| 平均 response token 數 | 最直接的退化訊號 |
| 抽取出的 code block 平均字元數 | 排除「話變多但程式碼變少」 |
| AST parse／編譯成功率（依語言） | 語法完整性 |
| 空實作比例 | body 只有 `pass` / `TODO` / `...` / 空 `{}` / 只有 `throw new UnsupportedOperationException` |
| 回絕比例 | 回覆中不含任何 code block（模型拒答或只給說明） |

**先跑在 Exp1 與 Exp2 的既有輸出上**——這是零額外算力的診斷，且能立刻驗證 Exp2 的 C++ 崩盤是不是退化造成的。

#### 產出

* `eval/check_degeneration.py`
* Exp1 / Exp2 的退化報表
* **一條寫進 §7.3 公平比較清單的新規則**（見 §11 修訂表）：任何回報的 security 改善，都必須附帶 ON/OFF 的長度與 parse rate，證明它不是靠「少寫程式碼」換來的。

#### 通過標準（作為之後每次實驗的守門條件）

* ON 的平均 code block 長度 ≥ OFF 的 **90%**
* ON 的 parse 成功率不低於 OFF **3 個百分點**
* ON 的空實作比例不高於 OFF **2 個百分點**

任何一項不通過的實驗，其 security 數字**不得單獨回報**，必須並列退化指標。

---

### 9.3 S1：DPO → SimPO 目標函數切換 【L1】

#### 為什麼要做（原理）

PROSEC 原方法用 SimPO，目前實作用 DPO。這個差異必須消除，否則 §7.3 的公平比較不成立。但更重要的是，SimPO 的三個設計對**本研究的資料特性**有直接好處：

**(1) Reference-free —— 消除一個 Prefix 特有的隱含假設**

DPO 需要 reference model。目前的作法是 TRL 對 PEFT 模型自動用 `disable_adapter` 當 reference，這隱含假設「關閉 prefix adapter 後的前向 == 原始 base model」。對 LoRA 這個假設很乾淨（ΔW 直接不加），對 Prefix 則牽涉到 `past_key_values` 的建構、attention mask 的長度對齊、以及 position id 的偏移是否被正確還原。這個假設多半成立，但它是一個**你必須額外驗證的東西**。

SimPO 不需要 reference 前向，這個假設直接消失，而且**訓練時的前向次數減半**（不用跑 policy + reference 兩次），算力成本明顯下降。

**(2) Length normalization —— 移除 ProSec 資料的一個系統性雜訊**

ProSec 的 `y_f`（修補版）在結構上幾乎必然比 `y_v`（漏洞版）長：加了 sanitize、加了輸入驗證、加了 try/except、換用更冗長的安全 API。這代表在 Dsec 上，「chosen 比較長」是一個**與安全性高度相關但本身無意義的捷徑特徵**。

DPO 的 loss 是未歸一化的序列 log-likelihood 加總，模型可以透過「學會生成更長的東西」來部分滿足偏好，而不真的學會安全語意。SimPO 除以 `|y|` 之後，這條捷徑被切斷。

（注意這與 S0.5 的風險是**同一個機制的兩個方向**：length norm 切斷了「變長」的捷徑，但同時開啟了「變短」的捷徑。所以 S0.5 的守門機制在切到 SimPO 之後更重要，不是更不重要。）

**(3) Target reward margin γ —— 避免過早停止推動**

SimPO 的 loss 內含 `−γ`，要求 margin 必須超過 γ 才視為滿足。Exp1 的 `rewards/margins` 到 1.1 附近就趨緩，DPO 沒有機制強迫它繼續推。γ 提供這個機制。

#### 怎麼做

**實作重點：TRL 0.12.2 的 SimPO 不在 `DPOTrainer` 裡，在 `CPOTrainer` 裡。**

已驗證 `CPOConfig` 具備 `loss_type` 與 `simpo_gamma` 兩個欄位：

```python
from trl import CPOConfig, CPOTrainer
cfg = CPOConfig(
    loss_type="simpo",
    simpo_gamma=0.5,   # PROSEC 論文設定
    beta=1.5,          # PROSEC 論文設定
    ...
)
```

對 `train_prefix.py` 的改動：

```text
新增 --objective {dpo, simpo}   預設 simpo
  dpo   → DPOTrainer   + DPOConfig(beta=args.beta)
  simpo → CPOTrainer   + CPOConfig(loss_type="simpo",
                                   beta=args.beta, simpo_gamma=args.gamma)
新增 --gamma  預設 0.5
```

兩條路徑共用同一個 `build_dataset`、同一個 peft config 建構、同一套 logging，只有 trainer 類別不同。

#### 三個必須注意的陷阱

1. **β 的語意完全不同**，不能沿用調參經驗。DPO 的 `β=0.1` 控制的是與 reference 的 KL 強度；SimPO 的 `β=1.5` 是縮放 length-normalized reward 的溫度。兩者不在同一個尺度上，`β=0.1` 的 SimPO 幾乎等於沒有訓練訊號。**第一輪嚴格用論文的 β=1.5、γ=0.5。**
2. **lr 用 S0 的結果**，不要沿用 2e-5。
3. **`epochs` 固定為 1。** Exp2 已經證明 epochs=2 會 over-train（REPORT §12.3），而且那次還跟 β 混淆了。這是計畫裡的一條硬規則：**一次只改一個變數。**

#### 產出與通過標準

* `train_prefix.py` 支援 `--objective`
* 一組 SimPO + Prefix 的訓練曲線，與 S0 最佳 DPO 曲線並列
* **通過標準**：訓練穩定（grad_norm 無爆炸）、`rewards/accuracies` 收斂到 > 0.75、通過 S0.5 的退化守門

---

### 9.4 S2：E1 — LoRA baseline（同管線、同資料）【L1，論文成敗關鍵】

#### 為什麼要做（原理）

**這一步是整篇論文能不能成立的關鍵，而它的必要性不是「復現論文」。**

RQ1 問的是「Prefix 能否接近 LoRA」，這是一個**相對**主張。要回答它，你需要兩個在同一條件下測出來的數字。目前 REPORT §10.3 的比較是拿自己的 Prefix 數字去對論文 Table 1 的 LoRA 數字，這個比較**在方法學上無效**，因為地基有三處不同：

| 差異 | 我方 | 論文 |
| :--- | :--- | :--- |
| base 模型的 OFF 表現 | 45.21% | 50.57% |
| 訓練資料 | 未篩選的 45.8k mixed pool | §3.3 influence-selected ~10k |
| 目標函數 | DPO | SimPO |

在三個變數同時不同的情況下，任何 Δ 都無法歸因。**S2 的作用是把這三個變數全部消除，讓 LoRA 與 Prefix 唯一的差異是 PEFT 方法本身。**

#### 怎麼做

嚴格鎖定的變數（擴充 §7.3）：

```text
相同：base model + revision + tokenizer + chat template
相同：訓練資料 = 同一份 45,785 pool（S4 之前不做 selection）
相同：train/validation split（同一個 stratified split 檔案）
相同：objective = SimPO, beta=1.5, gamma=0.5
相同：epochs=1, effective batch size=64, seed
相同：checkpoint selection criterion（validation SimPO loss 最低）
相同：評測腳本、生成參數（temperature/top_p/num_gen）、評測 seed
唯一變數：peft_method ∈ {lora, prefix}
```

**LoRA 設定**：`r=8, α=16`（PROSEC 論文設定），target modules 為 attention 的 `q,k,v,o` 投影。

#### 一個容易搞錯的公平性問題

**「相同的學習率」對不同 PEFT 方法不是公平，而是不公平。**

如 §9.1 所述，LoRA 與 Prefix 的梯度尺度差兩個數量級。強迫它們用同一個 lr，等於故意讓其中一方 underfit。**公平的定義是「各自在同等預算下調到最佳」**：

```text
LoRA   lr 搜尋：1e-4, 3e-4, 5e-4   （3 組，各 1 epoch 子集）
Prefix lr 搜尋：沿用 S0 的結果      （已完成，同樣 3~4 組）
```

兩邊的搜尋預算相同（同樣數量的 run、同樣的子集大小），這才是可辯護的公平。

> §7.3 現在寫的是「除 PEFT-specific learning rate 外，其他設定盡量一致」——方向正確但太模糊，要明確化成上面這個「同等搜尋預算」的定義，否則審稿人會問。

#### 額外做一次 sanity check（不是主結果）

拿 S2 的 LoRA + SimPO 結果去對 PROSEC 論文 Table 1 的 Phi-3 數字。**因為資料不同（未做 selection），不要求數字相同**，只檢查：

* 方向一致（漏洞率下降）
* 量級合理（下降幅度在論文的同一個數量級內）

若方向或量級明顯不合（例如漏洞率反而上升），代表管線有 bug，必須先排查再繼續。若合理，就在論文裡寫一句「我們的 LoRA 重現結果與原論文趨勢一致，差異可歸因於未套用 §3.3 的 influence selection」。

#### 產出

* `train_prefix.py` 支援 `--peft_method {lora, prefix}`
* E1 完整結果（security 5 語言 + HumanEval + MultiPL-E + 退化指標）
* **RQ2 的資源對照表**（見下）

#### RQ2 的量測必須在訓練當下記錄

這些數字事後補很痛苦，要在 trainer callback 裡直接寫出來：

| 指標 | 怎麼取得 |
| :--- | :--- |
| trainable parameters | `model.print_trainable_parameters()` |
| checkpoint / adapter 磁碟大小 | `du -sb` 存檔目錄 |
| peak GPU memory | `torch.cuda.max_memory_allocated()` |
| optimizer state 大小 | trainable params × 8 bytes（AdamW 的 m, v，fp32） |
| wall-clock GPU-hours | trainer 的 `train_runtime` |
| 推論額外開銷 | Prefix 會增加 `num_virtual_tokens` 的 KV cache；LoRA 若不 merge 則多一次矩陣乘 |

**已算好的參數量參考（Phi-3-mini：32 層、hidden 3072）**：

| 設定 | 訓練期可訓練參數 |
| :--- | :--- |
| Prefix, `prefix_projection=False`, nvt=16（現況） | **3.15 M** |
| Prefix, `prefix_projection=False`, nvt=20 | **3.93 M** |
| Prefix, `prefix_projection=True`, nvt=20 | **613.48 M** ⚠️ |
| LoRA r=8 (q,k,v,o) | **6.29 M** |

即 Prefix ≈ LoRA 的 **1/2**。這個倍率是誠實的數字，不要誇大——H1 的賣點不該只是參數量，而是「**adapter 極小 + 可插拔 + 一個 base 可掛多組安全策略**」的組合（見 §10.2）。

---

### 9.5 S3：完整實驗矩陣 + Ablation + 多 seed 【L1】

#### 為什麼要做

S2 只給了 E1 vs E2 的單點比較。S3 把它擴成可發表的完整證據：ablation 回答「為什麼有效」，多 seed 回答「是不是雜訊」，分層分析回答 RQ4。

#### 9.5.1 先補一個目前完全沒做的東西：validation split

§4.4 要求做，但目前的流程是「訓練完直接評測」，**沒有 checkpoint selection**。這有兩個問題：

1. §7.3 要求 LoRA 與 Prefix 用「相同 checkpoint selection criterion」——沒有 criterion 就無法滿足。
2. 沒有 validation 就無法偵測 over-train。Exp2 的 epochs=2 崩盤如果有 validation curve，當場就會看出來，不必等到跑完整套評測。

**做法**：從 45,785 筆按 `(lang, cwe)` 做 stratified sampling 抽 **5%**（約 2,290 筆）作 validation，其餘 43,495 筆訓練。split 存成固定檔案（`data/split_seed42.json`），所有實驗共用。

**checkpoint selection criterion**：validation 上的 SimPO loss 最低。每 N steps 評估一次。

#### 9.5.2 順便補 leakage 檢查

§4.4 要求但未執行。針對 HumanEval / MultiPL-E / PurpleLlama 的 prompt，對訓練資料的 `original_instruction` 做：

* exact match
* normalized match（去空白、統一大小寫、去標點）
* fuzzy similarity（例如 token-level Jaccard > 0.8）

命中的移除並**記錄移除數量**。這個數字要寫進論文——就算是 0，「我們檢查過且為 0」跟「我們沒檢查」在審稿人眼中是兩件事。

#### 9.5.3 實驗矩陣（擴充 §7.1）

| ID | 模型 | 資料 | Objective | PEFT | 用途 | 層級 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| E0 | Phi3-mini | — | — | — | Base（OFF） | L1 |
| E1 | Phi3-mini | 43.5k pool | SimPO | LoRA r=8 | **主 baseline** | L1 |
| E2 | Phi3-mini | 43.5k pool | SimPO | Prefix nvt=16 | **第一階段主方法** | L1 |
| E3 | Phi3-mini | 僅 Dsec | SimPO | Prefix | Dnorm ablation | L1 |
| E2a/b/c | Phi3-mini | 43.5k pool | SimPO | Prefix nvt=10/20/40 | 容量 ablation | L1 |
| E2-init | Phi3-mini | 43.5k pool | SimPO | Prefix, 隨機 init | 初始化 ablation | L1 |
| E2-dpo | Phi3-mini | 43.5k pool | DPO | Prefix | 目標函數 ablation | L1 |

#### 9.5.4 每個 ablation 回答的問題

| Ablation | 回答什麼 | 為什麼重要 |
| :--- | :--- | :--- |
| **無 Dnorm（E3）** | Dnorm 對 utility 維持貢獻多少 | 若 E3 的 utility 明顯更差，證明 ProSec 的 Dnorm 機制在 Prefix 上同樣有效；若差不多，代表 Prefix 因容量小天然就不會過度安全化——**這本身是支持 H2 的有趣發現** |
| **nvt 10/20/40** | Prefix 容量與安全性的關係 | 直接量測 H1 的邊界。若 40 明顯優於 16，代表 Exp1/Exp2 的容量設太小；若持平，代表容量不是瓶頸，可以放心主打參數效率 |
| **隨機 vs zero init** | zero-init 是必要還是可有可無 | REPORT §6 是本研究目前最紮實的工程發現，但要注意措辭（見 §13.2） |
| **DPO vs SimPO** | 目標函數貢獻多少 | 讓 REPORT 既有的 DPO 實驗不白做，變成正式 ablation 的一格 |

#### 9.5.5 多 seed

依 §6.4：開發階段用單一 seed，**最終回報的設定（E1、E2）必須用至少 3 個 seed**，回報 mean ± std。

理由不只是嚴謹：目前 Exp1 與 Exp2 的 **OFF** 數字就不一致（45.21 vs 44.79；pooled 41.47 vs 41.66）。這是因為 `gen_for_icd.py` 用 `temperature=0.8` 取樣且**沒有 seed 控制**（見 §13.3）。在改善幅度只有 4~8 個百分點的情境下，這種漂移不是可以忽略的。

#### 9.5.6 RQ4 的分層分析

在 E2 的結果上，把每個測試案例依下列維度分組，看 Δ 漏洞率：

* **CWE**（seen in training vs unseen）
* **語言**（5 種）
* **訓練資料中該 (lang, cwe) 的樣本數**（檢查是否只是資料量效應）
* **edit ratio 分桶**（低／中／高，定義見 §4.4）——這一項是 S5 的前置

#### 產出

* 完整的主結果表（E0~E3 × security 5 語言 × utility 3 benchmark × 退化指標）
* RQ2 的資源對照表
* RQ4 的分層分析圖表
* **論文第一階段可以寫完了**

---

### 9.6 S4：Dnorm influence selection 【L2，條件執行】

#### 為什麼要做（原理）

**這一步是修復，不是創新。** 它的目的是把 utility 掉幅拉回來。

REPORT §4.3 已經考證清楚：目前用的 `prosec-mixed-phi3mini-4k-inst`（45,785 筆）是 ProSec **§3.2 的混合候選池**，不是 §3.3 influence selection 之後的最終訓練集（論文約 10k）。論文釋出的 selected 版本（`prosecalign/top2-cds-0.8-kendall-…`）是空的 placeholder。

**原理**：Dnorm 的功能是教模型「在一般任務上不要過度安全化」。但候選池裡的 Dnorm pair 品質參差——有些 pair 的 `y_n` 與 `y_f` 差異很小（沒有偏好訊號），有些甚至方向相反（`y_f` 其實比較好）。這些低品質 pair 會注入雜訊梯度，直接傷害 utility。

ProSec §3.3 用 **training dynamics 為基礎的影響力分數**篩選：追蹤每個樣本在訓練過程中的 loss 軌跡，估計它對目標指標的邊際影響，取 top-2 並丟掉 bottom 20%。

Exp1 的 HumanEval **−3.05**、Exp2 的 **−11.59** 很可能主要來自這裡。

#### 為什麼排在 S3 之後而不是之前

三個理由：

1. **它不是本研究的貢獻**，是 ProSec 的既有機制。優先做自己的主線比較。
2. **有可能不需要做**。H2 的一個潛在結論是「Prefix 因為容量小，天然就比 LoRA 更不容易破壞 base model 的能力」。若 S3 顯示 Prefix 的 utility 掉幅本來就小於 LoRA，那 selection 對 Prefix 的必要性就低——**這個「不需要」本身就是支持 H2 的證據，比做了 selection 更有價值**。
3. 若在 S3 之前做，S2 的 LoRA 對照就要一起重跑，成本翻倍。

#### 條件

**只有當 S3 顯示 E2 的 utility 掉幅仍 > 3 個百分點時才執行。**

#### 怎麼做

1. 先確認 ProSec repo 是否有對應的 selection script（`src/` 下）。有就直接沿用，這是最忠實的作法。
2. 若沒有，用 gradient-based influence 近似：訓練時記錄每個樣本的 per-sample loss 軌跡，用軌跡的變異與下降幅度估計影響力。這是近似而非原方法，論文中要**明確標註為近似實作**。
3. 篩選後的資料集大小應落在論文提到的量級（約 10k，即「7× SafeCoder」）。若差太多，代表實作有偏差。
4. **selection 必須在 LoRA 與 Prefix 上都重跑**，否則又破壞 §7.3 的公平比較。

#### 產出

* `data/influence_select.py`
* 篩選後資料集的統計（大小、CWE/語言分布變化、被丟掉的 pair 有什麼共同特徵）
* E1'/E2'（selected 資料版）的完整結果
* 論文中的一段：「influence selection 對 LoRA 與 Prefix 的效果是否對稱」——若 selection 對 Prefix 幫助較小，正好呼應 H2

---

### 9.7 S5：Diff-token masked SimPO + Locality 分析 【L2，本研究最有創新性的一段】

> **重要定位修正**：原 §1.2 把第二層稱為「FIM-localized SimPO」。**本節將主線改為 diff-token masking，FIM 格式降級為 §10.5 的一個 ablation。** 理由見下方「為什麼不用 FIM」。

#### 9.7.1 為什麼要做（原理）

**問題：完整序列的 preference loss 把大部分梯度花在與安全無關的 token 上。**

ProSec 的 `(y_f, y_v)` 是**同一個 prompt 下的修補前後版本**，兩者共享大量完全相同的 token——常常只有 1~3 行不同（換一個 API、加一個驗證、改一個參數）。但完整序列 SimPO 對兩邊的**每一個 token** 都算 log-likelihood 並相減。

理論上相同的 token 應該互相抵消，但實際上抵消不完全，原因有二：

1. **前文 context 不同**。就算第 `t` 個 token 相同，它在 `y_f` 與 `y_v` 中的前綴不同（因為前面已經有 diff），所以 `log π(t | y_f_{<t})` ≠ `log π(t | y_v_{<t})`。差值不為零，但這個差值**不承載安全語意**。
2. **長度歸一化的分母不同**。SimPO 除的是 `|y_w|` 與 `|y_l|`，而 `|y_f| ≠ |y_v|`（修補版通常較長）。即使分子的相同部分完全相等，除以不同分母後也不會抵消。

結果就是：**大量梯度預算被花在懲罰／獎勵完全正確的通用程式碼上**。這正是階段二 SVEN+DPO 時期診斷出的「後綴懲罰」問題，換到 ProSec 資料上並沒有消失——ProSec 的 pair 相似度只會更高，問題只會更嚴重。

這也提供了 Exp2 utility 崩盤的**機制解釋**：多訓練一個 epoch，等於讓這些無關的梯度多累積一輪，於是安全性微幅上升、通用能力大幅下降。β 與 epochs 這種鈍器只能在這條曲線上移動工作點，**不能改變曲線本身的形狀**。masking 才是結構性的修正。

#### 9.7.2 為什麼這對 Prefix 特別重要（H3 的理論根據）

這是本研究**最值得投資的一句話**：

> Prefix 只有約 3.15 M 可訓練參數（LoRA 的一半），而且只能透過 attention 的 softmax 權重間接影響輸出。當可訓練容量與梯度預算都有限時，把梯度**集中**在真正決定安全性的 token 上，邊際效益應該高於容量較大的 LoRA。

換句話說，**localized loss 與 Prefix 是互補的**：Prefix 的弱點（容量小）正是 localized loss 的強項（梯度集中）能補的。這個論點把「Prefix + PROSEC」從單純的方法替換，提升成一個有理論動機的組合——這是把研究從「最低度增量」拉到「有貢獻」的關鍵。

#### 9.7.3 為什麼不用 FIM

原計畫想用 FIM (Fill-In-the-Middle) 格式來限制損失範圍。改用 diff-token masking 的三個理由：

| 面向 | FIM 格式 | Diff-token masking |
| :--- | :--- | :--- |
| **模型依賴** | 需要模型原生支援 `<PRE>/<SUF>/<MID>`。Phi-3-mini-4k-instruct **不是 FIM-trained**；且 StructureCoder 指出經過 RLHF 的 instruct 模型對 FIM token 的注意力分布會退化 | 完全不依賴。任何 causal LM 都能用 |
| **評測一致性** | 訓練用 FIM 格式、評測用 chat 格式 → 訓練/推論分布不匹配，違反 §7.3 | prompt 格式完全不變，訓練與評測都用 chat template |
| **實作複雜度** | 要重建資料集、處理特殊 token、驗證重組後可 parse（§4.4 已列這些檢查） | 只是在既有 loss 上乘一個 mask 張量 |

FIM 仍值得做，但作為 ablation（§10.5），用來回答「格式化 vs 遮罩，哪種 locality 注入方式較好」。

#### 9.7.4 前置：Edit-locality 分析（必須先做）

**在寫任何 masking 程式碼之前，先確認資料真的是局部的。** 若 ProSec 的 pair 其實 edit ratio 很高，masking 就沒有意義，H3 直接不成立——這個檢查只要幾小時，卻能省下一週的實作。

對全部 27,400 筆 Dsec 計算（§4.5 已列，這裡具體化）：

| 統計量 | 定義 | 用途 |
| :--- | :--- | :--- |
| edit ratio ρ | `changed_tokens / max(|y_f|, |y_v|)`（§4.4 公式） | 主要分桶依據 |
| edit hunk 數 | diff 中連續變更區塊的數量 | 判斷是單點修補還是散佈修改 |
| 單一 contiguous diff 比例 | hunk 數 == 1 的樣本佔比 | 若很高，masking 極可能有效 |
| longest common prefix / suffix 長度 | 兩序列頭尾相同的部分 | 直接量化「浪費的梯度」有多少 |
| diff 與 analyzer finding 行號重疊率 | 變更區域是否真的落在漏洞位置 | **驗證 mask 抓到的是安全相關的 token，而不是格式差異** |

最後一項最重要：如果 diff 與 analyzer 標記的漏洞行大量重疊，代表 mask 抓對了東西，H3 有依據；若不重疊，代表 `y_f` 與 `y_v` 的差異大多是無關的重寫，masking 會抓到雜訊。

**分桶**：依 ρ 切成低／中／高三桶（切點由實際分布決定，不要事先猜）。

#### 9.7.5 實作

**Token-level diff**：對 `y_f` 與 `y_v` 的 token 序列跑 `difflib.SequenceMatcher`（或 Myers diff），取得 opcodes。

**Mask 定義**：
```text
對 chosen (y_f)：m_t = 1  若 token t 屬於 opcode 為 'insert' 或 'replace' 的區段
對 rejected (y_v)：m_t = 1  若 token t 屬於 opcode 為 'delete' 或 'replace' 的區段
'equal' 區段：m_t = 0（不進 loss）
```

**改寫後的 SimPO reward**（長度歸一化的分母同步改成 mask 內的 token 數）：
```text
r(x, y) = (β / Σ_t m_t) · Σ_t m_t · log π_θ(y_t | x, y_<t)
```

注意 `π_θ(y_t | x, y_<t)` 的**條件仍然是完整的前文**——我們只是不對某些位置的 log-prob 計入 loss，而不是把那些 token 從序列裡刪掉。前向照常跑完整序列，只在 loss 聚合時套 mask。這一點如果搞錯，模型會看到不連續的 context，結果會壞掉。

**邊界處理（重要）**：
* 若某個 pair 的 mask 全為 0（`y_f` 與 `y_v` token 序列完全相同），該樣本必須丟棄。`convert_prosec_to_pref.py` 目前只做字串層級的 `strip()` 比較，token 化後仍可能相同。
* 若 mask 內的 token 數過少（例如 < 3），length normalization 的分母太小會讓 reward 方差爆炸，需要設下限或改用平滑。
* **Dnorm 要不要套 mask 需要單獨決定**。Dnorm 的偏好是「正常寫法 ≻ 過度安全化」，它的 diff 語意跟 Dsec 相反。第一版建議 **Dnorm 維持完整序列，只對 Dsec 套 mask**，並把「Dnorm 也套 mask」列為 ablation。

#### 9.7.6 Adaptive full/local（H4）

§2.2 的 H4 明確指出 localized loss 不會對所有資料有效：大範圍、多 hunk、有長距離依賴的修補仍適合完整序列。所以最終方法應該是**自適應**的：

```text
若 ρ_i < threshold 且 hunk 數 ≤ k：使用 masked SimPO
否則：使用完整序列 SimPO
```

**threshold 與 k 由 §9.7.4 的分桶結果決定，不要事先設定。** 正確的做法是先跑「全部 masked」與「全部 full」兩個版本，看各分桶的表現差異，再從資料反推門檻——若表現差異在分桶間是單調的，門檻就落在交叉點。

#### 9.7.7 產出

* `data/edit_locality.py`（統計 + 分桶）
* `data/build_masks.py`（token-level diff → mask）
* `train_prefix.py` 支援 `--loss_scope {full, masked, adaptive}`
* 三組對照：full / masked / adaptive × {Prefix, LoRA}
* **H3 的直接證據**：masked 對 Prefix 的增益 vs 對 LoRA 的增益。若 Prefix 的增益顯著較大，H3 成立，這是本論文最強的一張圖。

#### 9.7.8 通過標準

* masked 版本在**不損失安全性**的前提下，utility 掉幅小於 full 版本；或
* 在**相同 utility** 下，安全性優於 full 版本

（兩者滿足其一即可——這是一個 trade-off 曲線的比較，不是單點比較。建議畫成 security-utility 的 Pareto 圖。）

---

### 9.8 S6：泛化驗證 【L3】

#### 9.8.1 CodeLlama-7B（§5.2）

**目的**：證明結論不是 Phi-3 特有的。

**只跑 E4（base）、E5（LoRA）、E6（Prefix，最佳設定）**，不做超參數搜尋——超參數沿用 Phi-3 的結論，只按模型規模調整（7B 的 lr 通常略低於 3.8B）。

**資料捷徑**：REPORT §4.2 已考證出 `prosecalign/prosec-mixed-clm7b-inst`（61.2k）是用**相同的 instruction 集**但由 CodeLlama 生成程式碼的版本。直接用它，不需要自己重跑生成，這省下大量成本。

**注意架構差異**：CodeLlama-7B 用 GQA（Phi-3-mini 是 MHA）。Prefix 的 `past_key_values` 維度要依 `num_key_value_heads` 而非 `num_attention_heads` 計算。PEFT 應該會處理，但**必須先用 `poc_prefix_smoke.py` 驗證前向與生成正常**，再開始訓練——這正是該腳本存在的目的。

#### 9.8.2 CWEval（RQ3）

**為什麼需要**：目前的評測是**分離**的——PurpleLlama 量安全、HumanEval/MultiPL-E 量功能。這無法回答「同一段程式碼是否既正確又安全」。CWEval 的 `func-sec@k` 同時要求通過功能測試與安全檢查，是 RQ3 的唯一支撐。

它也是 §9.2 退化守門的**終極版本**：如果模型靠寫短程式碼來降低漏洞率，`func-sec@k` 會直接崩，因為功能測試會失敗。

**為什麼排最後**：需要 Docker 環境與可執行的測試框架，整合成本高。在主線跑通之前不要開這個坑。

**產出**：E0 / E1 / E2（最佳設定）在 CWEval 上的 `func@k` 與 `func-sec@k` 對照。

---

## 10. 延伸層（L4）：有餘力才做，不影響主線成立

以下每一項都是**獨立可選**的。任何一項不做，論文的主張都不受影響。依「投資報酬率」排序。

### 10.1 CWE-specific 多 Prefix 【推薦度：高】

**這是 Prefix 相對 LoRA 最有結構性的優勢，也是目前計畫最被低估的一張牌。**

一個 base model 可以同時掛載多組 prefix（每組僅約 3 MB），依部署環境的安全政策切換：

```text
base + prefix_web      → 強化 XSS / CSRF / SQLi（CWE-79, 352, 89）
base + prefix_crypto   → 強化弱加密 / 隨機性（CWE-327, 338, 916）
base + prefix_memory   → 強化記憶體安全（CWE-119, 125, 787, 416）
base + 無 prefix       → 原始模型
```

LoRA 也能做多 adapter，但每組 6.29 M 且需要 merge/unmerge，切換成本較高。Prefix 只是換一組 `past_key_values`，**切換成本趨近於零，且可以在同一個 batch 內對不同請求用不同 prefix**。

這直接呼應 SVEN 的原始動機（用 prefix 控制生成屬性），也讓 §6.2 的「可插拔安全模組」從口號變成可量測的實驗。

**做法**：把 Dsec 依 CWE 分組，各訓一組 prefix，測「專用 prefix 在自己的 CWE 上 vs 通用 prefix」的表現差異，以及是否有跨 CWE 的負遷移。

### 10.2 Prefix 組合與插拔行為分析 【推薦度：中高】

延續 10.1：兩組 prefix 能不能疊加？（拼接 KV，或加權平均）安全性會不會相加？utility 會不會相乘惡化？

這類「模組化組合」的分析在 LoRA 上有大量文獻，在 Prefix + 安全性的設定下相對空白。

### 10.3 KTO 【推薦度：低（第一階段）】

§1.3 已經正確判斷：ProSec 已提供成對資料，KTO 的 binary desirable/undesirable 設定沒有用武之地。

**只有在**後續引入大量**未配對**的資料時才值得——例如直接拿 SAST 掃描結果標記的單邊樣本（只知道「這段有漏洞」但沒有對應的修補版）。若真的走到這一步，KTO 才會是必要的而非可選的。

### 10.4 `prefix_projection=True` ablation 【推薦度：中，且必須附帶代價】

Li & Liang (2021) 提出用 MLP 投影穩定訓練，訓練後丟棄 MLP。這在文獻上是標準技巧，但**在本研究的脈絡下有嚴重代價**：

| 設定 | 訓練期可訓練參數 | 相對 LoRA |
| :--- | :--- | :--- |
| `prefix_projection=False`（nvt=20） | 3.93 M | **0.6×** |
| `prefix_projection=True`（nvt=20） | **613.48 M** | **97.5×** |

開啟後訓練期參數量是 LoRA 的近 **100 倍**，**RQ2 的 optimizer memory 與 GPU-hours 兩個指標直接報廢**（推論期只留 3.93 M 的 KV，但 RQ2 明列了訓練資源）。

此外 PEFT 0.19.1 的 `load_prompt_embeddings()` 在 `prefix_projection=True` 時會直接 `raise ValueError`（因為無法反推投影），**adapter 的可攜性受損**，這正好打到「可插拔安全模組」的賣點。

**結論**：維持 §6.4 的順序——**預設 `False`**，只有 S0/S3 確認 underfit 之後才做這個 ablation，且報告時必須連同參數量代價一起呈現。**不應該把它當預設值。**

（穩定性問題已經由 zero-init 解決，見 §13.2 —— 這是本研究採用的、成本為零的替代方案。）

### 10.5 FIM 格式 ablation 【推薦度：中】

S5 改用 diff-token masking 之後，FIM 變成「另一種 locality 注入方式」的對照組。值得做一次，回答「格式化 vs 遮罩，哪種較好」，也讓計畫中對 FIM 文獻的引用有實質對應。

**前提**：先驗證目標模型是否真的能解析 FIM token。Phi-3-mini-4k-instruct 很可能不行，此時這個 ablation 就變成「為什麼我們沒用 FIM」的證據，同樣有價值。

### 10.6 Verifier-based weighting 【推薦度：低】

用 analyzer 的信心分數或 CWE 嚴重度對 loss 加權。概念簡單，但引入額外超參數，且與 S5 的 masking 在功能上部分重疊。排在最後。

### 10.7 Qwen2.5-Coder-3B 【推薦度：低】

§5.3 已正確指出：沒有對應的 target-specific data。若要做，必須清楚區分「target-specific 訓練」與「用 Phi-3 資料的跨模型 transfer」——**後者其實更有趣**（Prefix 學到的安全知識能不能跨模型？），但也更容易得到負面結果，且不影響主線。

---

## 11. 對本文件前面章節的修訂決議

以下修訂是本次規劃的結論，**優先於前面章節的原始描述**。

| 章節 | 原內容 | 修訂後 | 理由 |
| :--- | :--- | :--- | :--- |
| **§1.2 第 2 層** | 「Locality-Aware Prefix-ProSec：選擇完整序列或 **FIM-localized** SimPO」 | 「Locality-Aware Prefix-ProSec：選擇完整序列或 **diff-token masked** SimPO」 | Phi-3-mini 非 FIM-trained；FIM 會造成訓練/評測格式不一致。詳見 §9.7.3 |
| **§1.3 順序** | 1. 重現 PROSEC-LoRA → 2. 完成 Prefix-ProSec → … | **由 §8.2 的 S0~S6 取代** | 必須先做 lr 校準（S0）與退化診斷（S0.5），否則後續比較的地基不可靠 |
| **§1.3 「重現 PROSEC-LoRA」** | 措辭像是要復現論文數字 | 明確定位為「**同管線 LoRA baseline**（S2）」，不是復現論文 table | 資料未做 selection、base OFF 不同，數字本來就不會相同；目的是取得可比較的對照，不是重現數字。詳見 §9.4 |
| **§3 元件表「FIM localized preference」** | 第二階段主線 | 降為 ablation（§10.5）；主線改為 **LPO / DISCo 的 diff-token localized loss** | 同上。這也讓 LPO/DISCo 從 related work 升格為直系方法來源 |
| **§3.1 「第 9 節的 adaptive FIM」** | 指向不存在的章節 | 指向 **§9.7（S5）的 adaptive full/local masked SimPO** | 原文件在 §7.3 就中止，§8 以後從未寫成 |
| **§4.5 資料統計** | 只涵蓋**訓練資料**的長度統計 | 新增**生成輸出**的退化指標（§9.2） | 兩者是不同的東西；靜態分析器對空程式碼一律回報 safe，不量測就無法排除偽造的安全性 |
| **§6.4 Prefix 設定** | `learning rate: 1e-3, 5e-3, 1e-2` | 保持不變，但**提升為 S0 的最優先任務**，並加入 2e-5 與 1e-4 作為對照下界 | 現行實作用的 2e-5 落在這個範圍之外兩個數量級，必須先確認它是不是既有結果的主要瓶頸 |
| **§6.4 `prefix_projection`** | 「若明顯 underfit，才測 `True`」 | **維持此判斷**，並補上參數量代價（613 M vs 3.93 M）與 adapter 可攜性問題 | 有外部建議主張預設開啟，本計畫**明確拒絕**。詳見 §10.4、§13.1 |
| **§7.1 實驗矩陣** | E1/E2 只標「PROSEC 資料」 | 明確標註「**同一份 43.5k pool（S4 之前不做 selection）**」，並擴充為 §9.5.3 的完整矩陣 | 不寫死資料版本，RQ1 從一開始就無解 |
| **§7.3 公平比較** | 「除 PEFT-specific learning rate 外，其他設定盡量一致」 | 明確化為「**各自在同等搜尋預算下調到最佳**」（同樣數量的 run、同樣的子集大小） | 強迫兩種 PEFT 用同一個 lr 不是公平，是讓其中一方 underfit。詳見 §9.4 |
| **§7.3 公平比較** | （無） | **新增一條**：任何回報的 security 改善必須附帶 ON/OFF 的生成長度與 parse rate | 見 §9.2 退化守門 |
| **§7.3 相同 random seeds** | 已列為要求 | **目前的程式碼做不到**，需先修 `gen_for_icd.py` | 見 §13.3 |

---

## 12. 既有資產盤點（沿用 / 修改 / 作廢）

### 12.1 直接沿用 —— 最貴的資產，不要動

| 項目 | 為什麼有價值 |
| :--- | :--- |
| `eval/gen_for_icd.py` 的 `SAFECODER_CWES` 子集重建 | 論文說「38 對 / 694 題」但**沒有釋出這個子集**。自行重建出 36 對 / 693 題，這是原創工作，論文可寫 |
| `eval/normalize_responses.py` | 修掉 fence-less 抽取的**真 bug**（84% code block 為空、漏洞率被稀釋約 6 倍）。修完之後 base OFF 才與論文對齊（JS 52.24 vs 52.24）——**這是整個評測效度的基礎** |
| `eval/score_detected.py` / `run_humaneval.py` / `run_multipl_e.py` | 評測管線完整可用 |
| `data/convert_prosec_to_pref.py` | 資料轉換正確，含 Dnorm 欄位已預先對調的處理 |
| `poc_prefix_smoke.py` | 架構驗證。S6 換 CodeLlama（GQA）時**必定會再用到** |
| zero-init 的發現（REPORT §6） | 工程貢獻，論文可寫（措辭需調整，見 §13.2） |
| REPORT §4.2 的資料血緣考證 | 用 100% exact-match 驗證 Dsec/Dnorm 的來源與生成方式。品質很高，可直接改寫進論文 §4 |
| REPORT §4.3 的 selection 狀態考證 | 確認手上是 §3.2 候選池而非 §3.3 最終集。這是 S4 的依據，也是 utility 掉幅的解釋 |

### 12.2 需要修改 —— 不是重來

| 項目 | 改什麼 | 對應步驟 |
| :--- | :--- | :--- |
| `train_prefix.py` | 新增 `--peft_method {lora, prefix}` | S2 |
| `train_prefix.py` | 新增 `--objective {dpo, simpo}`；SimPO 走 `CPOTrainer` + `CPOConfig(loss_type="simpo", simpo_gamma=…)` | S1 |
| `train_prefix.py` | 新增 `--gamma`（SimPO 的 target margin） | S1 |
| `train_prefix.py` | `--prefix_init_scale` 的手動 `p.mul_()` 改用 PEFT 原生 `PrefixTuningConfig(init_weights="zero")` | S1 |
| `train_prefix.py` | 新增 validation split 與 checkpoint selection | S3 |
| `train_prefix.py` | 新增資源指標記錄 callback（RQ2） | S2 |
| `train_prefix.py` | 新增 `--loss_scope {full, masked, adaptive}` | S5 |
| `eval/gen_for_icd.py` | **加 `--seed` 並實際呼叫 `torch.manual_seed`**（目前完全沒有 seed 控制） | S0.5 |
| 超參數 `lr=2e-5` | 改用 S0 的結果 | S0 |
| 超參數 `beta=0.1` | SimPO 改用 `β=1.5, γ=0.5`（論文設定），**不可沿用 DPO 的調參經驗** | S1 |
| 超參數 `epochs` | **固定為 1**（Exp2 已證明 2 會 over-train） | 全部 |

### 12.3 結果作廢 —— 但過程有價值

| 項目 | 為什麼作廢 | 殘餘價值 |
| :--- | :--- | :--- |
| Exp1 的 ON 數字（security 40.54 / HumanEval 61.59） | 目標函數是 DPO 非 SimPO；lr 疑似嚴重 underfit；資料未做 selection；未做退化檢查 | 作為 §9.5.4「DPO vs SimPO」ablation 的一格；系統整合的可行性證明 |
| Exp2 全組（β=0.05, ep2） | **變數混淆**（β 與 epochs 同時改），REPORT §12.3 已自承 | 「epochs=2 會 over-train」這個結論仍成立且有用，直接寫進計畫的硬規則 |
| Exp1/Exp2 的 OFF 數字 | 兩次不一致（45.21 vs 44.79），因為取樣無 seed 控制 | 修好 seed 之後重跑一次即可，成本低 |

> **重點：沒有任何東西需要「全部重來」。** 要重跑的只有訓練（本來就要重跑），評測管線一行都不用改（除了補 seed）。

---

## 13. 防坑清單（實作前必讀）

### 13.1 `prefix_projection=True` 的隱藏代價

已實測（Phi-3-mini：32 層、hidden 3072、nvt=20）：

| 設定 | 訓練期可訓練參數 |
| :--- | :--- |
| `prefix_projection=False` | **3.93 M** |
| `prefix_projection=True` | **613.48 M** |
| LoRA r=8 (q,k,v,o) | 6.29 M |

PEFT 的實作是 `Embedding(nvt, d) → Linear(d, d) → Tanh → Linear(d, L*2*d)`，最後一層的 `d × (32×2×3072)` 就是 613 M 的來源。**訓練期參數量是 LoRA 的近 100 倍**，直接摧毀 RQ2 的一半。另外 `load_prompt_embeddings()` 在此模式下會拋 `ValueError`，adapter 可攜性受損。**不要當預設值。**

### 13.2 zero-init 已是 PEFT 原生功能 —— 論文措辭要調整

PEFT 0.19.1 的 `PrefixTuningConfig` 已有 `init_weights` 欄位，設為 `"zero"` 時會直接把 `embedding.weight` 歸零（`prefix_projection=True` 時則歸零 transform 的最後一層）。

**這代表 `train_prefix.py` 裡手動 `p.mul_(scale)` 的 hack 可以直接換掉**，也代表**論文不能把 zero-init 寫成「我們提出的新機制」**——會被審稿人指出上游已有。

正確的框架是**實證發現**：
> 「我們發現 zero 初始化是 Prefix Tuning 結合 preference learning 時**穩定訓練的必要條件**：隨機初始化會使 policy 在 step 0 就大幅偏離 reference，導致 loss 達 53、grad_norm 達 1383、reward 量級異常（−700）；zero-init 後 loss 從 ln 2 ≈ 0.69 起步、grad_norm 穩定在 6–12。」

這個版本站得住，而且有完整的數據支撐（REPORT §6），是一個對複現者真正有用的資訊。

### 13.3 評測沒有 seed 控制

`eval/gen_for_icd.py` 用 `do_sample=True, temperature=0.8`，**但完全沒有設定隨機種子**。這就是 Exp1 與 Exp2 的 OFF 數字漂移的原因（45.21 vs 44.79；pooled 41.47 vs 41.66）。

在改善幅度只有 4~8 個百分點的情境下，這個漂移**不可忽略**，而且直接違反 §7.3 的「相同 random seeds」。

**修法**：加 `--seed` 參數並在生成前呼叫 `torch.manual_seed`。或更省事：**同一批實驗共用同一份 OFF 生成檔**，不要每次重生成。

（`run_humaneval.py` 用 `do_sample=False`（greedy），沒有這個問題。）

### 13.4 TRL 0.12.2 的 SimPO 在 `CPOTrainer`，不在 `DPOTrainer`

已驗證 `CPOConfig` 具備 `loss_type` 與 `simpo_gamma` 欄位。若在 `DPOConfig` 裡找 SimPO 會找不到。

**連帶影響**：CPO/SimPO 是 reference-free，不需要 `ref_model` 也不需要 `disable_adapter` 前向。目前 `train_prefix.py` 裡「`ref_model=None` → TRL 用 disable_adapter 當 reference」的那段邏輯在 SimPO 路徑下**不適用**，改寫時不要照抄。

### 13.5 β 在 DPO 與 SimPO 之間不可換算

* DPO 的 `β=0.1`：控制與 reference model 的 KL 約束強度
* SimPO 的 `β=1.5`：縮放 length-normalized reward 的溫度

兩者不在同一個尺度上。**把 DPO 的 0.1 套到 SimPO 上，訓練訊號會弱到幾乎不存在。** 第一輪嚴格用論文的 `β=1.5, γ=0.5`。

### 13.6 masking 只作用於 loss 聚合，不作用於前向

S5 實作時最容易犯的錯：把 mask 為 0 的 token 從序列裡移除。**不可以。** 前向必須跑完整序列（模型要看到完整 context 才能算出正確的 `log π(y_t | x, y_<t)`），mask 只在把 per-token log-prob 加總成 sequence reward 時套用。

### 13.7 CodeLlama 的 GQA

Phi-3-mini 是 MHA，CodeLlama-7B 是 GQA（`num_key_value_heads < num_attention_heads`）。Prefix 的 `past_key_values` 維度要依 KV head 數計算。PEFT 應該會處理，但 **S6 開始前務必先跑 `poc_prefix_smoke.py` 驗證前向與生成**——這正是當初寫這支腳本的目的（REPORT §8 測試 1、2）。

### 13.8 全 0 mask 的樣本

`convert_prosec_to_pref.py` 目前只做字串層級的 `chosen.strip() == rejected.strip()` 過濾。token 化之後仍可能出現 diff 為空的樣本（例如只差空白或註解）。S5 的 mask 建構階段必須再過濾一次，否則 `Σ m_t = 0` 會產生除以零。

---

## 14. 里程碑與檢查點

| 里程碑 | 內容 | 完成後可以說什麼 |
| :--- | :--- | :--- |
| **M0** | S0 + S0.5 完成 | 「我們知道 Prefix 的正確操作點，而且我們的安全性數字不是靠生成退化偽造的」 |
| **M1** | S1 + S2 完成 | 「我們有一個與 PROSEC 對齊的 LoRA baseline，跑在自己的管線上」 |
| **M2** | S3 完成 | **第一階段論文可以寫完**：RQ1、RQ2、RQ4 有答案，H1、H2 有驗證 |
| **M3** | S4 完成（若需要） | 「utility 的代價可以用 ProSec 既有機制回收，且我們知道它對 Prefix 與 LoRA 的效果是否對稱」 |
| **M4** | S5 完成 | **方法創新成立**：H3、H4 有答案，論文從「最低度增量」升級 |
| **M5** | S6 完成 | 「結論在第二個模型與 outcome-driven benchmark 上都成立」 |

### 14.1 最小可發表集合

若時間或資源受限，**M2 是最低門檻**（第一階段完整版：L0 + L1）。此時論文的定位是 §1.2 所說的「低難度增量式研究」，誠實地把貢獻寫成：

1. 把 proactive security alignment 系統化為 Prefix-only 的可插拔方法
2. 在完全受控的條件下公平比較 LoRA 與 Prefix 的安全／功能／資源 trade-off
3. 重建論文未釋出的評測子集，並修正 fence-less 抽取問題
4. Prefix + preference learning 的 zero-init 必要性與學習率區間（複現價值）

**若要達到「有方法創新」的標準，M4 是真正的目標。** §3.1 已經正確指出：提高創新性應該優先完成 adaptive localized 方法（S5），**而不是增加更多無關的 benchmark**。

### 14.2 每次實驗都要記錄的最小集合

為了滿足 §5.5 與 §7.3，每個 run 都要存一份 metadata：

```text
model revision / tokenizer / chat template hash
dataset revision / split 檔案 hash / preprocessing hash
peft_method + 完整 peft config
objective + beta + gamma + loss_scope
lr / scheduler / warmup / grad_clip / epochs / effective batch / seed
transformers / peft / trl / torch / CUDA 版本
評測：instruct.json 版本、semgrep/weggli 版本、生成參數 + 評測 seed
資源：trainable params / adapter 大小 / peak GPU mem / train_runtime
退化守門：ON/OFF 的生成長度、parse rate、空實作比例
```

建議直接在 `train_prefix.py` 結束時輸出成 `run_meta.json`，評測腳本再附加自己的欄位。事後補這些資訊的成本遠高於當下記錄。
