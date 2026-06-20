# 操作手冊（端到端：資料 → 訓練 → 安全評測 → 功能評測）

> 從零把整套跑完的逐步指令。每步都說明「用哪支程式、它做什麼、輸入→輸出」。
> Server 路徑以 `~/314581038/` 為例（自行替換）。

## 全流程總覽

```
[資料] convert_prosec_to_pref.py      ProSec HF 資料 → (prompt,chosen,rejected) jsonl
   │
[訓練] train_prefix.py                Phi-3 凍結 + PEFT prefix + TRL DPO → prefix adapter
   │
   ├─[安全] gen_for_icd.py → normalize_responses.py → (ProSec)detect_all.py → score_detected.py
   │         生成程式碼           補 code fence            semgrep/weggli 偵測   算 vulnerable ratio
   │
   └─[功能] run_humaneval.py（Python）   run_multipl_e.py（js/cpp/java）
             pass@1                        pass@1
```

---

## 0. 環境準備（每台機器一次）

```bash
cd ~/314581038/ProSec-prefix
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch                                   # GPU 機器：確認 torch.cuda.is_available()=True
pip install "transformers>=4.43,<4.47" "peft>=0.12" "trl>=0.11,<0.13" \
            "accelerate>=0.33" "datasets>=2.20"
# 安全評測偵測器：
pip install semgrep==1.51.0 pyyaml
# （之後做 C/C++ 偵測才需要）weggli：cargo install weggli（見 §3 註）

export HF_HOME=~/314581038/ProSec-prefix/.hf_cache   # 模型/資料下載快取放這
```

**作用**：建立隔離的 Python 環境並裝齊套件。`torch` 給模型、`transformers/peft/trl` 給訓練、
`semgrep` 給安全偵測、`datasets` 給下載資料集。

---

## 1. 資料準備

```bash
python data/convert_prosec_to_pref.py \
    --hf_dataset prosecalign/prosec-mixed-phi3mini-4k-inst \
    --out data/train_pref.jsonl
wc -l data/train_pref.jsonl     # 應約 45,785
```

| 項目 | 說明 |
|---|---|
| 程式 | `data/convert_prosec_to_pref.py` |
| 作用 | 從 HuggingFace 下載 ProSec 的混合偏好資料，轉成 DPO 要的格式 |
| 對應 | `original_instruction`→prompt、`fixed_code`→chosen(安全)、`original_code`→rejected(脆弱) |
| 輸入 | HF 資料集 `prosecalign/prosec-mixed-phi3mini-4k-inst`（45,785 筆，Dsec+Dnorm） |
| 輸出 | `data/train_pref.jsonl`（每行 `{prompt, chosen, rejected, lang, cwe}`） |

---

## 2. 訓練 prefix

```bash
tmux new -s train                       # 防斷線
python train_prefix.py --model microsoft/Phi-3-mini-4k-instruct \
    --train_file data/train_pref.jsonl \
    --output_dir ./outputs/phi3-prefix-dpo-full \
    --bf16 --num_virtual_tokens 16 \
    --lr 2e-5 --beta 0.1 --epochs 1 \
    --batch_size 1 --grad_accum 16 --max_length 1024 \
    2>&1 | tee train.log
```

| 項目 | 說明 |
|---|---|
| 程式 | `train_prefix.py` |
| 作用 | 載入 Phi-3（**凍結**）→ 套 PEFT PrefixTuning（只訓練 prefix）→ TRL DPOTrainer 用偏好對訓練 |
| 關鍵 | `--prefix_init_scale 0`（預設）零初始化讓訓練穩定；reference 用「停用 adapter」自動取得 |
| 輸入 | `data/train_pref.jsonl` |
| 輸出 | `./outputs/phi3-prefix-dpo-full/`（prefix adapter：`adapter_model.safetensors` 等） |
| 時間 | 1 epoch 約 2.5h（RTX Pro 6000）；epochs 2 約 5h |

**重要參數**（調這些做實驗）：
- `--beta`：偏離 base 的力道。**小(0.05)→保 utility**；大→security 強但傷 utility。
- `--epochs`：1~2（過多會 overfit）。
- `--num_virtual_tokens`：prefix 長度（16/32）。
- **每次換參數都改 `--output_dir`**，不要覆蓋舊 adapter（要留著對照）。

（可選）訓練前先驗 PEFT prefix 在模型上正常：
```bash
python poc_prefix_smoke.py --model microsoft/Phi-3-mini-4k-instruct
```
`poc_prefix_smoke.py` 作用：驗證 prefix 的 forward/generate/reference 行為正常（看到「✅ prefix 生效」即可）。

---

## 3. 安全性評測（PurpleLlama，對齊論文子集）

前置：需要 ProSec 的 PurpleLlama submodule（含偵測器 ICD + 題庫）。
```bash
cd ~/314581038
git clone --recurse-submodules https://github.com/PurCL/ProSec.git   # 第一次才要
# C/C++ 偵測需 weggli（無 sudo 用 cargo 裝到家目錄）：
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env && cargo install weggli
```

### 3a. 生成程式碼（prefix ON/OFF）
```bash
cd ~/314581038/ProSec-prefix
python eval/gen_for_icd.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --adapter ./outputs/phi3-prefix-dpo-full \
    --instruct_json ~/314581038/ProSec/PurpleLlama/CybersecurityBenchmarks/datasets/instruct/instruct.json \
    --langs c,cpp,java,javascript,python --safecoder_only --num_gen 10 \
    --out_prefix ./outputs/icd_paper
```
| 程式 | `eval/gen_for_icd.py` |
|---|---|
| 作用 | 對 PurpleLlama 題目，用模型生程式碼，**prefix ON 與 OFF 各一份** |
| `--safecoder_only` | 只留 SafeCoder 涵蓋的 CWE → 重建論文的 36 對/693 題子集 |
| `--langs` | 對齊論文 5 語言 |
| 輸出 | `outputs/icd_paper.on.jsonl`、`.off.jsonl`（每行含 `lang,cwe,prompt,responses`） |

### 3b. 補 code fence（必要！）
```bash
python eval/normalize_responses.py --in_file outputs/icd_paper.off.jsonl --out outputs/icd_paper.off.norm.jsonl
python eval/normalize_responses.py --in_file outputs/icd_paper.on.jsonl  --out outputs/icd_paper.on.norm.jsonl
```
| 程式 | `eval/normalize_responses.py` |
|---|---|
| 作用 | Phi-3 常輸出**裸 code 無 ```fence```**，會被偵測器漏抓（當成安全）。此步把無 fence 回應補上 fence |
| 為何必要 | 不做的話 ~84% 程式碼被丟棄，vulnerable ratio 被嚴重低估 |
| 輸出 | `*.norm.jsonl` |

### 3c. 靜態分析偵測（用 ProSec 的程式）
```bash
cd ~/314581038/ProSec/PurpleLlama
source ~/314581038/ProSec-prefix/.venv/bin/activate    # 要有 semgrep
source ~/.cargo/env                                    # 要有 weggli
python prosec_scripts/detect_all.py --fin ~/314581038/ProSec-prefix/outputs/icd_paper.off.norm.jsonl
python prosec_scripts/detect_all.py --fin ~/314581038/ProSec-prefix/outputs/icd_paper.on.norm.jsonl
```
| 程式 | `prosec_scripts/detect_all.py`（ProSec/PurpleLlama 內，**非本 repo**） |
|---|---|
| 作用 | 用 Insecure Code Detector（semgrep+weggli+regex）掃每段程式碼，標記觸發了哪些 CWE |
| 注意 | **必須在 `PurpleLlama/` 目錄下執行**（它用相對路徑 import） |
| 輸出 | `*.norm.jsonl.detected.jsonl`（每段碼附 `detection_results`，空=安全） |

### 3d. 計分（ON vs OFF）
```bash
cd ~/314581038/ProSec-prefix
python eval/score_detected.py \
    --on  ./outputs/icd_paper.on.norm.jsonl.detected.jsonl \
    --off ./outputs/icd_paper.off.norm.jsonl.detected.jsonl \
    | tee ./outputs/security_paper.txt
```
| 程式 | `eval/score_detected.py` |
|---|---|
| 作用 | 統計 vulnerable ratio（非空程式碼中被標脆弱的比例），印「依語言」「依 CWE」「OVERALL」三表，ON vs OFF 對照 |
| 看什麼 | Δ 為負 = prefix 變更安全；對齊論文看「依語言」表，對 5 語言取平均 |
| 輸出 | 終端機表格 + `security_paper.txt` |

---

## 4. 功能性評測

### 4a. Python（原始 HumanEval）
```bash
python eval/run_humaneval.py --adapter ./outputs/phi3-prefix-dpo-full \
    --out ./outputs/humaneval_result.json
```
| 程式 | `eval/run_humaneval.py` |
|---|---|
| 作用 | 對 HumanEval 164 題，instruct 模式生解答→子行程執行測試→算 pass@1，prefix ON vs OFF |
| 輸出 | 終端機摘要 + `humaneval_result.json`（`off_pass@1 / on_pass@1 / delta`） |

### 4b. 多語言（MultiPL-E：js / cpp，java 可選）
前置工具鏈（無 sudo，下載預編譯到家目錄）：node、g++（多半已有）。
```bash
export PATH=$HOME/node-v20.18.0-linux-x64/bin:$PATH    # node 路徑換成你的
python eval/run_multipl_e.py --adapter ./outputs/phi3-prefix-dpo-full \
    --langs js,cpp --out ./outputs/multipl_e_result.json
```
| 程式 | `eval/run_multipl_e.py` |
|---|---|
| 作用 | 對 MultiPL-E（多語言 HumanEval）算 pass@1，prefix ON vs OFF |
| 範式 | **instruct 模式**（chat template 請模型實作→抽 code→組裝測試編譯執行）。⚠️ 不能用裸補全，否則 prefix OOD → ON 變 0% |
| 工具 | js 要 `node`、cpp 要 `g++`；java 要 JDK + javatuples jar（`--javatuples_jar`，較麻煩，可略） |
| 注意 | 絕對值與 Python 版不可比，只看 ON/OFF 的 Δ |
| 輸出 | 終端機表 + `multipl_e_result.json` |

（可選）定性檢查：
```bash
python eval/gen_compare.py --adapter ./outputs/phi3-prefix-dpo-full --prompts eval/prompts_security.jsonl
```
`gen_compare.py` 作用：對幾個 CWE-prone 提示，並排印 prefix ON/OFF 的生成，肉眼看差異（非數字評測）。

---

## 5. 結果彙整

把上面數字填進主結果表（REPORT.md §10/§11）：

| 指標 | base(OFF) | +prefix(ON) | Δ |
|---|---|---|---|
| 安全 Vulnerable Ratio ↓（5 語言平均，看 score_detected「依語言」表） | … | … | … |
| 功能 HumanEval pass@1 ↑（humaneval_result.json） | … | … | … |
| 功能 MultiPL-E pass@1（multipl_e_result.json，js/cpp） | … | … | … |

---

## 附錄：程式一覽

| 程式 | 階段 | 作用 |
|---|---|---|
| `data/convert_prosec_to_pref.py` | 資料 | ProSec HF 資料 → (prompt,chosen,rejected) |
| `train_prefix.py` | 訓練 | PEFT PrefixTuning + TRL DPO 訓練 prefix |
| `poc_prefix_smoke.py` | 驗證 | 確認 PEFT prefix forward/generate/reference 正常 |
| `eval/gen_for_icd.py` | 安全 | 對 PurpleLlama 生成程式碼（ON/OFF；`--safecoder_only`/`--langs`） |
| `eval/normalize_responses.py` | 安全 | 給無 fence 回應補 fence（detect 前必跑） |
| `prosec_scripts/detect_all.py` | 安全 | （ProSec）ICD 靜態分析偵測 CWE |
| `eval/score_detected.py` | 安全 | 算 vulnerable ratio（依語言/CWE/總體，ON vs OFF） |
| `eval/run_humaneval.py` | 功能 | Python HumanEval pass@1（ON/OFF） |
| `eval/run_multipl_e.py` | 功能 | 多語言 MultiPL-E pass@1（ON/OFF） |
| `eval/gen_compare.py` | 定性 | prefix ON/OFF 並排生成比較 |

## 附錄：常見卡點

| 症狀 | 處理 |
|---|---|
| 安全 ratio 異常低 | 忘了跑 `normalize_responses.py`（裸 code 被漏抓） |
| `detect_all.py` import error | 沒在 `PurpleLlama/` 目錄下跑；或缺 `pip install pyyaml` |
| `semgrep/weggli: command not found` | 沒 `source .venv`（semgrep）或沒 `source ~/.cargo/env`（weggli） |
| MultiPL-E ON 全 0% | 用了裸補全模式（prefix OOD）→ 確認用的是 instruct 版 run_multipl_e.py |
| MultiPL-E 全 0% | node/g++ 不在 PATH，或 java 缺 javatuples jar |
| 訓練 grad_norm 爆炸 | 確認 `--prefix_init_scale 0`（預設）；必要時降 `--lr` |
| OOM | 降 `--max_length`、開 `--gradient_checkpointing`、或 `--load_4bit` |
