# 工作規則

## 分析紀律（最重要）

**只做我們讀過的論文會做的分析。** 不要自創指標、不要為了解釋一個現象而發明新的診斷。

| 場合 | 只看這些 | 不要做 |
|---|---|---|
| 訓練 | `loss`、`rewards/margins`、`rewards/accuracies`、`rewards/chosen`、`rewards/rejected`、`grad_norm` | 自創的保留率、比值、擾動量等衍生指標 |
| 安全性 | 漏洞率（5 語言平均，對齊 ProSec Table 1） | 退化守門、語法完整率、stub 率 —— 除非數字看起來有異常才回頭查 |
| 功能性 | HumanEval pass@1（python）+ MultiPL-E（cpp/js） | 額外的生成品質分析 |

**結果與論文主張不符時**：先查文獻找可能原因，再給建議。不要自己發明機制解釋然後為它寫一支腳本。

**一次只改一個變數。** 要加實驗臂前先問「這回答哪個具體問題」，答不出來就不要加。

## 溝通

- 精簡、講清楚。結論先講，理由跟在後面。
- 不確定就說不確定，不要用一堆分析包裝。
- 提出選項時給建議，不要列一堆讓使用者選。

## 記錄

**每次跑出有用的實驗數據，立刻寫進 `EXPERIMENTS.md`**（日期 + 設定 + 數字 + 一句判讀）。不要只留在對話裡。

## 環境

```
model        microsoft/Phi-3-mini-4k-instruct   （H=3072, L=32, 總參數 3.82B）
訓練資料      data/train_pref.jsonl             45,785 筆、12 個 CWE
adapter      outputs/dpo-arms/<arm>
評測          $ICD = ProSec/PurpleLlama 的絕對路徑；$REPO = 本 repo
GPU          單張 RTX PRO 6000（96GB），一次只跑一件事
push         我沒有 GitHub 憑證，commit 後要請使用者自己 push
```

## 固定的訓練設定（不要動，動了就不是同一組對照）

```
--objective dpo --beta 0.05 --max_steps 800 --batch_size 1 --grad_accum 64
--max_length 2048 --max_prompt_length 1024 --max_grad_norm 0.3 --seed 42 --bf16
lr：LoRA 5e-6、prefix 5e-5
```

## 固定的評測指令

```bash
# 安全性（全量，693 題 × 10 樣本 × 5 語言；OFF 共用，會自動跳過）
ARMS="<arm>" bash scripts/run_full_eval.sh

# 功能性
python eval/run_humaneval.py --adapter outputs/dpo-arms/<arm> --max_new_tokens 2048 \
    --out outputs/humaneval_<arm>.json
python eval/run_multipl_e.py --adapter outputs/dpo-arms/<arm> --langs js,cpp \
    --max_new_tokens 2048 --out outputs/multipl_e_<arm>.json
```

- **`--max_new_tokens` 一律 2048。** 512/1024 會截斷，光 base 的 HumanEval 就被低估 6 pt。
- MultiPL-E **沒有 python**，python 那欄來自 `run_humaneval.py`。

## 已知的坑

- **`torch.distributed.tensor`**：peft 0.19.1 讀 `DTensor` 但 torch 不自動載入該 submodule，任何帶 LoRA 的腳本都要先 `try: import torch.distributed.tensor`。
- **PrefixTuning + gradient checkpointing 不相容**，會噴 tensor size 不符。
- **零初始化 prefix 沒有 dropout 時，nvt 個位置永遠相同**（有效長度 = 1）。`train_prefix.py` 已加守門。
- **生成時 VRAM 會漲到 90GB+**：`num_return_sequences=10` × 2048 tokens，加上 DynamicCache 每步重新配置造成的快取碎片。要並行跑別的就加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **統計解析度**：安全性 SE ≈ 0.60 pt（n=6,930）；功能性 SE ≈ 2~3 pt（n≈486）。功能性差異小於 3 pt 不能宣稱有效果。
