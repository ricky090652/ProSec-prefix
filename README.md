# prosec-prefix

用 **ProSec 的主動式安全對齊資料**，訓練一組 **prefix（PEFT PrefixTuning）** 插入
現代 code LLM（CodeLlama-7B），透過 **DPO/SimPO** 做安全對齊。

概念上承接 SVEN 的「sec control-prefix」，但實作完全移到現代棧：
prefix 注入的水管（Cache / generate / RoPE / GQA）交給 PEFT 處理，不再手刻 modeling。

## 三方角色分工

| Repo | 角色 | 是否修改 |
|---|---|---|
| `../sven` | 舊 baseline（CodeGen + 真實世界資料 + 手刻 prefix） | 凍結，不動 |
| `../ProSec` | 資料工廠（合成 CWE 誘發指令 → 生脆弱碼 → 修復 → 掃描 → 混合） | 就地使用，不 fork |
| `prosec-prefix`（本 repo） | 新方法：ProSec 資料 + prefix + DPO | 主要開發處 |

## 為什麼是 prefix 而非 LoRA

ProSec 原版用 LoRA + SimPO。本專案的差異化貢獻：用 **prefix-tuning** 取代 LoRA —
prefix 可即插即拔、可同時保留多組控制（sec / vul），且只訓練極少參數。
在 PEFT 裡這只是 `LoraConfig` → `PrefixTuningConfig` 的替換。

## 快速開始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 步驟 0：先驗唯一的風險（建議第一步）

確認 PEFT PrefixTuning 在 CodeLlama 上的 forward / generate / reference 行為正確：

```bash
# 想先用小模型快速驗程式碼路徑（不下載 7B）：
python poc_prefix_smoke.py --model sshleifer/tiny-gpt2 --no_chat_template
# 正式驗 CodeLlama：
python poc_prefix_smoke.py --model codellama/CodeLlama-7b-Instruct-hf
```
看到「✅ prefix 生效、policy≠reference」才繼續往下。

### 步驟 1：取得 ProSec 偏好資料

ProSec 已把最終混合(Dsec+Dnorm)偏好資料**發佈在 HuggingFace**，第一階段可
**直接下載，免自己跑 vLLM+Claude 生成**：

| HF 資料集 | target 分佈 | 筆數 | 欄位 |
|---|---|---|---|
| `prosecalign/prosec-mixed-phi3mini-4k-inst` | Phi-3-mini（第一階段用） | 45.8k | `original_instruction` / `fixed_code` / `original_code` / `lang` / `cwe` / `benign` |
| `prosecalign/prosec-mixed-clm7b-inst` | CodeLlama-7B（第二階段） | 61.2k | 同上 |

欄位名與本轉換腳本的預設相符，**不需覆寫**：

```bash
# 第一階段（Phi-3-mini）
python data/convert_prosec_to_pref.py \
    --hf_dataset prosecalign/prosec-mixed-phi3mini-4k-inst \
    --out data/train_pref.jsonl
```

> 若日後要自己用 CodeLlama 生 in-distribution 資料，再到 `../ProSec` 跑它的
> pipeline（target 用 CodeLlama-7B-Instruct），產出後一樣用本腳本 `--in_jsonl` 轉換。

### 步驟 2：訓練 prefix

```bash
python train_prefix.py \
    --model codellama/CodeLlama-7b-Instruct-hf \
    --train_file data/train_pref.jsonl \
    --output_dir outputs/codellama7b-prefix-dpo \
    --num_virtual_tokens 16 --beta 0.1 --lr 5e-5 \
    --epochs 1 --batch_size 1 --grad_accum 16 --bf16
# 顯存吃緊可加 --load_4bit
```

### 步驟 3：評測

見 `eval/README.md`（採 ProSec 的 PurpleLlama 評測，非舊 SVEN 的 CodeQL）。

## SimPO（reference-free）

若要用 SimPO 而非標準 DPO：SimPO 在 TRL 對應 `CPOTrainer` + `CPOConfig(loss_type="simpo", cpo_alpha=...)`。
可在 `train_prefix.py` 基礎上替換 trainer（介面幾乎相同）。

## 與舊 SVEN 的關係

- **保留**：sec prefix 的概念、DPO 偏好目標的概念。
- **丟棄**：SVEN 的 dataset/diff-mask、`hf/` 手刻 modeling、舊環境、CodeGen、completion 評測。
- 舊 `dpo-hybrid` 分支的 DPO loss 數學可當參考，但本 repo 用 TRL 直接提供。
