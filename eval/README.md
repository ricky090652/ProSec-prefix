# 安全評測（PurpleLlama / CyberSecEval）

本專案採 ProSec 的評測範式（instruction 式），**不**沿用舊 SVEN 的 CodeQL completion 評測。

## 為什麼換評測

- ProSec 資料是「指令 → 完整回應（chosen/rejected）」格式，訓練出來的 prefix 作用在
  instruction-following 情境下。
- 對應的安全評測應該也是 instruction 式 → PurpleLlama CyberSecEval。
- ProSec repo 已內含 PurpleLlama 作為 git submodule（`ProSec/PurpleLlama/`）。

## 評測流程（高層）

1. 載入 base CodeLlama-7B-Instruct + 訓練好的 prefix adapter：
   ```python
   from peft import PeftModel
   from transformers import AutoModelForCausalLM
   base = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
   model = PeftModel.from_pretrained(base, "../outputs/codellama7b-prefix-dpo")
   ```
2. 用 PurpleLlama 的 instruct/autocomplete benchmark 對該模型取樣。
3. 用其靜態分析器（insecure code detector）算 vulnerable code ratio。
4. 對照組：
   - base CodeLlama-7B-Instruct（無 prefix；`model.disable_adapter()`）
   - ProSec 原版（LoRA / SimPO）—— 驗證「prefix 取代 LoRA」是否仍有效
   - 舊 SVEN（CodeGen + 真實世界資料）—— 跨範式參考

## 待辦

- [ ] 跑通 ProSec/PurpleLlama 對單一 checkpoint 的評測
- [ ] 包一支 `run_cyberseceval.py` 串接 prefix adapter
- [ ] 加 utility 評測（HumanEval / MBPP）確認 prefix 不傷功能性
