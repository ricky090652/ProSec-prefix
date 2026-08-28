#!/usr/bin/env bash
# 主線實驗：DPO 之下比較 Prefix (E2) 與 LoRA (E1)。
#
# 為什麼是 DPO 而不是論文預設的 SimPO：
#   SimPO 在這份資料上會失控——prefix 與 LoRA 兩次都把 chosen/rejected 一起壓垮
#   （EXPERIMENTS.md S2-repro）。論文 Appendix D.3 Table 8 本身就驗證過 DPO 在
#   同一份資料上有效（Vul 40.76 → 34.65、utility 42.78 → 44.20），所以改用 DPO
#   是論文認可的選項，不是退讓。
#
# 對照的成立條件（兩臂唯一的差別只有 --peft_method）：
#   同資料、同 beta、同 steps、同 effective batch、同 seed、同評測設定，
#   lr 各自用「同等搜尋預算」調（見 run_lr_sweep.sh）。
#
# 用法：
#   # 1) 先各掃 3 個 lr（同等預算），看 eval/compare_lr_sweep.py 的判讀挑贏家
#   PEFT_METHOD=prefix LRS="5e-5 1e-4 2e-4" bash run_lr_sweep.sh
#   PEFT_METHOD=lora   LRS="5e-6 2e-5 5e-5" bash run_lr_sweep.sh
#   # 2) 用挑出來的 lr 跑全長
#   PREFIX_LR=1e-4 LORA_LR=2e-5 bash scripts/run_dpo_arms.sh
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT="${OUT:-outputs/dpo-arms}"

# --- 兩臂共用（不可分開調，否則對照不成立）---
BETA=0.1              # S0 / Exp1 的診斷都在這個值上做的。論文 Table 6 的 DPO 用 0.05，
                      # 但 Exp2 在 0.05 下 utility 崩掉，取較保守（錨較強）的 0.1
STEPS=800             # @ batch 64 ≈ 1.1 epoch over 45,785；同時也是論文 Table 6 的 DPO 步數
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-64}"
MAX_LENGTH=2048       # S1-loc：1024 會截斷 12% 的樣本
MAX_PROMPT_LENGTH=1024
SEED=42

# --- 各臂的 lr（唯一允許不同的東西）---
PREFIX_LR="${PREFIX_LR:-1e-4}"
LORA_LR="${LORA_LR:-2e-5}"

mkdir -p "$OUT"
common=(
  --model "$MODEL" --train_file "$TRAIN_FILE"
  --objective dpo --beta $BETA
  --max_steps $STEPS --batch_size "$BATCH" --grad_accum "$ACCUM"
  --max_length $MAX_LENGTH --max_prompt_length $MAX_PROMPT_LENGTH
  --seed $SEED --bf16 --gradient_checkpointing
  --save_steps 100 --save_total_limit 10
)
# 注意：**不帶 --system_prompt**。ProSec 程式庫裡有三個互不相同的 system prompt，
# 訓練/評測用哪一個無從查證；不帶才能與既有的 prefix 實驗與 base OFF 基線相容。

echo "=== E1: LoRA + DPO (lr=$LORA_LR) ==="
python train_prefix.py "${common[@]}" \
  --peft_method lora --lora_r 8 --lora_alpha 16 \
  --lr "$LORA_LR" --output_dir "$OUT/lora" \
  2>&1 | tee "$OUT/lora.log"

echo "=== E2: Prefix + DPO (lr=$PREFIX_LR) ==="
python train_prefix.py "${common[@]}" \
  --peft_method prefix --num_virtual_tokens 16 --prefix_init_scale 0 \
  --lr "$PREFIX_LR" --output_dir "$OUT/prefix" \
  2>&1 | tee "$OUT/prefix.log"

cat <<EOF

兩臂訓練完成。先判讀，再評測：

  python eval/compare_lr_sweep.py $OUT/lora.log $OUT/prefix.log

看 chosen 有沒有守住、margin 是不是雙邊貢獻、grad_norm 全程 < 50。
過關後照 RUNBOOK_zh.md §3-4 跑評測，兩臂用**同一組評測參數與 seed**。
最終的 prefix vs LoRA 對照一定要用全套 693 題 x 10 samples——
100 題 x 5 的標準誤約 2.2pt，分不出兩臂的差距。
EOF
