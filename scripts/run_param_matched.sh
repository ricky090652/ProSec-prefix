#!/usr/bin/env bash
# 參數量精確配對的 prefix vs LoRA 對照（P-match）。
#
# 為什麼需要：D-run 的兩臂在可訓練參數上差 4 倍，
#   prefix nvt=16        3,145,728
#   LoRA r=8 all-linear 12,582,912
# 所以「prefix 安全性只有三分之一、utility 卻掉兩倍多」無法區分是方法的問題
# 還是預算的問題。
#
# Phi-3-mini（hidden 3072、32 層）的數字剛好可以精確配對：
#   prefix 參數 = nvt × 層數 × 2(k,v) × hidden
#   nvt=64 → 64 × 32 × 2 × 3072 = 12,582,912  ==  LoRA r=8 all-linear
#   nvt=16 →                       3,145,728  ==  LoRA r=8 只掛 qkv_proj
#
# 跑完會有四個點，可以畫出「參數量 × 安全性 × utility」的完整圖：
#
#            3.1M 參數              12.6M 參數
#   prefix   nvt=16  ✅已有          nvt=64   ← 本腳本
#   LoRA     qkv_proj ← 本腳本        all-linear ✅已有
#
# 除了 --peft_method 與各自的 lr，其餘設定與 scripts/run_dpo_arms.sh 完全相同。
#
# 用法：
#   bash scripts/run_param_matched.sh          # 兩組都跑（約 7-8 小時）
#   ARMS=prefix64 bash scripts/run_param_matched.sh   # 只跑其中一組
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT="${OUT:-outputs/dpo-arms}"
ARMS="${ARMS:-prefix64 loraqkv}"

# --- 與 run_dpo_arms.sh 逐項相同，不可分開調 ---
BETA="${BETA:-0.05}"
STEPS="${STEPS:-800}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-64}"
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
MAX_GRAD_NORM=0.3
SEED=42

# lr 沿用各自那一臂 D-lr sweep 掃出來的值。嚴格說改變容量後最佳 lr 可能移動，
# 但先固定 lr 才能把「容量」單獨隔離出來——一次只改一個變數。
PREFIX_LR="${PREFIX_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-6}"

mkdir -p "$OUT"
common=(
  --model "$MODEL" --train_file "$TRAIN_FILE"
  --objective dpo --beta $BETA
  --max_steps $STEPS --batch_size "$BATCH" --grad_accum "$ACCUM"
  --max_length $MAX_LENGTH --max_prompt_length $MAX_PROMPT_LENGTH
  --max_grad_norm $MAX_GRAD_NORM
  --seed $SEED --bf16
  --save_steps 100 --save_total_limit 10
)

for arm in $ARMS; do
  case "$arm" in
    prefix64)
      echo "=== P-match-hi: prefix nvt=64（12,582,912 參數，lr=$PREFIX_LR）==="
      python train_prefix.py "${common[@]}" \
        --peft_method prefix --num_virtual_tokens 64 --prefix_init_scale 0 \
        --lr "$PREFIX_LR" --output_dir "$OUT/prefix_nvt64" \
        2>&1 | tee "$OUT/prefix_nvt64.log"
      ;;
    loraqkv)
      echo "=== P-match-lo: LoRA 只掛 qkv_proj（3,145,728 參數，lr=$LORA_LR）==="
      python train_prefix.py "${common[@]}" \
        --peft_method lora --lora_r 8 --lora_alpha 16 \
        --lora_target_modules qkv_proj \
        --lr "$LORA_LR" --output_dir "$OUT/lora_qkv" \
        2>&1 | tee "$OUT/lora_qkv.log"
      ;;
    *) echo "未知的 arm：$arm（可用：prefix64 loraqkv）"; exit 1 ;;
  esac
done

cat <<EOF

兩組訓練完成。先判讀，再評測：

  python eval/compare_lr_sweep.py $OUT/prefix_nvt64.log $OUT/lora_qkv.log

看 acc 有沒有比原本那一臂高、比值有沒有掉到 3 以下、grad 峰值有沒有爆。

接著各跑便宜篩選 + HumanEval（**兩者都要帶 --max_new_tokens 2048**，
512/1024 會截斷，光是 base 的 HumanEval 就會被低估 6 pt）：

  for ARM in prefix_nvt64 lora_qkv; do
    python eval/gen_for_icd.py --adapter $OUT/\$ARM \\
        --instruct_json \$ICD/CybersecurityBenchmarks/datasets/instruct/instruct.json \\
        --safecoder_only --langs c,cpp --limit 100 --num_gen 5 \\
        --max_new_tokens 2048 --out_prefix outputs/screen2_\$ARM
    python eval/run_humaneval.py --adapter $OUT/\$ARM \\
        --max_new_tokens 2048 --out outputs/humaneval_\$ARM.json
  done

然後 normalize → detect_all → score_detected → check_degeneration（見 RUNBOOK_zh.md §3）。
EOF
