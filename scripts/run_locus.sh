#!/usr/bin/env bash
# 安全對齊住在哪裡？MLP vs 注意力的參數量配對對照（L-locus）。
#
# 為什麼需要：目前「prefix 輸給 LoRA」還缺一個機制解釋。
# 「prefix 容量比較小」講不通——prefix nvt=64 的參數量 12,582,912 與
# LoRA all-linear 完全相同，utility 卻是 −32.93 對 −4.12。
#
# 真正的差別是**作用位置**：prefix 只在每層的 key/value 前面接東西，
# 結構上碰不到 MLP；LoRA 的 gate_up_proj / down_proj 就在射程內。
#
# 已有的線索指向 MLP —— CWE-338 上 all-linear −22.13、qkv_proj 只有 −1.27，
# 兩者的差集正是 o_proj + MLP。但 all-linear(12.6M) 與 qkv(3.1M) 差 4 倍參數，
# 分不出是「位置」還是「預算」。這兩臂把參數量配平，只留下位置這一個變數：
#
#   MLP-only        gate_up_proj + down_proj   r=8    7,864,320
#   Attention-only  qkv_proj + o_proj          r=13   7,667,712   （差 2.5%）
#
# 判讀：
#   MLP-only 拿到大部分安全性、attention-only 拿不到
#     → 安全對齊集中在 MLP，prefix 是「結構上搆不到」而非「能力不足」
#   兩邊差不多
#     → 位置假說不成立，安全性跟著參數量走；prefix 的劣勢要改從資料局部性解釋
#
# 除了 target_modules 與 r，其餘設定與 run_dpo_arms.sh / run_param_matched.sh
# 逐項相同。lr 沿用 LoRA 已掃出的 5e-6——換 target 後最佳 lr 可能移動，
# 但先固定才能把位置單獨隔離出來。
#
# 用法：
#   bash scripts/run_locus.sh                 # 兩臂都跑（約 7 小時）
#   ARMS=mlp bash scripts/run_locus.sh        # 只跑其中一臂
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT="${OUT:-outputs/dpo-arms}"
ARMS="${ARMS:-mlp attn}"

# --- 與 run_dpo_arms.sh 逐項相同，不可分開調 ---
BETA="${BETA:-0.05}"
STEPS="${STEPS:-800}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-64}"
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
MAX_GRAD_NORM=0.3
SEED=42
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
    mlp)
      # r*(3072+16384) + r*(8192+3072) = r*30720，×32 層，r=8 → 7,864,320
      echo "=== L-locus-mlp: LoRA 只掛 MLP（7,864,320 參數）==="
      python train_prefix.py "${common[@]}" \
        --peft_method lora --lora_r 8 --lora_alpha 16 \
        --lora_target_modules gate_up_proj,down_proj \
        --lr "$LORA_LR" --output_dir "$OUT/lora_mlp" \
        2>&1 | tee "$OUT/lora_mlp.log"
      ;;
    attn)
      # r*(3072+9216) + r*(3072+3072) = r*18432，×32 層，r=13 → 7,667,712
      # alpha 照 2r 的慣例給 26，維持與其他臂相同的 alpha/r 比值
      echo "=== L-locus-attn: LoRA 只掛注意力（7,667,712 參數）==="
      python train_prefix.py "${common[@]}" \
        --peft_method lora --lora_r 13 --lora_alpha 26 \
        --lora_target_modules qkv_proj,o_proj \
        --lr "$LORA_LR" --output_dir "$OUT/lora_attn" \
        2>&1 | tee "$OUT/lora_attn.log"
      ;;
    *) echo "未知的 arm：$arm（可用：mlp attn）"; exit 1 ;;
  esac
done

cat <<EOF

兩臂完成。先確認參數量真的配平了，再跑評測：

  python scripts/verify_arms.py $OUT/lora_mlp $OUT/lora_attn \\
      $OUT/lora $OUT/lora_qkv

  python eval/compare_lr_sweep.py $OUT/lora_mlp.log $OUT/lora_attn.log

接著併進全量評測（OFF 已經生成過，會自動跳過）：

  ARMS="lora_mlp lora_attn" bash scripts/run_full_eval.sh

  for ARM in lora_mlp lora_attn; do
    python eval/run_multipl_e.py --adapter $OUT/\$ARM --max_new_tokens 2048 \\
        --out outputs/multipl_e_\$ARM.json
  done

最想看的是 CWE-338：all-linear −22.13、qkv_proj −1.27。
若 lora_mlp 靠近前者、lora_attn 靠近後者，位置假說就成立，
prefix 的劣勢可以寫成「結構上搆不到 MLP」而不只是「表現比較差」。
EOF
