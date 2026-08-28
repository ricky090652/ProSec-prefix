#!/usr/bin/env bash
# Prefix 學習率校準（詳見 prefix-prosec_research_plam.md §9.1）
#
# 目的不是挑出要用的 adapter，是分離兩個假設：
#   (A) lr 太低導致 underfit  vs  (B) 目標函數/Prefix 容量的限制
# 兩者的曲線很像，但處置完全相反。同一批子集，唯一變數是 lr。
#
# 兩個臂（prefix / lora）必須用「同等搜尋預算」——同樣的步數、同樣的子集、
# 同樣數量的 lr 候選。這是論文裡要寫清楚的方法細節，否則對照不成立。
#
# 用法：
#   PEFT_METHOD=prefix bash run_lr_sweep.sh
#   PEFT_METHOD=lora   bash run_lr_sweep.sh
# 兩臂用同一張預設網格（5 個點），不要各自指定不同的 LRS。
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OBJECTIVE="${OBJECTIVE:-dpo}"
PEFT_METHOD="${PEFT_METHOD:-prefix}"
OUT_DIR="${OUT_DIR:-outputs/lr_sweep_${OBJECTIVE}_${PEFT_METHOD}}"
# 兩臂掃**同一張網格**——不只個數相同，候選也相同。這比「各自在自己的
# 尺度範圍內掃」更好辯護：沒有人能說某一臂的範圍被挑過。
# 網格同時涵蓋論文的 5e-6（Table 6 所有目標函數都用這個值）與
# S0 在 prefix 上找到的 1e-4。
LRS="${LRS:-5e-6 2e-5 5e-5 1e-4 2e-4}"
# 論文的 effective batch 是 64（Appendix C）。S0 用的 16 梯度雜訊大一倍，
# 已證實會讓同一個 lr 的結論翻轉，所以兩臂都對齊 64。
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
# ⚠️ 以下四項必須與 scripts/run_dpo_arms.sh 一致，否則是在 A 條件下挑 lr、
# 拿去跑 B 條件的訓練。之前 max_length 用預設的 1024（截斷 12% 樣本）、
# 全長訓練卻用 2048，就是這個錯。
BETA="${BETA:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.3}"
# 16000 筆 / effective batch 64 = 250 個 optimizer step，夠看出趨勢。
MAX_SAMPLES="${MAX_SAMPLES:-16000}"
SEED="${SEED:-42}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "找不到 $TRAIN_FILE，請先執行："
  echo "  python data/convert_prosec_to_pref.py \\"
  echo "      --hf_dataset prosecalign/prosec-mixed-phi3mini-4k-inst \\"
  echo "      --out $TRAIN_FILE"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "objective：$OBJECTIVE   peft：$PEFT_METHOD   effective batch：$((BATCH * GRAD_ACCUM))"
echo "資料：$TRAIN_FILE（$(wc -l < "$TRAIN_FILE") 筆，取子集 $MAX_SAMPLES / seed $SEED）"
echo "beta：$BETA   max_length：$MAX_LENGTH   max_grad_norm：$MAX_GRAD_NORM"
echo "掃描 lr：$LRS"
echo

for LR in $LRS; do
  LOG="$OUT_DIR/lr_${LR}.log"
  if [[ -s "$LOG" ]] && grep -q "train_runtime" "$LOG"; then
    echo "===== lr=${LR} 已完成，略過（要重跑請刪 $LOG）====="
    continue
  fi
  echo "===== lr=${LR}  $(date +%H:%M:%S) ====="
  python train_prefix.py \
    --model "$MODEL" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUT_DIR/lr_${LR}" \
    --max_samples "$MAX_SAMPLES" --seed "$SEED" \
    --peft_method "$PEFT_METHOD" \
    --num_virtual_tokens 16 --prefix_init_scale 0 \
    --lora_r 8 --lora_alpha 16 \
    --objective "$OBJECTIVE" --beta "$BETA" --epochs 1 --lr "$LR" \
    --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
    --max_length "$MAX_LENGTH" --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --bf16 --gradient_checkpointing \
    2>&1 | tee "$LOG"
  echo
done

echo "===== 全部完成，產出對照表 ====="
python eval/compare_lr_sweep.py "$OUT_DIR"/*.log --csv_dir "$OUT_DIR/curves" \
  2>&1 | tee "$OUT_DIR/summary.txt"
