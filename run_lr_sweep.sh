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
#   PEFT_METHOD=prefix LRS="5e-5 1e-4 2e-4" bash run_lr_sweep.sh
#   PEFT_METHOD=lora   LRS="5e-6 2e-5 5e-5" bash run_lr_sweep.sh
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OBJECTIVE="${OBJECTIVE:-dpo}"
PEFT_METHOD="${PEFT_METHOD:-prefix}"
OUT_DIR="${OUT_DIR:-outputs/lr_sweep_${OBJECTIVE}_${PEFT_METHOD}}"
# 不同 PEFT 的自然 lr 尺度不同，所以候選範圍不同，但**個數要一樣**。
#   prefix：S0 在 batch 16 下找到 1e-4 最健康
#   lora  ：論文 Table 6 的 DPO 用 5e-6
LRS="${LRS:-5e-5 1e-4 2e-4}"
# 論文的 effective batch 是 64（Appendix C）。S0 用的 16 梯度雜訊大一倍，
# 已證實會讓同一個 lr 的結論翻轉，所以兩臂都對齊 64。
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
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
    --objective "$OBJECTIVE" --epochs 1 --lr "$LR" \
    --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" --bf16 \
    2>&1 | tee "$LOG"
  echo
done

echo "===== 全部完成，產出對照表 ====="
python eval/compare_lr_sweep.py "$OUT_DIR"/*.log --csv_dir "$OUT_DIR/curves" \
  2>&1 | tee "$OUT_DIR/summary.txt"
