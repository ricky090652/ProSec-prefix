#!/usr/bin/env bash
# S0：Prefix 學習率校準（詳見 prefix-prosec_research_plam.md §9.1）
#
# 目的不是挑出要用的 adapter，是要分離兩個假設：
#   (A) lr 太低導致 underfit  vs  (B) 目標函數/Prefix 容量的限制
# 兩者的曲線很像，但處置完全相反。四組共用同一批 4,000 筆子集，唯一變數是 lr。
#
# 用法：  bash run_lr_sweep.sh
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/lr_sweep}"
LRS="${LRS:-2e-5 1e-4 1e-3 5e-3}"
MAX_SAMPLES="${MAX_SAMPLES:-4000}"
SEED="${SEED:-42}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "找不到 $TRAIN_FILE，請先執行："
  echo "  python data/convert_prosec_to_pref.py \\"
  echo "      --hf_dataset prosecalign/prosec-mixed-phi3mini-4k-inst \\"
  echo "      --out $TRAIN_FILE"
  exit 1
fi

mkdir -p "$OUT_DIR"
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
    --num_virtual_tokens 16 \
    --prefix_init_scale 0 \
    --beta 0.1 --epochs 1 --lr "$LR" \
    --batch_size 1 --grad_accum 16 --bf16 \
    2>&1 | tee "$LOG"
  echo
done

echo "===== 全部完成，產出對照表 ====="
python eval/compare_lr_sweep.py "$OUT_DIR"/*.log --csv_dir "$OUT_DIR/curves" \
  2>&1 | tee "$OUT_DIR/summary.txt"
