#!/usr/bin/env bash
# Prefix 學習率校準（詳見 prefix-prosec_research_plam.md §9.1）
#
# 目的不是挑出要用的 adapter，是分離兩個假設：
#   (A) lr 太低導致 underfit  vs  (B) 目標函數/Prefix 容量的限制
# 兩者的曲線很像，但處置完全相反。同一批子集，唯一變數是 lr。
#
# 用法：
#   bash run_lr_sweep.sh                      # 預設 SimPO（S1 用）
#   OBJECTIVE=dpo LRS="2e-5 1e-4 1e-3 5e-3" \
#     OUT_DIR=outputs/lr_sweep_dpo bash run_lr_sweep.sh    # S0 用（已完成）
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OBJECTIVE="${OBJECTIVE:-simpo}"
OUT_DIR="${OUT_DIR:-outputs/lr_sweep_${OBJECTIVE}}"
# ProSec 論文 Appendix C Table 6：SimPO lr=5e-6, beta=1.5, gamma=0.5,
# 總 batch 64, Phi3-mini 跑 1500 steps。論文的 5e-6 是 LoRA 的值；
# Prefix 通常需要略高，所以由 5e-6 往上包住一段。
LRS="${LRS:-5e-6 2e-5 5e-5}"
# 論文的 effective batch 是 64。SimPO 的 margin 訊號在高相似度 pair 上很弱，
# batch 太小會被梯度雜訊淹掉，所以這裡對齊論文而不是沿用 S0 的 16。
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
echo "objective：$OBJECTIVE   effective batch：$((BATCH * GRAD_ACCUM))"
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
    --objective "$OBJECTIVE" --epochs 1 --lr "$LR" \
    --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" --bf16 \
    2>&1 | tee "$LOG"
  echo
done

echo "===== 全部完成，產出對照表 ====="
python eval/compare_lr_sweep.py "$OUT_DIR"/*.log --csv_dir "$OUT_DIR/curves" \
  2>&1 | tee "$OUT_DIR/summary.txt"
