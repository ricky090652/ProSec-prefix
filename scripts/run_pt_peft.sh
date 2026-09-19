#!/usr/bin/env bash
# PT-PEFT 式的 prefix + LoRA 組合（Kim et al. 2024, arXiv:2411.00029）。
#
# 兩階段，第一階段已完成（prefix_nvt8），這裡只跑第二階段：
#   第一階段  只訓 prefix                    → outputs/dpo-arms/prefix_nvt8
#   第二階段  掛上 LoRA，prefix + LoRA 一起訓（原文 §3「including prefixes」）
#
# ⚠️ 論文用 CE + SCST，沒有 DPO，所以「第二階段的 reference 接哪裡」是我們的決定：
#    reference = 第一階段的模型（base + prefix），因為 DPO 的 reference 在理論上
#    應等於策略的初始化。報告時要標明這不是論文的做法。
#
# lr 沿用 LoRA 已掃出的 5e-6，比第一階段的 5e-5 低——與 PT-PEFT Table 13 的方向
# 一致（他們第二階段也用比第一階段低的 lr）。
#
# LoRA 預設掛 qkv_proj（PT-PEFT 用 Q/K/V）。我們自己的結果顯示掛 MLP 的取捨較好，
# 要試就 LORA_TARGETS=gate_up_proj,down_proj。
#
# 用法：
#   bash scripts/run_pt_peft.sh
#   LORA_TARGETS=gate_up_proj,down_proj bash scripts/run_pt_peft.sh
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT="${OUT:-outputs/dpo-arms}"
STAGE1="${STAGE1:-$OUT/prefix_nvt8}"
ARM="${ARM:-prefix_then_lora}"
LORA_TARGETS="${LORA_TARGETS:-qkv_proj}"
LORA_LR="${LORA_LR:-5e-6}"

# --- 與其他臂逐項相同 ---
BETA="${BETA:-0.05}"
STEPS="${STEPS:-800}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-64}"

[ -d "$STAGE1" ] || { echo "找不到第一階段的 prefix：$STAGE1"; exit 1; }
mkdir -p "$OUT"

echo "=== PT-PEFT 第二階段：$STAGE1 + LoRA($LORA_TARGETS) ==="
python train_prefix.py \
  --model "$MODEL" --train_file "$TRAIN_FILE" \
  --peft_method prefix_then_lora --stage1_prefix "$STAGE1" \
  --lora_r 8 --lora_alpha 16 --lora_target_modules "$LORA_TARGETS" \
  --objective dpo --beta $BETA \
  --max_steps $STEPS --batch_size "$BATCH" --grad_accum "$ACCUM" \
  --max_length 2048 --max_prompt_length 1024 --max_grad_norm 0.3 \
  --seed 42 --bf16 --save_steps 100 --save_total_limit 10 \
  --lr "$LORA_LR" --output_dir "$OUT/$ARM" \
  2>&1 | tee "$OUT/$ARM.log"

cat <<EOF

完成。存檔結構：$OUT/$ARM（根 = prefix、lora/ = LoRA），checkpoint 同樣結構。

評測（step 800 若過度最佳化，改用 checkpoint-N，路徑直接指過去即可）：

  python eval/run_humaneval.py --adapter $OUT/$ARM --max_new_tokens 2048 \\
      --out outputs/humaneval_$ARM.json
  ARMS=$ARM bash scripts/run_full_eval.sh
  python eval/run_multipl_e.py --adapter $OUT/$ARM --langs js,cpp \\
      --max_new_tokens 2048 --out outputs/multipl_e_$ARM.json

對照：prefix_nvt8 安全性 −3.00 / 功能性 −7.22；lora_qkv −2.58 / +1.03。
EOF
