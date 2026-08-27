#!/usr/bin/env bash
# ProSec 論文原始設定的復現：Phi-3-mini + LoRA + SimPO + §3.3 influence selection。
#
# 超參數全部來自論文，不是我們調出來的：
#   Appendix C Table 6  SimPO: lr=5e-6, beta=1.5, gamma=0.5, steps=1500 (Phi-3-mini)
#   Appendix C          LoRA r=8, alpha=16；total batch size 64
#   §5 / §6             warm-up: 在 D_sec 上跑 1000 steps，每 100 步存 checkpoint
#   §5                  D_norm 選擇：每個 D_sec 取 top-2，再丟掉全域最低 20%
#
# 論文沒寫、我們必須自己決定的部分列在 docs/REPRO_PAPER.md，改動時請同步更新。
#
# 用法：
#   bash scripts/repro_paper_lora_simpo.sh            # 跑完整四階段
#   STAGE=3 bash scripts/repro_paper_lora_simpo.sh    # 只從第 3 階段開始
#   STAGE=4 FINAL_TRAIN_FILE=data/train_pref.jsonl bash scripts/repro_paper_lora_simpo.sh
#                                                     # 跑法 A：跳過 selection，直接訓練
set -euo pipefail

MODEL=${MODEL:-microsoft/Phi-3-mini-4k-instruct}
TRAIN_FILE=${TRAIN_FILE:-data/train_pref.jsonl}
OUT=${OUT:-outputs/repro-paper}
STAGE=${STAGE:-1}
# 階段 4 吃的資料。預設是階段 3 篩出來的；跑法 A（見 docs/REPRO_PAPER.md §4）
# 直接指到原始的 train_pref.jsonl，跳過階段 1–3。
FINAL_TRAIN_FILE=${FINAL_TRAIN_FILE:-$OUT/train_pref_selected.jsonl}

# --- 論文超參數（不要在這裡調；要調就是另一個實驗，不叫復現） ---
LR=5e-6
BETA=1.5
GAMMA=0.5
LORA_R=8
LORA_ALPHA=16
MAIN_STEPS=1500
WARMUP_STEPS=1000
SAVE_STEPS=100
SYSTEM_PROMPT="You are helpful coding assistant."

# --- 硬體相關：只要乘積 = 64（論文的 total batch size）就等價 ---
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}

echo "effective batch = $((BATCH * ACCUM))（論文 64）"
[ $((BATCH * ACCUM)) -eq 64 ] || echo "⚠️  不等於 64，這不是論文設定"

common_args=(
  --model "$MODEL"
  --peft_method lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA
  --objective simpo --beta $BETA --gamma $GAMMA --cpo_alpha 0
  --lr $LR
  --batch_size "$BATCH" --grad_accum "$ACCUM"
  --max_length "$MAX_LENGTH" --max_prompt_length "$MAX_PROMPT_LENGTH"
  --system_prompt "$SYSTEM_PROMPT"
  --max_grad_norm 1.0            # LLaMA-Factory 預設；不是我們主線的 0.3
  --lr_scheduler_type cosine --warmup_ratio 0.1   # SimPO 官方 recipe
  --bf16 --gradient_checkpointing
)

# ── 階段 1：暖身訓練（只用 D_sec，1000 步，每 100 步存檔）────────────────
if [ "$STAGE" -le 1 ]; then
  echo "=== [1/4] warm-up on D_sec：$WARMUP_STEPS steps ==="
  python train_prefix.py "${common_args[@]}" \
    --train_file "$TRAIN_FILE" --subset dsec \
    --output_dir "$OUT/warmup-dsec" \
    --max_steps $WARMUP_STEPS --save_steps $SAVE_STEPS --save_total_limit 20
fi

# ── 階段 2：在 10 個 checkpoint 上量 training dynamics ────────────────────
if [ "$STAGE" -le 2 ]; then
  echo "=== [2/4] collect training dynamics ==="
  python data/collect_training_dynamics.py \
    --train_file "$TRAIN_FILE" --base_model "$MODEL" \
    --ckpt_root "$OUT/warmup-dsec" --out_dir "$OUT/dynamics" \
    --ckpt_every $SAVE_STEPS --sides both \
    --max_length "$MAX_LENGTH" --system_prompt "$SYSTEM_PROMPT" --bf16
fi

# ── 階段 3：influence selection，產出最終訓練集 ──────────────────────────
if [ "$STAGE" -le 3 ]; then
  echo "=== [3/4] D_norm influence selection（top-2 + 丟最低 20%）==="
  python data/select_dnorm.py \
    --train_file "$TRAIN_FILE" --dynamics_dir "$OUT/dynamics" \
    --out "$OUT/train_pref_selected.jsonl" \
    --stats_out "$OUT/influence_scores.jsonl" \
    --corr_method kendall --f on --g neg_if \
    --top_n 2 --downsample_ratio 0.8
fi

# ── 階段 4：正式對齊訓練（1500 步）─────────────────────────────────────
if [ "$STAGE" -le 4 ]; then
  echo "=== [4/4] alignment training：$MAIN_STEPS steps ==="
  python train_prefix.py "${common_args[@]}" \
    --train_file "$FINAL_TRAIN_FILE" \
    --output_dir "$OUT/lora-simpo" \
    --max_steps $MAIN_STEPS --save_steps 300 --save_total_limit 6
fi

cat <<EOF

完成。接著跑評測（eval 腳本用 PeftModel.from_pretrained + disable_adapter，
LoRA 與 prefix 通用，不需要改）：

  python eval/gen_for_icd.py --adapter $OUT/lora-simpo \\
      --instruct_json <PurpleLlama>/CybersecurityBenchmarks/datasets/instruct/instruct.json \\
      --safecoder_only --langs c,cpp,java,javascript,python --num_gen 10 \\
      --out_prefix outputs/icd_repro
  python eval/normalize_responses.py outputs/icd_repro.on.jsonl outputs/icd_repro.off.jsonl
  python eval/run_humaneval.py --adapter $OUT/lora-simpo
EOF
